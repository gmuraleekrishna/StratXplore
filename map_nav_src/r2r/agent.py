import sys
from torch.nn import functional as F
import cv2
import math
import numpy as np
import torch
from models.bev_utils import transfrom3D, bevpos_polar, PointCloud
from models.bev_visualize import draw_ob
# from models.bev_visualize import draw_ob
from models.graph_utils import MAX_DIST, GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad
from torch.nn.utils.rnn import pad_sequence
from utils.ops import pad_tensors, gen_seq_masks

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw
from .agent_utils import panorama_windows, weighted_mean_around_index, compare_paths, calculate_confidence_scores

VIZ = False


def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return dist


class GMapNavAgent(Seq2SeqAgent):

    def __init__(self, args, env, rank=0):
        super().__init__(args, env, rank)
        self.use_weighted_candidate = args.use_weighted_candidate
        self.use_complete_candidate = args.use_complete_candidate

    def _build_model(self):
        self.vln_bert = VLNBert(self.args).to(self.args.device)
        self.critic = Critic(self.args).to(self.args.device)
        # buffer
        self.scanvp_cands = {}

    def _build_projector(self):
        self.map_dim = self.args.map_dim
        self.map_res = self.args.map_res
        self.projector = PointCloud(math.radians(90),
                                    1,
                                    feature_map_height=14,
                                    feature_map_width=14,
                                    map_dim=self.map_dim,
                                    map_res=self.map_res,
                                    world_shift_origin=torch.FloatTensor([0, 0, 0]).to(self.args.device),
                                    z_clip_threshold=0.5,
                                    device=torch.device(self.device))

        self.bev_pos = bevpos_polar(self.map_dim).to(self.args.device)
        self.bev_pos = self.bev_pos[None, :, :, :].expand(self.args.batch_size, -1, -1,
                                                          -1)  # bs x map_dim x map_dim x 3

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().to(self.args.device)
        mask = torch.from_numpy(mask).to(self.args.device)
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):

                cc_img_fts, cc_ang_fts = cc['feature'][:self.args.image_feat_size], cc['feature'][
                                                                                    self.args.image_feat_size:]
                pointId = cc['pointId']
                if self.use_complete_candidate:
                    cand_view_with_neighbours_pointIds = panorama_windows[pointId]
                    cand_with_neighbour_view_img_fts = ob['feature'][cand_view_with_neighbours_pointIds,
                                                       :self.args.image_feat_size].mean(0)
                    view_img_fts.append(cand_with_neighbour_view_img_fts)
                elif self.use_weighted_candidate:
                    cand_with_neighbour_view_img_fts = weighted_mean_around_index(ob['feature'][:,
                                                                                  :self.args.image_feat_size], pointId)
                    view_img_fts.append(cand_with_neighbour_view_img_fts)
                else:
                    view_img_fts.append(cc_img_fts)

                view_ang_fts.append(cc_ang_fts)
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).to(self.args.device)
        batch_loc_fts = pad_tensors(batch_loc_fts).to(self.args.device)
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).to(self.args.device)
        batch_view_lens = torch.LongTensor(batch_view_lens).to(self.args.device)

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids,
        }

    def lift(self, obs):
        '''  unproject rgbs and depths to pointcloud in world coord '''
        bs = len(obs)
        views = 12

        xyzhe = np.zeros([bs, views, 5])
        for i, ob in enumerate(obs):
            x, y, z = ob['position']
            xyzhe[i, :, 0] = x
            xyzhe[i, :, 1] = z
            xyzhe[i, :, 2] = -y
            xyzhe[i, :, 3] = -(np.arange(12) * np.radians(30) + ob['heading'])  # counter-clock
            xyzhe[i, :, 4] = np.pi
        T = transfrom3D(xyzhe.reshape(-1, 5))
        T = torch.from_numpy(T).to(self.args.device)  # bs*views, 4, 4

        depths = np.stack([ob['depth'] for ob in obs], axis=0).reshape(-1, 1, 14, 14)
        depths = torch.from_numpy(depths * 10).to(self.args.device)
        pc, pc_mask = self.projector.forward(depths, T)
        pc = pc.reshape(bs, -1, 3)
        pc_mask = pc_mask.reshape(bs, -1)

        rgbs = np.stack([ob['rgb'] for ob in obs], axis=0).reshape(-1, 14, 14, 768)
        rgbs = torch.from_numpy(rgbs).to(self.args.device)
        pc_feat = rgbs.reshape(bs, -1, 768)

        return pc, pc_mask, pc_feat

    def splat(self, obs, pc, pc_mask, pc_feat):
        '''
            1. transform pointcloud to ego coord
            2. project to bev
        '''
        bs = len(obs)

        S = []
        for i, ob in enumerate(obs):
            x, y, z = ob['position']
            S.append([np.array([x, z, -y])])
        S = np.vstack(S).astype(np.float32)  # bs, 3
        S = torch.from_numpy(S).to(self.args.device)
        xyzhe = np.zeros([bs, 5])
        for i, ob in enumerate(obs):
            xyzhe[i, 3] = ob['heading']
        T = torch.from_numpy(transfrom3D(xyzhe)).to(self.args.device)  # bs, 4, 4

        # transform to ego coord
        pc = pc - S[:, None, :]
        ones = torch.ones(pc.shape[:2]).unsqueeze(-1).to(self.args.device)
        pc1 = torch.cat([pc, ones], dim=-1)  # bs, N, 4
        pc1 = torch.matmul(pc1, T.transpose(1, 2))  # bs, N, 4
        pc = pc1[:, :, :3]  # bs, N, 3

        # project to bev
        viz = False
        bev_fts, bev_masks = self.projector.project_bev(pc, pc_mask, pc_feat)
        if viz:
            feat_masks = []
            for ob, bev_mask in zip(obs, bev_masks):
                cand_pos = self._map_cand_to_bev(ob)
                bev_mask = bev_mask.cpu().numpy()[:, :, None] * np.array([255, 255, 255])[None, None, :]
                bev_mask = bev_mask.astype(np.uint8)
                for p in cand_pos:
                    bev_mask[p[1], p[0], :] = np.array([0, 255, 0]).astype(np.uint8)
                feat_masks.append(bev_mask)
            feat_masks = np.concatenate(feat_masks, axis=1)
            cv2.imwrite('feat_masks.png', feat_masks)

            bev_imgs = [draw_ob(ob) for ob in obs]
            bev_imgs = np.concatenate(bev_imgs, axis=0)
            cv2.imwrite('bev_imgs.png', bev_imgs)

        bev_masks = torch.ones_like(bev_masks)
        bev_fts = bev_fts.reshape(bs, -1, 768)
        bev_masks = bev_masks.reshape(bs, -1)
        bev_pos_fts = self.bev_pos.reshape(bs, -1, 3)

        return bev_fts, bev_masks, bev_pos_fts

    def _nav_gmap_variable(self, obs, gmaps, use_knowledge):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        batch_knowledge = []
        batch_labels = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.nav_graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )  # cuda, zero is stop token

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )
            # knowledge, detects_labels = zip(*[gmap.gather_node_knowledge(vp) for vp in gmap_vpids[1:]])

            # if use_knowledge:
            #     batch_knowledge.append(torch.cat([torch.zeros(1, knowledge[0].shape[-1]), torch.stack(knowledge, 0)], 0))
            #     batch_labels.append(torch.cat([torch.zeros(1, len(detects_labels[0])), torch.tensor(
            #         detects_labels)], 0))

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.nav_graph.distance(gmap_vpids[i], gmap_vpids[j]) / MAX_DIST

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).to(self.args.device)
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).to(self.args.device)
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).to(self.args.device)
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).to(self.args.device)
        # if use_knowledge:
        #     batch_knowledge = pad_sequence(batch_knowledge, batch_first=True)
        #     batch_labels = pad_sequence(batch_labels, batch_first=True)
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.to(self.args.device)

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left, 'detect_feats': batch_knowledge, 'detect_labels': batch_labels
        }

    def _map_cand_to_bev(self, ob, map_dim=None, map_res=None):
        if map_dim is None:
            map_dim = self.map_dim
        if map_res is None:
            map_res = self.map_res

        S = np.array(ob['position'])[None, :].astype(np.float32)  # 1 x 3
        S = S[:, [0, 2, 1]] * np.array([1, 1, -1], dtype=np.float32)  # x, z, -y
        xyzhe = np.zeros([1, 5])
        xyzhe[:, 3] = -ob['heading']
        T = transfrom3D(xyzhe)[0, :, :]  # 4 x 4

        cand_pos = np.array([c['position'] for c in ob['candidate']]).astype(np.float32)
        cand_pos = cand_pos[:, [0, 2, 1]] * np.array([1, 1, -1], dtype=np.float32)  # x, z, -y
        cand_pos = cand_pos - S
        ones = np.ones([cand_pos.shape[0], 1]).astype(np.float32)
        cand_pos1 = np.concatenate([cand_pos, ones], axis=-1)
        cand_pos1 = np.dot(cand_pos1, T.transpose(0, 1))
        cand_pos = cand_pos1[:, :3]
        cand_pos = (cand_pos[:, [0, 2]] / map_res).round() + (map_dim - 1) // 2
        cand_pos[cand_pos < 0] = 0
        cand_pos[cand_pos >= map_dim] = map_dim - 1
        cand_pos = cand_pos.astype(np.int)

        return cand_pos

    def _nav_bev_variable(self, obs, gmaps):
        batch_pc = []
        batch_pc_mask = []
        batch_pc_feat = []

        batch_bev_nav_masks = []
        batch_bev_cand_vpids = []
        batch_bev_cand_idxs = []
        batch_gmap_pos_fts = []
        # max_cand_len = max([len(ob['candidate']) for ob in obs])
        #
        # batch_cand_masks = torch.zeros((len(obs), (max_cand_len + 1)), dtype=torch.bool)

        for i, (ob, gmap) in enumerate(zip(obs, gmaps)):
            vp = ob['viewpoint']
            pc, pc_mask, pc_feat = gmap.gather_node_pc(vp, self.args.pc_order)
            batch_pc.append(pc)
            batch_pc_mask.append(pc_mask)
            batch_pc_feat.append(pc_feat)

            # map candidate to bev
            cands = [c['viewpointId'] for c in ob['candidate']]
            bev_cand_vpids = [None] + cands # stop is None
            # batch_cand_masks[i, :len(cands)+1] = [False] + [True] * len(cands)
            bev_cand_pos = self._map_cand_to_bev(ob)
            bev_cand_idxs = bev_cand_pos[:, 1] * self.map_dim + bev_cand_pos[:, 0]
            bev_cand_idxs = np.insert(bev_cand_idxs, 0,
                                      (self.map_dim * self.map_dim - 1) // 2)  # stop token, the center of BEV
            bev_nav_masks = np.zeros(self.map_dim * self.map_dim).astype(np.bool)
            bev_nav_masks[bev_cand_idxs] = True
            batch_bev_cand_vpids.append(bev_cand_vpids)
            batch_bev_nav_masks.append(torch.from_numpy(bev_nav_masks))
            batch_bev_cand_idxs.append(torch.from_numpy(bev_cand_idxs))
            # global relative pos
            gmap_pos_fts = gmap.get_pos_fts(ob['viewpoint'], [gmap.start_vp], ob['heading'], ob['elevation'])
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))

        # collate
        batch_pc = pad_sequence(batch_pc, batch_first=True)
        batch_pc_mask = pad_sequence(batch_pc_mask, batch_first=True, padding_value=True)  # no depth mask
        batch_pc_feat = pad_sequence(batch_pc_feat, batch_first=True)

        batch_bev_nav_masks = pad_tensors(batch_bev_nav_masks).to(self.args.device)
        batch_bev_cand_idxs = pad_tensors(batch_bev_cand_idxs)
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).expand(-1, self.map_dim * self.map_dim, -1).to(
            self.args.device)

        bev_fts, bev_masks, bev_pos_fts = self.splat(obs, batch_pc, batch_pc_mask, batch_pc_feat)
        bev_pos_fts = torch.cat([batch_gmap_pos_fts, bev_pos_fts], dim=-1)

        return {
            'bev_fts': bev_fts, 'bev_pos_fts': bev_pos_fts,
            'bev_masks': bev_masks, 'bev_nav_masks': batch_bev_nav_masks,
            'bev_cand_idxs': batch_bev_cand_idxs, 'bev_cand_vpids': batch_bev_cand_vpids,
            # 'batch_cand_masks': batch_cand_masks
        }

    def _knowledge_variable(self, obs):
        detect_labels = []
        detect_features = []
        for ob in obs:
            detect_labels.append([di[:3] + [-1] * (3 - len(di[:3])) for di in ob['detect_label'].values()])
            detect_features.append(torch.stack([torch.stack(embed).contiguous().mean(dim=0) for embed in ob[
                'glove_embeds'].values()]))
        return {
            'detect_labels': torch.from_numpy(np.array(detect_labels)),
            'detect_feats': torch.stack(detect_features, 0)
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0  # Stop if arrived
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                   + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).to(self.args.device)

    def _teacher_action_r4r(
            self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0  # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        # try:
                        #     a[i] = vpids[i].index(goal_vp)
                        #     break
                        # except IndexError:
                        #     continue
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0  # Stop if arrived
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan],
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][
                                                                   1:],
                                        ob['gt_path'],
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                           + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).to(self.args.device)

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:  # None is the <stop> action
                traj[i]['path'].append(gmaps[i].nav_graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = 0  # (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, train_pc=None, reset=True, **kwargs):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build nav_graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)

        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        last_curiosity = np.zeros(batch_size, np.float32)
        last_curiosity_diff = np.zeros(batch_size, np.float32)
        # for i, ob in enumerate(obs):  # The init distance from the view point to the target
        #     last_dist[i] = ob['distance']
        #     path_act = [vp[0] for vp in traj[i]['path']]
        #     last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.
        path_loss = 0.
        rewards = []
        policy_log_probs = []
        hidden_states = []

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1
                    candidate_vps = [vp['viewpointId'] for vp in obs[i]['candidate']]
                    gmap.update_step_cand_prob(obs[i]['viewpoint'],
                                               candidate_vps,
                                               [1e-8] * len(candidate_vps), t + 1, causality=False)
            # nav_graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # pointcloud representation
            pc, pc_mask, pc_feat = self.lift(obs)

            # add to gmap
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    gmap.update_node_pc(i_vp, pc[i], pc_mask[i], pc_feat[i])
                    gmap.update_node_knowledge(i_vp, obs[i]['glove_embeds'], obs[i]['detect_label'])
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.nav_graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])
                            gmap.update_node_knowledge(i_cand_vp, obs[i]['neighbour_glove_embeds'][i_cand_vp],
                                                       obs[i]['neighbour_detect_label'][i_cand_vp])
            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps, self.args.use_knowledge)
            nav_inputs.update(self._nav_bev_variable(obs, gmaps))

            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['bev_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            # if self.feedback == 'sample':
            #     gmap_embeds = nav_outs['gmap_embeds']
            #     hidden_states.append(gmap_embeds)
            # update nav_graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }
                    gmap.update_step_cand_prob(traj[i]['path'][t][-1],
                                               nav_inputs['bev_cand_vpids'][i][1:],
                                               [nav_probs[i][nav_inputs['gmap_vpids'][i].index(b)].item()
                                                for b in nav_inputs['bev_cand_vpids'][i][1:]], t + 1)
            if self.args.enable_path_correction and train_pc:
                recovery_confidence_scores = nav_outs['confidence_scores']
                # path_type_logits.masked_fill_(nav_inputs['gmap_visited_masks'][:, :, None].logical_not(), -float('inf'))
                max_action = nav_probs.argmax(-1)
                # path_type_target = -1 * torch.ones((path_type_logits.size(0), path_type_logits.size(1)), dtype=int)
                recovery_target = torch.zeros((nav_outs['local_logits'].shape[0],
                                               nav_outs['local_logits'].shape[1]),
                                              dtype=torch.float16)
                for b in range(batch_size):
                    if not ended[b]:
                        path = sum([p for p in traj[b]['path']], []) + [nav_vpids[b][max_action[b]]]
                        confidence_target = calculate_confidence_scores(obs[b]['gt_path'],
                                                                        nav_inputs['bev_cand_vpids'][b],
                                                                        path,
                                                                        self.env.shortest_distances[obs[b]['scan']]
                                                                        )
                        # path_type_target[b, :len(types)] = torch.tensor(types)[:path_type_logits.shape[1]]
                        recovery_target[b, :len(confidence_target)] = confidence_target
                # path_loss += self.mlce_criterion(path_type_logits[:t+1],
                #                                  path_type_target[:t+1].to(path_type_logits.device))
                mask = recovery_confidence_scores != -float('inf')
                path_loss += self.mse_criterion(recovery_confidence_scores[mask],
                                                recovery_target[mask].float().to(nav_outs['local_logits'].device))
            if train_ml is not None:
                # Supervised training
                if self.args.dataset == 'r2r':
                    # nav_targets = self._teacher_action(
                    #     obs, nav_vpids, ended, 
                    #     visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                    # )
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
                    )
                else:
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
                    )
                # print(t, nav_logits, nav_targets)
                ml_loss += self.criterion(nav_logits, nav_targets)

                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())

            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(nav_logits, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs[
                        'gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            elif self.feedback == 'pc_sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                if self.args.enable_path_correction:
                    recovery_confidence_scores = torch.sigmoid(nav_outs['confidence_scores'])
                    for b in range(batch_size):
                        if ((recovery_confidence_scores[b, 1:len(nav_inputs['bev_cand_vpids'][b]) + 1] <= 0.5).all().item()
                                and t > 1 and not ended[b]):
                            probs_dict = gmaps[b].get_unvisited_node_probs(t)
                            curiosity_dict, max_score = gmaps[b].get_node_curiosity_scores(obs[b]['obj_vg_vico'],
                                                                                           probs_dict)
                            total_importance = {c: p * curiosity_dict[c] for c, p in probs_dict.items()}
                            candidates = list(total_importance.keys())
                            if not candidates: continue
                            probs = np.array(list(total_importance.values()))
                            next_best_cand_id = probs.argmax()
                            a_t[b] = nav_inputs['gmap_vpids'][b].index(candidates[next_best_cand_id])
            elif self.feedback == 'pc_argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                if self.args.enable_path_correction:
                    recovery_confidence_scores = torch.sigmoid(nav_outs['confidence_scores'])
                    for b in range(batch_size):
                        if ((recovery_confidence_scores[b, 1:len(nav_inputs['bev_cand_vpids'][b])] < 0.5).all().item()
                                and t > 1 and not ended[b]):
                            probs_dict = gmaps[b].get_unvisited_node_probs(t)
                            curiosity_dict, max_score = gmaps[b].get_node_curiosity_scores(obs[b]['obj_vg_vico'], probs_dict)
                            total_importance = {c: p * curiosity_dict[c] for c, p in probs_dict.items() if
                                                curiosity_dict[c] > 0.}
                            candidates = list(total_importance.keys())
                            if not candidates:
                                continue
                            probs = np.array(list(total_importance.values()))
                            next_best_cand_id = probs.argmax()
                            frontier = candidates[next_best_cand_id]
                            a_t[b] = nav_inputs['gmap_vpids'][b].index(frontier)
                            if frontier in sum(traj[b]['path'], []):
                                ended[b] = True
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample' or self.feedback == 'pc_sample':  # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        path = gmaps[i].nav_graph.path(obs[i]['viewpoint'], stop_node)
                        traj[i]['path'].append(path)
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update nav_graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                curiosity_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']
                    curiosity_score[i] = gmaps[i].get_node_curiosity_score(ob['viewpoint'])[0].item()
                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            curiosity_reward = curiosity_score[i] - last_curiosity[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward + curiosity_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward + curiosity_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score
                last_curiosity[:] = curiosity_score

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break
            if train_rl:
                pano_inputs = self._panorama_feature_variable(obs)
                pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                  torch.sum(pano_masks, 1, keepdim=True)

                # pointcloud representation
                pc, pc_mask, pc_feat = self.lift(obs)

                # add to gmap
                for i, gmap in enumerate(gmaps):
                    if not ended[i]:
                        # update visited node
                        i_vp = obs[i]['viewpoint']
                        gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                        gmap.update_node_pc(i_vp, pc[i], pc_mask[i], pc_feat[i])
                        gmap.update_node_knowledge(i_vp, obs[i]['glove_embeds'],obs[i]['detect_label'])
                        # update unvisited nodes
                        for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                            if not gmap.nav_graph.visited(i_cand_vp):
                                gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])
                                gmap.update_node_knowledge(i_cand_vp, obs[i]['neighbour_glove_embeds'][i_cand_vp],
                                                           obs[i]['neighbour_detect_label'][i_cand_vp])
                # navigation policy
                nav_inputs = self._nav_gmap_variable(obs, gmaps, self.args.use_knowledge)
                nav_inputs.update(self._nav_bev_variable(obs, gmaps))
                nav_inputs.update({
                    'txt_embeds': txt_embeds,
                    'txt_masks': language_inputs['txt_masks'],
                })
                nav_outs = self.vln_bert('navigation', nav_inputs)

                # The value esti of the last state, remove the #grad  for safety
                last_value__ = self.critic(nav_outs['gmap_embeds']).detach()

                discount_reward = np.zeros(batch_size, np.float32)
                for i, gmap in enumerate(gmaps):
                    if not ended[i]:
                        # i_vp = obs[i]['viewpoint']
                        discount_reward[i] = last_value__[i] #gmap.gather_node_curiosity_node(i_vp)[0].item()

                rl_loss = 0.
                length = len(rewards)
                total = 0
                for t in range(length - 1, -1, -1):
                    discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will
                    # be 0
                    mask_ = torch.from_numpy(masks[t]).cuda()
                    clip_reward = discount_reward[t].copy()
                    r_ = torch.from_numpy(clip_reward).cuda()
                    v_ = self.critic(hidden_states[t])
                    a_ = (r_ - v_).detach()

                    t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                    t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss

                    rl_loss += t_policy_loss + t_critic_loss
                    if self.feedback == 'sample':
                        rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                    self.logs['critic_loss'].append(t_critic_loss.item())
                    self.logs['policy_loss'].append(t_policy_loss.item())

                    total = total + np.sum(masks[t])
                self.logs['total'].append(total)

                # Normalize the loss function
                if self.args.normalize_loss == 'total':
                    rl_loss /= total
                elif self.args.normalize_loss == 'batch':
                    rl_loss /= batch_size
                else:
                    assert self.args.normalize_loss == 'none'

                self.loss += rl_loss
                self.logs['RL_loss'].append(rl_loss.item())  # critic loss + policy loss + entropy loss

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())
        if train_pc is not None and self.args.enable_path_correction:
            pc_loss = train_pc * path_loss / batch_size
            self.loss += pc_loss
            self.logs['PC_loss'].append(pc_loss.item())
        return traj
