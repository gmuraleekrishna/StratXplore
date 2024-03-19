'''
Instruction and trajectory (view and object features) dataset
'''
import copy
import json
import random

import cv2
import h5py
import jsonlines
import math
import networkx as nx
import numpy as np
from model.bev_utils import transfrom3D

from .common import calculate_vp_rel_pos_fts, wrapped_window_indices, weighted_mean_around_index
from .common import get_angle_fts, get_view_rel_angles
from .common import load_nav_graphs
from .common import softmax
from model.bev_visualize import draw_state

MAX_KIDNAP_STEPS = 6

# from ..model.bev_visualize import draw_state

# from model.bev_visualize import draw_state

MAX_DIST = 30  # normalize
MAX_STEP = 10  # normalize
TRAIN_MAX_STEP = 20
ANCHOR_H = np.radians(np.linspace(0, 330, 12))
MP3D_CAT = 40


def nearest_anchor(query, anchors):
    cos_dis = np.cos(query) * np.cos(anchors) + np.sin(query) * np.sin(anchors)
    nearest = np.argmax(cos_dis)
    return nearest


class ReverieTextPathData(object):
    def __init__(
            self, anno_files, img_ft_file, obj_ft_file, rgb_file, depth_file, sem_file,
            scanvp_cands_file, connectivity_dir,
            image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
            obj_feat_size=None, obj_prob_size=None, max_objects=20,
            max_txt_len=100, in_memory=True, act_visited_node=False,
            val_sample_num=None, args=None,
    ):
        self.img_ft_file = img_ft_file
        self.obj_ft_file = obj_ft_file
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.sem_file = sem_file

        self.use_complete_candidate = args.use_complete_candidate
        self.use_weighted_candidate = args.use_weighted_candidate

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.bev_dim = 21
        self.bev_res = 0.5

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}

        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in
                                    self.all_point_rel_angles]

        self.perturbator = Perturbation(self.graphs, self.shortest_distances, self.shortest_paths, self.scanvp_cands,
                                        args, store_cache=True)
        self.data = []
        for anno_file in anno_files:
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    self.data.append(item)

        if val_sample_num:
            # cannot evaluate all the samples as it takes too much time
            sel_idxs = np.random.permutation(len(self.data))[:val_sample_num]
            self.data = [self.data[sidx] for sidx in sel_idxs]
        self.data = self.data[:int(len(self.data) * args.select_pct)]

    def __len__(self):
        return len(self.data)

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)

            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size + self.obj_prob_size), dtype=np.float32)
            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.max_objects]
            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_scanvp_grid_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        with h5py.File(self.rgb_file, 'r') as f:
            rgbs = f[key][...].astype(np.float32)
        with h5py.File(self.depth_file, 'r') as f:
            depths = f[key][...].astype(np.float32)
        with h5py.File(self.sem_file, 'r') as f:
            sems = f[key][...].astype(np.uint8)
        return rgbs, depths, sems

    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            # it occurs when the gt_objid is not in max_objects
            obj_label = -100  # ignore
            # print('No groundtruth obj_id', item['instr_id'], len(obj_ids))
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k  # [stop] is 0
            # local: 
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                                + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1  # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False,
            return_obj_label=False, end_vp=None, kidnap=False
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        pos_vps = item['pos_vps']
        gt_path = item['path']
        path_types = None

        if end_vp is None:
            if end_vp_type == 'pos':
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]
        if kidnap:
            gt_path, path_types, kidnap_at = self.perturbator.generate(scan, gt_path, idx=f"{scan}_{idx}")
            end_vp = gt_path[-1]
        else:
            gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            if path_types:
                path_types = path_types[:TRAIN_MAX_STEP] + [0]

        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        rgbs, depths, sems, T_c2w, T_w2c, S_w2c, bev_cand_idxs = \
            self.get_bev_inputs(scan, end_vp, cur_heading, cur_elevation, traj_cand_vpids[-1])
        bev_gpos_fts = self.get_gmap_pos_fts(scan, end_vp, [start_vp], cur_heading, cur_elevation)

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],

            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'path_types': path_types,

            'rgbs': rgbs,  # (12, 196, 768)
            'depths': depths,  # (12, 14, 14)
            'sems': sems,  # (12*14*14, 40)
            'T_c2w': T_c2w,  # (12, 4, 4)
            'T_w2c': T_w2c,  # (1, 4, 4)
            'S_w2c': S_w2c,  # (1, 3)
            'bev_cand_idxs': bev_cand_idxs,  # (K, )
            'bev_gpos_fts': bev_gpos_fts,  # (1, 7)
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            # outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s' % (scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            elevation = 0
            # elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation

    def get_path_viewidxs(self, scan, path, start_heading):
        viewidxs = [nearest_anchor(start_heading, ANCHOR_H) + 12]  # init viewidx
        for s, e in zip(path[:-1], path[1:]):
            print(scan, s)
            viewidx = self.scanvp_cands[f'%s_%s' % (scan, s)][e][0] % 12 + 12
            viewidxs.append(viewidx)
        return viewidxs

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []

        for vp in path:
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s' % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                pointId = v[0]
                used_viewidxs.add(v[0])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][pointId]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
                if self.use_complete_candidate:
                    look_down_pointId = pointId % 12
                    # look_down_opposite_pointId = (look_down_pointId + 6) % 12
                    cand_view_with_neighbours_pointIds = wrapped_window_indices[pointId]
                    cand_with_neighbour_view_img_fts = view_fts[cand_view_with_neighbours_pointIds].mean(0)
                    view_img_fts.append(cand_with_neighbour_view_img_fts)
                else:
                    view_img_fts.append(view_fts[pointId])

            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)

            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h / self.obj_image_h, w / self.obj_image_w, (h * w) / self.obj_image_size]
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids

    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[t + 1] = vp
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s' % (scan, vp)].keys():
                if next_vp not in list(visited_vpids.values()):
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.values()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.keys()) + list(unvisited_vpids.values())
        if self.act_visited_node:
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)

        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i + 1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]] / MAX_DIST

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists

    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'],
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                     (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)

        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len + 1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts

        return vp_pos_fts

    def get_bev_inputs(self, scan, cur_vp, cur_heading, cur_elevation, cand_vpids):
        assert cur_elevation == 0

        x, y, z = self.graphs[scan].nodes[cur_vp]['position'][:3]
        rgbs, depths, sems = self.get_scanvp_grid_feature(scan, cur_vp)  # idx0: view 12
        sems = np.eye(MP3D_CAT)[sems.flatten()]

        # camera to world
        xyzhe = np.zeros([12, 5]).astype(np.float32)
        xyzhe[:, 0] = x
        xyzhe[:, 1] = z
        xyzhe[:, 2] = -y
        xyzhe[:, 3] = -np.arange(12) * np.radians(30)  # counter-clock
        xyzhe[:, 4] = np.pi
        T_c2w = transfrom3D(xyzhe)

        # world to camera
        S_w2c = xyzhe[:1, :3].copy()
        xyzhe = np.zeros([1, 5]).astype(np.float32)
        xyzhe[:, 3] = cur_heading
        T_w2c = transfrom3D(xyzhe)

        # cand idxs in bev
        S_cand = S_w2c[0]
        xyzhe = np.zeros([1, 5]).astype(np.float32)
        xyzhe[:, 3] = -cur_heading
        T_cand = transfrom3D(xyzhe)[0]

        cand_pos = np.array([self.graphs[scan].nodes[vp]['position'] for vp in cand_vpids]).astype(np.float32)
        cand_pos = cand_pos[:, [0, 2, 1]] * np.array([1, 1, -1], dtype=np.float32)  # x, z, -y
        cand_pos = cand_pos - S_cand
        ones = np.ones([cand_pos.shape[0], 1]).astype(np.float32)
        cand_pos1 = np.concatenate([cand_pos, ones], axis=-1)
        cand_pos1 = np.dot(cand_pos1, T_cand.transpose(0, 1))
        cand_pos = cand_pos1[:, :3]
        cand_pos = (cand_pos[:, [0, 2]] / self.bev_res).round() + (self.bev_dim - 1) // 2
        cand_pos[cand_pos < 0] = 0
        cand_pos[cand_pos >= self.bev_dim] = self.bev_dim - 1
        cand_pos = cand_pos.astype(np.long)

        bev_cand_idxs = cand_pos[:, 1] * self.bev_dim + cand_pos[:, 0]
        bev_cand_idxs = np.insert(bev_cand_idxs, 0,
                                  (self.bev_dim * self.bev_dim - 1) // 2)  # stop token, the center of BEV

        return rgbs, depths, sems, T_c2w, T_w2c, S_w2c, bev_cand_idxs


class R2RTextPathData(ReverieTextPathData):
    def __init__(
            self, anno_files, img_ft_file, rgb_file, depth_file, sem_file,
            scanvp_cands_file, connectivity_dir,
            image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
            max_txt_len=100, in_memory=True, act_visited_node=False,
            val_sample_num=None, args=None
    ):
        super().__init__(
            anno_files, img_ft_file, None, rgb_file, depth_file, sem_file,
            scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0,
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node, val_sample_num=val_sample_num, args=args
        )


    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store[key] = view_fts
        return view_fts

    def get_act_labels(self, end_vp, end_idx, item, gmap_vpids, traj_cand_vpids):
        if end_vp == item['path'][-1]:  # stop
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item['path'][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1  # [stop] is 0
                    break
        return global_act_label, local_act_label

    def get_input(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None, kidnap=False
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']
        path_types = None
        if end_vp is None:
            if end_vp_type == 'pos':
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        if kidnap:
            gt_path, path_types, kidnap_at = self.perturbator.generate(scan, gt_path, idx=f"{scan}_{idx}")
            end_vp = gt_path[-1]
        else:
            gt_path = self.shortest_paths[scan][start_vp][end_vp]

        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)
        # path_viewidxs = self.get_path_viewidxs(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            if path_types:
                path_types = path_types[:TRAIN_MAX_STEP] + [0]

        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles = self.get_traj_pano_fts(scan, gt_path, path_viewidxs=None)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # if kidnap:
        #     for vp, visited in zip(gmap_vpids[1:], gmap_visited_masks[1:]):
        #         if visited != 1:
        #             dist = min([self.shortest_distances[scan][vp][try_vp] for try_vp in gt_path[kidnap_at+1:]])
        #             confidence = (MAX_DIST - dist) / MAX_DIST
        #             recovery_confidence.append(confidence)

        # local: the first token is [stop]
        rgbs, depths, sems, T_c2w, T_w2c, S_w2c, bev_cand_idxs = \
            self.get_bev_inputs(scan, end_vp, cur_heading, cur_elevation, traj_cand_vpids[-1])
        bev_gpos_fts = self.get_gmap_pos_fts(scan, end_vp, [start_vp], cur_heading, cur_elevation)

        viz = False
        if viz:
            bev_imgs = draw_state(scan, end_vp, cur_heading)
            bev_masks = np.zeros(self.bev_dim * self.bev_dim).astype(np.uint8)
            bev_masks[bev_cand_idxs] = 255
            bev_masks = bev_masks.reshape(self.bev_dim, self.bev_dim)
            cv2.imwrite('tmp1.png', bev_imgs)
            cv2.imwrite('tmp2.png', bev_masks)


        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],

            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'path_types': path_types,

            'rgbs': rgbs,  # (12, 196, 768)
            'depths': depths,  # (12, 14, 14)
            'sems': sems,  # (12*14*14, 40)
            'T_c2w': T_c2w,  # (12, 4, 4)
            'T_w2c': T_w2c,  # (1, 4, 4)
            'S_w2c': S_w2c,  # (1, 3)
            'bev_cand_idxs': bev_cand_idxs,  # (K, )
            'bev_gpos_fts': bev_gpos_fts,  # (1, 7)
        }

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)

        return outs

    def get_traj_pano_fts(self, scan, path, path_viewidxs=None):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], []

        for t, vp in enumerate(path):
            view_fts = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s' % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():  # donot include cur vp
                pointId = v[0]
                used_viewidxs.add(pointId)
                # TODO: whether using correct heading at each step
                if path_viewidxs is None:
                    view_angle = self.all_point_rel_angles[12][v[0]]
                else:
                    cur_viewidx = path_viewidxs[t]
                    view_angle = self.all_point_rel_angles[cur_viewidx][v[0]]
                if self.use_complete_candidate:
                    cand_view_with_neighbours_pointIds = wrapped_window_indices[pointId]
                    cand_with_neighbour_view_img_fts = view_fts[cand_view_with_neighbours_pointIds].mean(0)
                    view_img_fts.append(cand_with_neighbour_view_img_fts)
                elif self.use_weighted_candidate:
                    cand_with_neighbour_view_img_fts = weighted_mean_around_index(view_fts, pointId)
                    view_img_fts.append(cand_with_neighbour_view_img_fts)
                else:
                    view_img_fts.append(view_fts[pointId])

                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_loc_fts.append(np.concatenate([view_ang_fts, view_box_fts], 1))
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))
            traj_cand_vpids.append(cand_vpids)

            last_vp_angles = view_angles

        return traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles


class Perturbation:
    def __init__(self, graphs, shortest_distances, shortest_paths, scanvp_cands, args, store_cache=False):
        self.kidnap_bias = ['first', 'rand', 'last']
        self.kidnap_type = ['seen', 'unseen']
        self.kidnap_node_location = ['gt_inline', 'gt_close']
        self.kidnap_nodes = args.kidnap_nodes
        self.kidnap_turn_ang = random.uniform(1, 2 * math.pi)
        self.kidnap_away_distance = args.kidnap_away_distance
        self.kidnap_prob = np.random.uniform(0.2, 0.8)
        self.graphs = graphs
        self.node_closeness = {}
        self.scan_nodes = {}
        for scan, graph in graphs.items():
            dists = nx.floyd_warshall_numpy(graph)
            mask = np.zeros_like(dists, dtype=np.bool)
            mask[np.where(dists < self.kidnap_away_distance)] = True
            self.node_closeness[scan] = dists[mask]
            self.scan_nodes[scan] = np.array(graph.nodes())[np.any(mask, axis=0)].tolist()
        self.shortest_paths = shortest_paths
        self.node_neigbours = scanvp_cands
        self.shortest_distances = shortest_distances
        self.store_cache = store_cache
        self._cache = {}

    def vertices_within_distance_excluding_neighbors(self, scan, nodes_list, d):
        """
        Find vertices in the nav_graph G whose distance to any vertex in nodes_list is less than d,
        excluding the nodes in nodes_list and their direct neighbors.

        :param G: A networkx nav_graph.
        :param nodes_list: A list of target vertices.
        :param d: The distance threshold.
        :return: A list of vertices within distance d from any vertex in nodes_list, excluding the nodes in nodes_list and their neighbors.
        """
        # Compute the shortest path distances using Floyd-Warshall
        distance_matrix = self.node_closeness[scan]

        # Convert nav_graph nodes to a list for indexing

        # Find indices of nodes in nodes_list
        nodes_indices = {self.scan_nodes[scan].index(v) for v in nodes_list if v in self.scan_nodes[scan]}

        # Gather all neighbors of nodes in nodes_list
        neighbors_indices = {self.scan_nodes[scan].index(neighbour) for node in nodes_list
                             for neighbour in self.node_neigbours[f"{scan}_{node}"]}

        # Create a boolean mask for all vertices close to any node in nodes_list
        close_mask = np.zeros_like(distance_matrix, np.bool)
        close_mask[np.where(distance_matrix[list(nodes_indices)] < d)] = True

        # Exclude nodes in nodes_list and their neighbors from the mask
        all_excluded_indices = nodes_indices.union(neighbors_indices)
        close_mask[list(all_excluded_indices)] = False

        close_mask = close_mask[:len(self.scan_nodes[scan])]
        # Extract the list of close vertices
        close_vertices = np.array(self.scan_nodes[scan])[close_mask].tolist()

        return close_vertices

    def generate(self, scan, gt_path=None, idx=None):
        # if idx in self._cache:
        #     return self._cache[idx]
        oracle_path = copy.deepcopy(gt_path)
        rand_type = random.choice(self.kidnap_type)
        rand_node_dist = np.random.uniform(1, self.kidnap_away_distance)
        rand_node_location_type = random.choice(self.kidnap_node_location)
        rand_bias = random.choice(self.kidnap_bias)
        gt_path_len = len(oracle_path)
        kidnapped_path = []
        should_perturbed = np.random.uniform(0, 1) > 0.3
        if should_perturbed:
            if rand_bias == 'first':
                curr_index = int(random.uniform(0.2, 0.5) * gt_path_len)
            elif rand_bias == 'last':
                curr_index = int(random.uniform(0.51, 0.8) * gt_path_len)
            else:
                curr_index = int(random.uniform(0.2, 0.8) * gt_path_len)
            if curr_index <= 1:
                curr_index = 2
            rand_node_count = np.random.randint(1, curr_index)

            if len(oracle_path) <= 6:
                curr_index = 2
                rand_type = 'seen'
                rand_node_location_type = 'gt_inline'

            if len(gt_path) >= TRAIN_MAX_STEP * 0.8:
                perturbed_path = gt_path
                path_types = [0] * len(gt_path)
                if self.store_cache:
                    self._cache[idx] = perturbed_path, path_types
                return perturbed_path, path_types
            if self.kidnap_type == 'turn':
                # headings = [(headings[0] + self.kidnap_turn_ang) % 2 * np.pi]
                pass
            elif rand_type == 'seen':
                if rand_node_location_type == 'gt_inline':
                    new_index = np.clip(1, curr_index - rand_node_count, gt_path_len - 1)
                    kidnap_to_vp = oracle_path[new_index]
                    kidnapped_path = self.shortest_paths[scan][oracle_path[curr_index]][kidnap_to_vp]
                else:  # lif rand_node_location_type == 'gt_close':
                    neighbors_set = set()

                    # Collect neighbors of each node in vertex_list
                    for node in oracle_path:
                        neighbors_set.update(list(self.node_neigbours[f"{scan}_{node}"].keys()))

                    # Exclude nodes in vertex_list from the neighbors set
                    neighbors_set.difference_update(set(oracle_path))

                    # Select one neighbor randomly
                    kidnap_to_vp = random.choice(list(neighbors_set))
                    kidnapped_path = self.shortest_paths[scan][oracle_path[curr_index]][kidnap_to_vp]
            else:  # rand_type == 'unseen':
                if rand_node_location_type == 'gt_inline':
                    start_node = random.choice(oracle_path[1:int(len(oracle_path) * 0.4)])
                    end_node = random.choice(oracle_path[int(len(oracle_path) * 0.6):-1])
                    kidnapped_path = self.shortest_paths[scan][oracle_path[curr_index]][start_node]
                    current_node = start_node

                    i = 1
                    while current_node != end_node and i < TRAIN_MAX_STEP / 3:
                        neighbors = list(self.node_neigbours[f"{scan}_{current_node}"].keys())

                        # Randomly select a neighbor to walk to
                        current_node = random.choice(neighbors)

                        # Add the selected neighbor to the path
                        kidnapped_path.append(current_node)
                        i += 1
                    kidnapped_path.extend(self.shortest_paths[scan][kidnapped_path[-1]][end_node][1:])
                else:  # if rand_node_location_type == 'gt_close':
                    close_nodes = self.vertices_within_distance_excluding_neighbors(scan, oracle_path, rand_node_dist)
                    if len(close_nodes) > 0:
                        kidnap_to_vp = random.choice(close_nodes)
                    else:
                        new_index = np.clip(1, curr_index - rand_node_count, gt_path_len - 1)
                        kidnap_to_vp = oracle_path[new_index]
                    kidnapped_path = self.shortest_paths[scan][oracle_path[curr_index]][kidnap_to_vp]

            if len(kidnapped_path) > TRAIN_MAX_STEP / 2:
                kidnapped_path = kidnapped_path[:5]
            recovery_path = kidnapped_path[::-1]
            path_types = ([0] * (curr_index + 1)
                          + [2] * (len(kidnapped_path) - 1)
                          + [1] * (len(recovery_path) - 1)
                          + [0] * (gt_path_len - curr_index - 1))
            perturbed_path = (oracle_path[:curr_index + 1]
                              + kidnapped_path[1:]
                              + recovery_path[1:]
                              + oracle_path[curr_index + 1:])
        else:
            perturbed_path = gt_path
            path_types = [0] * len(gt_path)
            curr_index = 0
        # if self.store_cache:
        #     self._cache[idx] = perturbed_path, path_types
        return perturbed_path, path_types, curr_index


class SoonTextPathData(ReverieTextPathData):
    def __init__(
            self, anno_files, img_ft_file, obj_ft_file, rgb_file, depth_file, sem_file,
            scanvp_cands_file, connectivity_dir,
            image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
            obj_feat_size=None, obj_prob_size=None, max_objects=20,
            max_txt_len=100, in_memory=True, act_visited_node=False, args=None
    ):
        super().__init__(
            anno_files, img_ft_file, obj_ft_file, rgb_file, depth_file, sem_file,
            scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=obj_feat_size,
            obj_prob_size=obj_prob_size, max_objects=max_objects,
            max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node, args=args
        )
        self.obj_image_h = self.obj_image_w = 600
        self.obj_image_size = 600 * 600

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)

            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size + self.obj_prob_size), dtype=np.float32)
            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.max_objects]
                        obj_attrs['bboxes'] = np.array(obj_attrs['bboxes']).astype(np.float32)
                        obj_attrs['sizes'] = np.zeros((len(obj_attrs['bboxes']), 2), dtype=np.float32)
                        obj_attrs['sizes'][:, 0] = obj_attrs['bboxes'][:, 2] - obj_attrs['bboxes'][:, 0]
                        obj_attrs['sizes'][:, 1] = obj_attrs['bboxes'][:, 3] - obj_attrs['bboxes'][:, 1]
            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        obj_label = item['obj_pseudo_label']['idx']
        if obj_label >= self.max_objects:
            obj_label = -100
        return obj_label

    def get_input(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False,
            return_obj_label=False, end_vp=None, kidnap=False
    ):
        if end_vp_type == 'pos':
            end_vp = self.data[idx]['path'][-1]
        return super().get_input(
            idx, end_vp_type,
            return_img_probs=return_img_probs,
            return_act_label=return_act_label,
            return_obj_label=return_obj_label,
            end_vp=end_vp,
            kidnap=kidnap
        )
