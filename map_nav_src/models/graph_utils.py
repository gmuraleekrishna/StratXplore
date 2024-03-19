from collections import defaultdict

import math
import networkx as nx
import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

MAX_DIST = 30
MAX_STEP = 10


def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return dist


def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx ** 2 + dy ** 2), 1e-8)
    xyz_dist = max(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx / xy_dist)  # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist


def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda: defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda: defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return k in self._visited

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":  # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


def normalize(prob, dim=0):
    # logits: (n, d)
    # tmp = np.exp(logits)
    return prob / np.sum(prob, axis=dim, keepdims=True)


class BayesianNavigationGraph:
    def __init__(self, forget_method, lambda_value):
        self._dg = nx.DiGraph()  # bayesian_graph
        self._visited = set()
        self._forget_fn = lambda x, max_x: math.exp((x - max_x + 1) * lambda_value) \
            if forget_method == 'exp' else (x / (max_x - 1) * lambda_value)
        self._current_node = None

    def has_node(self, node):
        return self._dg.has_node(node)

    def visit(self, node, step):
        self._current_node = node
        nx.set_node_attributes(self._dg, {node: {'step': step}})
        for u, v in self._dg.in_edges(node):
            nx.set_edge_attributes(self._dg, {(v, u): {'weight': 1e-8}})

        self._visited.add(node)
        for vp, neigh in self._dg.out_edges(node):
            if not self.is_visited(neigh):
                nx.set_node_attributes(self._dg, {neigh: {'step': step}})

    def is_visited(self, node):
        return node in self._visited

    def add_node(self, node):
        self._dg.add_node(node)

    def add_edge(self, u, v, transition_prob=0.):
        if self._dg.has_edge(u, v):
            nx.set_edge_attributes(self._dg, {(u, v): {'weight': transition_prob}})
        else:
            self._dg.add_edge(u, v, weight=transition_prob)

    def get_unvisited_prior_probs(self):
        return {vp2: prob for vp1, vp2, prob in self._dg.edges(data='weight') if vp2 not in self._visited}

    # def get_unvisited_edge_exp_decayed_weights(self, steps_upto=100000):
    #     decayed_weights = defaultdict(lambda: 0.0)
    #     for vp1, neigh, prob in self._dg.edges(data='weight'):
    #         t = nx.get_node_attributes(self._dg, 'step')[vp1]
    #         if 0 < t < steps_upto:
    #             decayed_weights[neigh] = self._forget_fn(-(steps_upto - t)) + decayed_weights[neigh]
    #     return decayed_weights

    def complete_graph_prior_state(self):
        graph_state = {}
        total_prob = sum(prob for (u, v, prob) in self._dg.edges(data='weight'))
        for vp, nei in self._dg.edges():
            graph_state[(vp, nei)] = self._dg[vp][nei]['weight'] / total_prob
        return graph_state

    def get_state(self, steps_upto=None):
        graph_state = defaultdict(lambda: 0.0)
        for vp, neigh, prob in self._dg.edges(data='weight'):
            # if not self.is_visited(neigh):
            t = nx.get_node_attributes(self._dg, 'step')[vp]
            decayed_weights = 0.0
            if 0 < t < steps_upto:
                decayed_weights = self._forget_fn(t, steps_upto)
            graph_state[neigh] += prob * decayed_weights  # * in_degree

        total_prob = sum(graph_state.values())
        for vp, prob in graph_state.items():
            graph_state[vp] = prob / total_prob if total_prob != 0 else 0
        return graph_state


class GraphMap(object):
    def __init__(self, start_vp, forget_method='exp', lambda_value=0.1, novel_weight=0.5):
        self.start_vp = start_vp  # start viewpoint
        self.max_unique_objs = 5
        self.node_positions = {}  # viewpoint to position (x, y, z)
        self.nav_graph = FloydGraph()  # shortest path nav_graph
        self.bayesian_graph = BayesianNavigationGraph(lambda_value=lambda_value, forget_method=forget_method)
        self.node_embeds = {}  # {viewpoint: feature (sum feature, count)}
        self.node_pc = {}  # {viewpoint: (vp_pc, pc_mask, pc_feat)}
        self.node_stop_scores = {}  # {viewpoint: prob}
        self.node_nav_scores = {}  # {viewpoint: {t: prob}}
        self.node_step_ids = {}
        self.step_cand_probs = {}
        self.node_knowledge = {}
        self.novel_weight = novel_weight
        self.all_objects = set()

    def update_step_cand_prob(self, vp, candidates, probs, t, causality=True):
        self.bayesian_graph.add_node(vp)
        for nei, pr in zip(candidates, probs):
            if nei is not None:
                if not self.bayesian_graph.is_visited(nei):
                    self.bayesian_graph.add_node(nei)
                    self.bayesian_graph.add_edge(vp, nei, transition_prob=pr)
                elif not self.bayesian_graph.has_node(nei):
                    self.bayesian_graph.add_edge(vp, nei, transition_prob=1e-8)
        self.bayesian_graph.visit(vp, step=t)

    def update_graph(self, ob):
        self.node_positions[ob['viewpoint']] = ob['position']
        for cc in ob['candidate']:
            self.node_positions[cc['viewpointId']] = cc['position']
            dist = calc_position_distance(ob['position'], cc['position'])
            self.nav_graph.add_edge(ob['viewpoint'], cc['viewpointId'], dist)
        self.nav_graph.update(ob['viewpoint'])

    def get_unvisited_node_probs(self, step):
        return self.bayesian_graph.get_state(steps_upto=step)

    def update_node_embed(self, vp, embed, rewrite=False):
        if rewrite:
            self.node_embeds[vp] = [embed, 1]
        else:
            if vp in self.node_embeds:
                self.node_embeds[vp][0] += embed
                self.node_embeds[vp][1] += 1
            else:
                self.node_embeds[vp] = [embed, 1]

    def update_node_knowledge(self, vp, embed, labels, rewrite=False):
        unique_objects = labels - self.all_objects
        if rewrite:
            self.node_knowledge[vp] = [embed, 1, unique_objects]
        else:
            if vp in self.node_knowledge:
                if len(unique_objects) > 0:
                    self.node_knowledge[vp][0] += embed
                self.node_knowledge[vp][1] += 1
                self.node_knowledge[vp][2].update(unique_objects)
            else:
                self.node_knowledge[vp] = [embed, 1, unique_objects]
            self.all_objects.update(unique_objects)

    def update_node_pc(self, vp, pc, pc_mask, pc_feat):
        self.node_pc[vp] = [pc, pc_mask, pc_feat]

    def gather_node_pc(self, vp, order):
        ''' gather vp_pc from adjcent vp '''
        if order == 0:
            return self.node_pc[vp]
        else:
            cvps = [cvp for cvp in self.node_pc.keys() if len(self.nav_graph.path(vp, cvp)) <= order]
            pc = [self.node_pc[cvp][0] for cvp in cvps]
            pc_mask = [self.node_pc[cvp][1] for cvp in cvps]
            pc_feat = [self.node_pc[cvp][2] for cvp in cvps]

            pc = torch.cat(pc, dim=0)
            pc_mask = torch.cat(pc_mask, dim=0)
            pc_feat = torch.cat(pc_feat, dim=0)

            return pc, pc_mask, pc_feat

    def gather_node_knowledge(self, vp):
        return (self.node_knowledge[vp][0],
                list(self.node_knowledge[vp][2])[:self.max_unique_objs])

    def get_node_curiosity_score(self, vp):
        others_embedding = torch.vstack([_[0] for vp1, _ in self.node_knowledge.items() if vp1 != vp]).sum(0)
        score = torch.norm(self.node_knowledge[vp][0] - others_embedding)
        return score, list(self.node_knowledge[vp][2])

    def get_path_knowledge_between(self, start_vp, end_vp):
        path_knowledge = [self.node_knowledge[start_vp][0]]
        path = self.nav_graph.path(start_vp, end_vp)
        for v in path:
            path_knowledge.append(self.node_knowledge[v][0])
        return path_knowledge

    def get_node_curiosity_scores(self, instr_ob_embedd, probs_dict):
        scores = defaultdict(lambda: 0.)
        max_score = 0
        start_vp = [vp for vp, t in self.node_step_ids.items() if t == 1][0]
        for vp in self.node_knowledge.keys():
            if probs_dict.get(vp, 0) > 0.1:
                others_embedding = torch.vstack([_[0] for vp1, _ in self.node_knowledge.items() if vp1 != vp]).sum(0)
                similarity_score = torch.cosine_similarity(self.node_knowledge[vp][0].unsqueeze(0),
                                                           others_embedding.unsqueeze(0))
                path_know = self.get_path_knowledge_between(start_vp, vp)
                dtw_distance = 1000
                if len(path_know) == 0 or len(instr_ob_embedd) == 0:
                    dtw_distance = 1000
                else:
                    try:
                        dtw_distance = fastdtw([p.numpy() for p in path_know],
                                               [p.numpy() for p in instr_ob_embedd],
                                               dist=euclidean)[0]
                    except ValueError as e:
                        print(path_know)
                scores[vp] = (math.exp(-dtw_distance / ((len(path_know) * len(instr_ob_embedd)) + 1)) + 1 /
                              similarity_score.item())
                max_score += scores[vp]
        return scores, max_score

    def get_node_embed(self, vp):
        return self.node_embeds[vp][0] / self.node_embeds[vp][1]

    def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #        line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:  # stop
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.node_positions[cur_vp], self.node_positions[vp],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.nav_graph.distance(cur_vp, vp) / MAX_DIST, \
                     len(self.nav_graph.path(cur_vp, vp)) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def save_to_json(self):
        nodes = {}
        for vp, pos in self.node_positions.items():
            nodes[vp] = {
                'location': pos,  # (x, y, z)
                'visited': self.nav_graph.visited(vp),
            }
            if nodes[vp]['visited']:
                nodes[vp]['stop_prob'] = self.node_stop_scores[vp]['stop']
                nodes[vp]['og_objid'] = self.node_stop_scores[vp]['og']
            else:
                nodes[vp]['nav_prob'] = self.node_nav_scores[vp]

        edges = []
        for k, v in self.nav_graph._dis.items():
            for kk in v.keys():
                edges.append((k, kk))

        return {'nodes': nodes, 'edges': edges}


import unittest


class TestBayesianNavigationGraph(unittest.TestCase):
    def setUp(self):
        # Common setup for the tests
        self.graph_exp = BayesianNavigationGraph('exp', 0.1)
        self.graph_linear = BayesianNavigationGraph('linear', 0.1)

    def test_init(self):
        # Test initialization
        self.assertIsInstance(self.graph_linear._dg, nx.DiGraph)
        self.assertEqual(self.graph_exp._forget_fn(1, 2), 1.0)
        self.assertEqual(self.graph_linear._forget_fn(1, 2), 0.1)

    def test_add_node(self):
        # Test adding nodes
        self.graph_linear.add_node('A')
        self.assertIn('A', self.graph_linear._dg.nodes)

    def test_add_edge(self):
        # Test adding edges
        self.graph_linear.add_edge('A', 'B', 0.2)
        self.assertTrue(self.graph_linear._dg.has_edge('A', 'B'))
        self.assertEqual(self.graph_linear._dg['A']['B']['weight'], 0.2)

    def test_visit(self):
        # Test visiting nodes
        self.graph_linear.add_node('A')
        self.graph_linear.visit('A', 1)
        self.assertIn('A', self.graph_linear._visited)
        self.assertEqual(self.graph_linear._dg.nodes['A']['step'], 1)

    def test_is_visited(self):
        # Test checking if a node has been visited
        self.graph_linear.add_node('A')
        self.graph_linear.visit('A', 1)
        self.assertTrue(self.graph_linear.is_visited('A'))
        self.assertFalse(self.graph_linear.is_visited('B'))

    def test_get_unvisited_prior_probs(self):
        # Test calculating unvisited prior probabilities
        self.graph_linear.add_edge('A', 'B', 0.1)
        self.graph_linear.visit('A', 1)
        expected_probs = {'B': 0.1}
        self.assertEqual(self.graph_linear.get_unvisited_prior_probs(), expected_probs)

    def test_complete_graph_prior_state(self):
        # Test calculating complete graph prior state
        self.graph_linear.add_edge('A', 'B', 0.1)
        self.graph_linear.add_edge('A', 'C', 0.2)
        self.graph_linear.add_edge('B', 'C', 0.7)
        state = self.graph_linear.complete_graph_prior_state()
        total_prob = 0.1 + 0.2 + 0.7
        expected_state = {('A', 'B'): 0.1 / total_prob, ('A', 'C'): 0.2 / total_prob, ('B', 'C'): 0.7 / total_prob}
        self.assertEqual(state, expected_state)

    def test_get_state_exp(self):
        # Test getting the state with decayed weights for unvisited nodes
        curr_step = 1
        self.graph_linear.add_node('A')
        self.graph_linear.add_node('B')
        self.graph_linear.add_node('C')
        self.graph_linear.add_node('D')
        self.graph_linear.add_node('E')
        self.graph_linear.add_node('F')
        self.graph_linear.add_node('G')
        self.graph_linear.visit('A', 1)
        self.graph_linear.add_edge('A', 'B', 0.5)
        self.graph_linear.add_edge('A', 'C', 0.3)
        self.graph_linear.add_edge('A', 'D', 0.2)
        curr_step += 1
        self.graph_linear.visit('B', curr_step)
        self.graph_linear.add_edge('B', 'E', 0.5)
        self.graph_linear.add_edge('B', 'C', 0.4)
        self.graph_linear.add_edge('B', 'A', 0.1)
        curr_step += 1
        self.graph_linear.visit('E', curr_step)
        self.graph_linear.add_edge('E', 'F', 0.5)
        self.graph_linear.add_edge('E', 'G', 0.1)
        self.graph_linear.add_edge('E', 'C', 0.3)
        self.graph_linear.add_edge('E', 'B', 0.1)
        curr_step += 1
        self.graph_linear.visit('F', curr_step)

        steps_upto = 4  # Consider the state of the graph at step 4
        lambda_value = 0.1  # Assuming a positive lambda for increase over steps
        decay_factor_b = (curr_step / steps_upto) * lambda_value

        # Call get_state on the graph with steps up to 10
        actual_state = self.graph_linear.get_state(steps_upto)

        # Assert the expected state (with amplified weights) matches the actual state
        self.assertAlmostEqual(actual_state.get('C'), 0.7999999, places=5)
        self.assertNotIn('B', actual_state)  # 'B' should be in the state since it's visited

    def test_get_state_linear(self):
        # Test getting the state with decayed weights for unvisited nodes
        self.graph_exp.add_node('A')
        self.graph_exp.add_node('B')
        self.graph_exp.add_node('C')
        self.graph_exp.add_node('D')
        self.graph_exp.add_node('E')
        self.graph_exp.add_node('F')
        self.graph_exp.add_node('G')
        self.graph_exp.visit('A', 1)
        self.graph_exp.add_edge('A', 'B', 0.5)
        self.graph_exp.add_edge('A', 'C', 0.3)
        self.graph_exp.add_edge('A', 'D', 0.2)
        self.graph_exp.visit('B', 2)
        self.graph_exp.add_edge('B', 'E', 0.5)
        self.graph_exp.add_edge('B', 'C', 0.4)
        self.graph_exp.add_edge('B', 'A', 0.1)
        self.graph_exp.visit('E', 3)
        self.graph_exp.add_edge('E', 'F', 0.5)
        self.graph_exp.add_edge('E', 'G', 0.1)
        self.graph_exp.add_edge('E', 'C', 0.3)
        self.graph_exp.add_edge('E', 'B', 0.1)
        self.graph_exp.visit('F', 4)

        steps_upto = 4  # Consider the state of the graph at step 4
        lambda_value = 0.1  # Assuming a positive lambda for increase over steps
        decay_factor_b = math.exp((steps_upto - 4) * lambda_value)
        expected_weight_b = 0.5 * decay_factor_b  # Amplified weight for 'B'
        total_prob = expected_weight_b

        # Call get_state on the graph with steps up to 10
        actual_state = self.graph_exp.get_state(steps_upto)

        # Assert the expected state (with amplified weights) matches the actual state
        self.assertAlmostEqual(actual_state.get('C'), 0.774826, places=5)
        self.assertNotIn('B', actual_state)  # 'B' should be in the state since it's visited


if __name__ == '__main__':
    unittest.main()
