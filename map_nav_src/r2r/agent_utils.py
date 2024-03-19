import unittest

import numpy as np
import torch

from models.graph_utils import MAX_DIST
from tslearn.metrics import dtw_path_from_metric


def get_window_indices(i, rows=3, cols=12):
    # Calculate row and column position
    row, col = divmod(i, cols)

    # Generate row indices without wrapping (for cylindrical behavior)
    row_indices = [max(row - 1, 0), row, min(row + 1, rows - 1)]

    # Generate column indices with wrapping
    col_indices = [(col - 1) % cols, col, (col + 1) % cols]

    # Generate a grid of indices
    r_idx, c_idx = np.meshgrid(row_indices, col_indices, indexing='ij')
    surrounding_indices = r_idx * cols + c_idx

    return surrounding_indices.flatten()


panorama_windows = [get_window_indices(i) for i in range(36)]


def weighted_mean_around_index(panorama_features, i, centre_weight=0.8):
    indices = panorama_windows[i]

    # Initialize weights - 8 surrounding elements and the center element
    weights = np.ones(9, dtype=panorama_features.dtype) / 9
    weights[4] = centre_weight  # Assign larger weight to the center
    weights /= weights.sum()  # Normalize to make the sum of weights equal to 1

    # Select the elements and apply weighted mean
    selected_elements = panorama_features[indices, :]
    weighted_mean = np.sum(selected_elements * weights[:, None], axis=0)

    return weighted_mean


def compare_paths(A, B):
    result = []
    for i, a_element in enumerate(A):
        # If B is shorter than the current index, automatically assign 2
        if i >= len(B):
            result.append(2)
        # Check if the current element is at the same position in both lists
        elif a_element == B[i]:
            result.append(0)
        else:
            result.append(2)
    return result


def DTW(seq_a, seq_b, b_gt_length, band_width=None):
    """
    DTW is used to find the optimal alignment path;
    Returns GT like 001110000 for each seq_a
    """

    if band_width is None:
        path, dist = dtw_path_from_metric(seq_a.detach().cpu().numpy(),
                                          seq_b.detach().cpu().numpy())
    else:
        path, dist = dtw_path_from_metric(seq_a.detach().cpu().numpy(),
                                          seq_b.detach().cpu().numpy(),
                                          sakoe_chiba_radius=band_width)

    with torch.no_grad():
        att_gt = torch.zeros((seq_a.shape[0], b_gt_length)).cuda()

        for i in range(len(path)):
            att_gt[path[i][0], path[i][1]] = 1

        # v2 new: allow overlap
        for i in range(seq_a.shape[0]):
            pos = (att_gt[i] == 1).nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                pos = [i, i]
            if pos[0] - 1 >= 0:
                att_gt[i, pos[0] - 1] = 1
            if pos[-1] + 1 < seq_b.shape[0]:
                att_gt[i, pos[-1] + 1] = 1

    return att_gt


def calculate_confidence_scores(reference_nodes, target_nodes, path, distance_matrix):
    epsilon = 1e-6  # To prevent division by zero
    confidence_scores = []
    stop = False
    for target in target_nodes:
        if target is None:
            confidence_scores.append(0.0)
            continue
        closest_distance = MAX_DIST
        if target in path:
            closest_distance = 0
        else:
            for reference in reference_nodes:
                distance = distance_matrix[target][reference]
                if distance < closest_distance:
                    closest_distance = distance
        # Calculate the confidence score as the inverse of the closest distance

        confidence_scores.append(1 - (closest_distance / MAX_DIST))

        if path[-1] == target_nodes[-1]:
            confidence_scores[0] = 1.0
    return torch.softmax(torch.tensor(confidence_scores) * MAX_DIST, 0)


class TestComparePaths(unittest.TestCase):
    def test_identical_paths(self):
        self.assertEqual(compare_paths(['a', 'b', 'c'], ['a', 'b', 'c']), [0, 0, 0])

    def test_different_paths_same_length(self):
        self.assertEqual(compare_paths(['a', 'b', 'c'], ['a', 'x', 'c']), [0, 2, 0])

    def test_second_path_shorter(self):
        self.assertEqual(compare_paths(['a', 'b', 'c'], ['a']), [0, 2, 2])

    def test_first_path_shorter(self):
        self.assertEqual(compare_paths(['a'], ['a', 'b', 'c']), [0])

    def test_empty_first_path(self):
        self.assertEqual(compare_paths([], ['a', 'b', 'c']), [])

    def test_empty_second_path(self):
        self.assertEqual(compare_paths(['a', 'b', 'c'], []), [2, 2, 2])

    def test_both_paths_empty(self):
        self.assertEqual(compare_paths([], []), [])

    def test_different_paths_different_lengths(self):
        self.assertEqual(compare_paths(['a', 'b', 'c', 'd'], ['a', 'x', 'y']), [0, 2, 2, 2])
        self.assertEqual(compare_paths(['a', 'b', 'c', 'd'], ['z', 'x', 'y']), [2, 2, 2, 2])
        self.assertEqual(compare_paths(['a', 'b', 'c', 'd'], ['z', 'x', 'y', 'q']), [2, 2, 2, 2])

    def test_different_paths_shifted(self):
        self.assertEqual(compare_paths(['z', 'a', 'b', 'c'], ['a', 'b', 'c', 'd']), [2, 2, 2, 2])
        self.assertEqual(compare_paths(['a', 'b', 'c', 'd'], ['z', 'a', 'b', 'c', 'd']), [2, 2, 2, 2])


if __name__ == '__main__':
    unittest.main()
