import random
import numpy as np
import torch
from tslearn.metrics import dtw_path_from_metric


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
            > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask


def cosine_dist(a, b):
    '''
    i.e. inverse cosine similarity
    '''
    return 1 - np.dot(a, b)


def DTW(seq_a, seq_b, b_gt_length, band_width=None):
    """
    DTW is used to find the optimal alignment path;
    Returns GT like 001110000 for each seq_a
    """
    dist_func = cosine_dist

    if band_width is None:
        path, dist = dtw_path_from_metric(seq_a.detach().cpu().numpy(),
                                          seq_b.detach().cpu().numpy(),
                                          metric=dist_func)
    else:
        path, dist = dtw_path_from_metric(seq_a.detach().cpu().numpy(),
                                          seq_b.detach().cpu().numpy(),
                                          sakoe_chiba_radius=band_width,
                                          metric=dist_func)

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
