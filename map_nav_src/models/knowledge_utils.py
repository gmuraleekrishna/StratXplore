import os
import pickle

import numpy as np
import torch
from torch import nn
import cv2 as cv

from map_nav_src.r2r.parser import parsed_args


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, Q, V, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate Q through the network.

        Q: batch x query_dim
        V: batch x seq_len x query_dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(Q).unsqueeze(2)  # batch x query_dim x 1

        # Get attention
        attn = torch.bmm(V, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)  # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, V).squeeze(1)  # batch x query_dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, Q), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


def get_pano_affinity():
    # get the affinity diff_matrix of panorama, where edges between adjacent views are 1

    # low elevation view 0-11
    # middle elevation view 12-23
    # high elevation view 24-35

    # pano_a = np.zeros((36, 36))  # no self-connect
    pano_a = np.eye(36, dtype=float)  # self-connect

    # low elevation
    for view_id in range(0, 12):
        # up
        pano_a[view_id, view_id + 12] = 1

        # left, left-up
        if view_id == 0:
            pano_a[view_id, 11] = 1
            pano_a[view_id, 11 + 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 + 12] = 1

        # right, right-up
        if view_id == 11:
            pano_a[view_id, 0] = 1
            pano_a[view_id, 0 + 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 + 12] = 1

    # middle elevation
    for view_id in range(12, 24):
        # up
        pano_a[view_id, view_id + 12] = 1

        # down
        pano_a[view_id, view_id - 12] = 1

        # left, left-up, left-down
        if view_id == 12:
            pano_a[view_id, 23] = 1
            pano_a[view_id, 23 + 12] = 1
            pano_a[view_id, 23 - 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 + 12] = 1
            pano_a[view_id, view_id - 1 - 12] = 1

        # right, right-up, right-down
        if view_id == 23:
            pano_a[view_id, 12] = 1
            pano_a[view_id, 12 + 12] = 1
            pano_a[view_id, 12 - 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 + 12] = 1
            pano_a[view_id, view_id + 1 - 12] = 1

    # high elevation
    for view_id in range(24, 36):
        # down
        pano_a[view_id, view_id - 12] = 1

        # left, left-down
        if view_id == 24:
            pano_a[view_id, 35] = 1
            pano_a[view_id, 35 - 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 - 12] = 1

        # right, right-down
        if view_id == 35:
            pano_a[view_id, 24] = 1
            pano_a[view_id, 24 - 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 - 12] = 1

    # checking symmetry
    assert np.sum(pano_a - pano_a.T) == 0
    pano_a = cv.GaussianBlur(pano_a, (3, 3), 1, cv.BORDER_TRANSPARENT)
    pano_a[np.eye(36, dtype=int) == 1] = 1
    return pano_a


pano_affinity = get_pano_affinity()


class pano_att_gcn_v5(nn.Module):
    def __init__(self, query_dim, ctx_dim, knowledge_dim=300, device=None):
        '''Initialize layer.'''
        super(pano_att_gcn_v5, self).__init__()
        self.knowledge_dim = knowledge_dim
        self.query_dim = query_dim
        self.linear_key = nn.Linear(ctx_dim + self.knowledge_dim, query_dim, bias=False)

        self.linear_query = nn.Linear(query_dim, query_dim, bias=False)
        # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.scale = query_dim ** -0.5
        self.linear_out = nn.Linear(query_dim + ctx_dim + 100, query_dim, bias=False)
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(pano_affinity.copy()).float().to(device)
            self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.05

    def forward(self, h, context, detect_feats, knowledge_vector, teacher_action_view_ids=None, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_query(h)  # batch x dim x 1

        context_know = torch.cat((context, detect_feats.unsqueeze(1).expand(-1, 36, -1)), dim=2)
        ba, _, ck_dim = context_know.shape
        context_know = self.linear_key(context_know.reshape(-1, ck_dim))
        context_know = context_know.reshape(ba, 36, self.query_dim)

        # Get attention
        attn = torch.bmm(context_know, target).squeeze(2)  # batch x seq_len
        logit = attn

        # new: add scale
        attn *= self.scale

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))

        attn = self.sm(attn)  # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        # gcn new
        batch, seq_len, ctx_dim = context.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch),
                               teacher_action_view_ids, :]
            pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)

        attn3_gcn = attn3 * pano_a_tea_batch

        weighted_context = torch.bmm(attn3_gcn, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h, knowledge_vector), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class Entity_Knowledge(nn.Module):
    def __init__(self, fact_dropout, top_k_facts):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=300, out_channels=100, kernel_size=(1, 1), padding=0, bias=True)

        self.top_k_facts = top_k_facts

        self.fact_embed_drop = nn.Dropout2d(fact_dropout)

        with torch.no_grad():
            with open(os.path.join(parsed_args.know_root_path,'pkls/knowledge_rel_embed_glove.pkl'), 'rb') as f:
                self.knowledge_fact_dict = pickle.load(f)

            with open(os.path.join(parsed_args.know_root_path,'pkls/vg_class_vico_embed.pkl'), 'rb') as f:
                self.vg_class_glove_embed = pickle.load(f)

            self.vg_class_glove_embed[-1] = torch.zeros(self.vg_class_glove_embed[0].shape[-1])
            self.knowledge_fact_dict[-1] = torch.zeros(6, 600)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, detect_labels):
        # detect_labels: batch_size, max_len, n_labels

        batch_size, max_len, n_labels = detect_labels.shape
        detect_labels = detect_labels.reshape(-1)

        with torch.no_grad():
            facts = [self.knowledge_fact_dict[int(label.item())] for _, label in enumerate(detect_labels)]
            facts = torch.stack(facts, dim=0).to(self.device)  # n_entities, top_k_facts, 600 ([rel_entity_embed,
            # rel_embed])
            if self.top_k_facts < facts.shape[1]:
                facts = facts[:, :self.top_k_facts, :]
            n_entities = facts.shape[0]  # n_entities = batch_size * max_len * n_labels
            facts = facts.reshape(n_entities, self.top_k_facts * 2, 300, 1)
            facts = facts.permute(0, 2, 1, 3)

        x = self.conv(facts)  # (n_entities, 100, self.top_k_facts * 2, 1)
        x = x.permute(0, 2, 1, 3)  # (n_entities, self.top_k_facts * 2, 100, 1)
        x = x.reshape(batch_size * max_len, n_labels * self.top_k_facts * 2, 100)
        x = x.mean(1).reshape(batch_size, max_len, -1)

        final_embed = x  # (batch_size, max_len, 300)
        final_embed = self.fact_embed_drop(final_embed)

        return final_embed
