import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''

    def __init__(self, gamma=2, weight=None, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, targets, self.weight)
        return loss


class MultiLabelLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, inputs, targets):
        # inputs: Tensor of shape [B, N, C]
        # targets: Tensor of shape [B, N]

        # Flatten the inputs and targets to shape [B*N, C] and [B*N] respectively
        B, N, C = inputs.shape
        inputs_flat = inputs.reshape(-1, C)
        targets_flat = targets.reshape(-1)

        # Create a mask for non-ignored targets
        mask = targets_flat != self.ignore_index

        # Apply the mask to filter out ignored targets
        inputs_valid = inputs_flat[mask]
        targets_valid = targets_flat[mask]

        # Calculate the loss only on the valid data
        loss = self.criterion(inputs_valid, targets_valid)

        return loss


class MultiLabelCrossEntropyLoss(MultiLabelLoss):
    def __init__(self, weight=None, ignore_index=-1, reduction='none'):
        super(MultiLabelCrossEntropyLoss, self).__init__(weight=weight, ignore_index=ignore_index,
                                                         reduction=reduction)
        self.criterion = nn.CrossEntropyLoss(reduction=self.reduction, weight=self.weight)


class MultiLabelFocalLoss(MultiLabelLoss):
    def __init__(self, gamma=2, weight=None, ignore_index=-1, reduction='none'):
        super(MultiLabelFocalLoss, self).__init__(weight=weight, reduction=reduction, ignore_index=ignore_index)

        self.criterion = FocalLoss(gamma=gamma, weight=None, reduction='none')


def multilabel_cross_entropy(inputs, targets, weight=None, ignore_index=-1, reduction='none'):
    return MultiLabelCrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)(inputs, targets)


def multilabel_focal_loss(inputs, targets, gamma=2, weight=None, ignore_index=-1, reduction='none'):
    return MultiLabelFocalLoss(gamma=gamma, weight=weight, ignore_index=ignore_index, reduction=reduction)(inputs, targets)
