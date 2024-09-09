import torch
import torch.nn as nn
import torch.nn.functional as F


class focal_loss_with_logits(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction='mean'):

        super().__init__()

        self.NAME = 'focal_loss_with_Logits'
        self.TYPE = 'prediction'
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class binary_entropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.NAME = 'binary_entropy'
        self.TYPE = 'prediction'
        self.reduction = reduction

    def forward(self, inputs, targets):
        return F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)


class cross_entropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.NAME = 'cross_entropy'
        self.TYPE = 'prediction'
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.squeeze(-1)
        return F.cross_entropy(inputs, targets, reduction=self.reduction)


class euc_dist_loss(nn.Module):

    def __init__(self, factor: int = 1, normalize: bool = True):
        super().__init__()

        self.NAME = 'euc_dist_loss'
        self.TYPE = 'prototype'
        self.normalize = normalize
        self.factor = factor

    def forward(self, input):
        if not isinstance(input, list):
            input = [input]

        loss = 0.
        for i in input:
            if self.normalize:
                i = F.normalize(i, p=2, dim=-1)
            euclidean_distance = torch.cdist(i, i)
            l = -euclidean_distance
            loss += torch.sum(l)

        return self.factor * loss


class orthogonality_loss(nn.Module):

    def __init__(self, factor: int = 1):
        super().__init__()

        self.NAME = 'orthogonality_loss'
        self.TYPE = 'prototype'
        self.factor = factor

    def forward(self, input):
        if not isinstance(input, list):
            input = [input]

        loss = 0.
        for i in input:
            i = F.normalize(i, p=2, dim=-1)
            dot_product = torch.matmul(i, i.transpose(-1, -2))
            I = torch.eye(dot_product.size(0), dot_product.size(0), dtype=float, device=i.device)
            loss += torch.norm(dot_product - I, p='fro')**2

        return self.factor * loss


class contrastive_loss(nn.Module):

    def __init__(self, temperature: float = 0.5, factor: int = 1):
        super().__init__()

        self.NAME = 'contrastive_loss'
        self.TYPE = 'pretrain'
        self.temperature = temperature
        self.factor = factor

    def forward(self, input, mask):

        B = mask.size(0)
        D = input.size(-1)

        input = F.normalize(input, p=2, dim=-1)
        similarity_matrix = torch.matmul(input, input.T) / self.temperature
        similarity_matrix.fill_diagonal_(float('-inf'))
        logits = similarity_matrix.log_softmax(dim=-1)

        visit_mask = mask.all(dim=-1)
        visit_mask = ~visit_mask
        label_mask = visit_mask.to(int)
        label_mask = torch.sum(label_mask, dim=-1)
        label = torch.arange(B).to(label_mask.device)
        label = torch.repeat_interleave(label, label_mask)
        label = label.unsqueeze(-1) == label.unsqueeze(0)
        label.fill_diagonal_(False)

        positive_logits = logits[label]
        loss = -positive_logits.mean()

        return self.factor * loss


CRITERIONS = {
    'contrastive': contrastive_loss,
    'euc_dist': euc_dist_loss,
    'orthogonality': orthogonality_loss,
    'focal': focal_loss_with_logits,
    'binary_entropy': binary_entropy,
    'cross_entropy': cross_entropy
}
