import torch
import torch.nn as nn
import torch.nn.functional as F


class focal_loss_with_logits(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction='mean'):

        super().__init__()

        self.NAME = 'focal_loss_with_Logits'
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


class contrastive_loss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()

        self.NAME = 'contrastive_loss'
        self.margin = margin

    def forward(self, input):
        euclidean_distance = torch.cdist(input, input)
        loss = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.triu(loss, diagonal=1)
        return torch.sum(loss)


class binary_entropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.NAME = 'binary_entropy'
        self.reduction = reduction

    def forward(self, inputs, targets):
        return F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)


class cross_entropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.NAME = 'cross_entropy'
        self.reduction = reduction

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, reduction=self.reduction)


# class cross_entropy(F.cross_entropy):

#     def __init__(self):
#         super().__init__()
#         self.NAME = 'entropy'

# def init_criterion(contrastive_loss_margin=1.0):

#     CRITERION = {
#         'contrastive': ContrastiveLoss(margin=contrastive_loss_margin),
#         'inv_euc': InvEucLoss(),
#         'binary_entropy': F.binary_cross_entropy_with_logits,
#         'entropy': F.cross_entropy
#     }

#     return CRITERION

CRITERIONS = {
    'contrastive': contrastive_loss,
    'focal': focal_loss_with_logits,
    # 'binary_entropy': F.binary_cross_entropy_with_logits,
    'binary_entropy': binary_entropy,
    # 'entropy': F.cross_entropy
    'cross_entropy': cross_entropy
}
