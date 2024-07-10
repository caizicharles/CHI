import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, input):
        euclidean_distance = torch.cdist(input, input)
        loss = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.triu(loss, diagonal=1)
        return torch.sum(loss)


def init_criterion(args):

    CRITERION = {
        'contrastive': ContrastiveLoss(margin=args.contrastive_loss_margin),
        'binary_entropy': F.binary_cross_entropy_with_logits,
        'entropy': F.cross_entropy
    }

    return CRITERION
