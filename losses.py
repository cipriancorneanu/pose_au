import torch.nn.functional as F
import torch.nn as nn
import torch


def KLD(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def BCE(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target, size_average=False)


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20, alpha=0.5, gamma=1):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma

    def focal_loss(self, x, y):
        alpha = 0.75

        p = x.sigmoid()
        pt = p*y + (1-p)*(1-y)
        alpha = alpha*y + (1-alpha)*(1-y)
        w = alpha*(1-pt).pow(self.gamma)
        '''w = (1-pt).pow(self.gamma)'''

        return F.binary_cross_entropy_with_logits(x, y, w, size_average=False)

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets)
