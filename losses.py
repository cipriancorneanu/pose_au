import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


def kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


def npair_loss(x_a, x_p, x_n):
    '''
    x_a : anohor
    x_p : positive (same class as anchor)
    x_n : multiple negatives
    '''

    ''' Normalize '''
    x_a = x_a / x_a.norm()
    x_p = x_p / x_p.norm()
    x_n = x_n / x_n.norm()

    ''' Broadcast to negative dims'''
    x_a = x_a.repeat((x_n.size(0), 1))
    x_p = x_p.repeat((x_n.size(0), 1))

    f_apn = torch.matmul(x_a.transpose(0, 1), x_n) - \
        torch.matmul(x_a.transpose(0, 1), x_p)
    '''
    print('mean f_apn: {}, min f_apn:{}, max f_apn:{}'.format(
        torch.mean(f_apn), torch.min(f_apn), torch.max(f_apn)))
    print('mean exp: {}, min exp:{}, max exp:{}'.format(torch.mean(
        torch.exp(f_apn)), torch.min(torch.exp(f_apn)), torch.max(torch.exp(f_apn))))
    '''
    result = torch.mean(torch.log(torch.sum(torch.exp(f_apn))))

    return result


def angular_loss(x_a, x_p, x_n):
    '''
    x_a : anohor
    x_p : positive (same class as anchor)
    x_n : multiple negatives
    alpha : angle
    '''
    alpha = torch.tensor(np.pi/4).cuda()

    ''' Normalize '''
    x_a = x_a / x_a.norm()
    x_p = x_p / x_p.norm()
    x_n = x_n / x_n.norm()

    '''Expand x+a and x_p to match negatives size '''
    x_a = x_a.repeat((x_n.size(0), 1))
    x_p = x_p.repeat((x_n.size(0), 1))

    tan_sqr = torch.tan(alpha)**2

    f_apn = 4*tan_sqr*torch.matmul((x_a+x_p).transpose(0, 1), x_n) - 2 * \
        (1+tan_sqr)*torch.matmul(x_a.transpose(0, 1), x_p)
    '''
    print('mean f_apn: {}, min f_apn:{}, max f_apn:{}'.format(
        torch.mean(f_apn), torch.min(f_apn), torch.max(f_apn)))
    print('mean exp: {}, min exp:{}, max exp:{}'.format(torch.mean(
        torch.exp(f_apn)), torch.min(torch.exp(f_apn)), torch.max(torch.exp(f_apn))))
    '''
    return torch.mean(torch.log(torch.sum(torch.exp(f_apn))))


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
        loss = F.binary_cross_entropy_with_logits(x, y, w, size_average=False)

        return loss

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets)
