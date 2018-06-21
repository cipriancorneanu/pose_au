'''
Discriminative model with generative regularization.

It implements the following paper:
Yeh, C. K., Tsai, Y. H. H., & Wang, Y. C. F. (2017).
Generative-Discriminative Variational Model for Visual Recognition. arXiv preprint arXiv:1706.02295.

'''

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import optim
from dataset import Fera2017Dataset, ToTensor, TripletSampler
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from eval import evaluate_model
from topologies import GDVM_ResNet
from logger import Logger
from utils import save_checkpoint
import os
from losses import bce_loss, kld

parser = argparse.ArgumentParser()
parser.add_argument("--path", default='/data/data1/datasets/fera2017/',
                    help='input path for data')
parser.add_argument("--batch_size", type=int, default=64,
                    help='input batch size(default=64)')
parser.add_argument("--epochs", type=int, default=10,
                    help='number of training epochs(default=10)')
parser.add_argument("--encoder", default='resnet18',
                    help='Topology used to represent the input')
parser.add_argument("--criterion", default='bce',
                    help='Chooses the discriminative criterion in the loss: bce = binary cross entropy, focal: focal loss')
parser.add_argument("--lr", type=float, default=0.001,
                    help='optimization learning rate')
parser.add_argument("--size_latent", type=int, default=256,
                    help='size of latent representation')
parser.add_argument("--dropout", type=float, default=0.2,
                    help='Dropout probability.')
parser.add_argument("--log_interval", type=int, default=100,
                    help='how many iterations to wait before logging info')
parser.add_argument("--beta", type=float, default=0,
                    help='how much the KL weights in the final loss')
parser.add_argument("--gamma", type=float, default=0,
                    help='parameter of the focal loss, if 0 the focal loss is equivalent to the binary cross entropy')
parser.add_argument("--aus",  default='[0,1,2,3,4,5,6,7,8,9]',
                    help='Select specific classes.')
parser.add_argument("--poses",  default='[1,2,3,4,5,6,7,8,9]',
                    help='Select specific poses.')
parser.add_argument("--alpha", type=float, default=1,
                    help='parameter of the focal loss')
parser.add_argument("--quick_test", type=bool, default=False,
                    help='For quick testing use just one subject and pose.')
parser.add_argument("--model_fname",  default='',
                    help='Name of model to be loaded')
args = parser.parse_args()

aus = map(int, args.aus.strip('[]').split(','))
print(aus)

poses = [6]
n_classes = len(aus)
print(n_classes)
eval_thresholds = np.arange(0.5, 0.55, 0.05)
tsfm = ToTensor()

''' If just quick test reduce the training/test subjcets and poses to minimum '''
(t_subs_tr, t_subs_te) = (['F001'], ['F007']
                          ) if args.quick_test else (None, None)
poses = ([6]) if args.quick_test else poses

dl_test, n_iter_test = [], []
for pose in poses:
    dt_test = Fera2017Dataset('/data/data1/datasets/fera2017/',
                              partition='validation', tsubs=t_subs_te,  tposes=[pose], transform=tsfm, verbose=True)
    n_iter_test.append(len(dt_test)/args.batch_size)
    dl_test.append(DataLoader(dt_test, batch_size=args.batch_size,
                              num_workers=4))

n_iter_test_total = np.sum(n_iter_test)
print('n_iter in test per pose: {}, total: {}'.format(
    n_iter_test, n_iter_test_total))

encoder = models.resnet18()
model = GDVM_ResNet(encoder, latent=256, out=n_classes, dropout=0.4)
model.cuda()


def test():
    model.eval()
    f1s, mus = [], []
    for i, dl_test_pose in enumerate(dl_test):
        targets, preds = [], []
        print(
            '-----------------------------------Evaluating POSE {} ------------------------- '.format(poses[i]))
        for iter, (data, target, _) in enumerate(dl_test_pose):
            target = torch.clamp(target[:, aus], 0, 1)
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(
            ), Variable(target).float()

            if iter % args.log_interval == 0:
                print('Iter {}'.format(iter))

            with torch.no_grad():
                pred, mu, logvar = model(data)
                pred = F.sigmoid(pred)

                preds.append(pred)
                mus.append(mus)
                targets.append(target.data.cpu().numpy())

        preds = np.asarray(np.concatenate(preds))
        '''
        print('preds min:{}, max:{}, mean:{}'.format(
            preds.min(), preds.max(), np.mean(preds)))
        '''
        targets = np.clip(np.rint(np.concatenate(targets)),
                          0, 1).astype(np.uint8)

        ''' Evaluate model per pose'''
        f1_pose = []
        for t in eval_thresholds:
            preds_f = np.copy(preds)
            preds_f[np.where(preds_f < t)] = 0
            preds_f[np.where(preds_f >= t)] = 1

            preds_f = np.reshape(preds_f, (-1, n_classes))

            if t == 0.5:
                print('--------EVAL PRED------ t = {}'.format(t))
                _, _, f1, _, _ = evaluate_model(
                    targets, preds_f, verbose=True)
            else:
                _, _, f1, _, _ = evaluate_model(
                    targets, preds_f, verbose=False)

            f1_pose.append(f1)

        f1s.append(f1_pose)


if os.path.isfile(args.model_fname):
    print("=> loading checkpoint '{}'".format(args.model_fname))
    checkpoint = torch.load(args.model_fname)
    model.load_state_dict(checkpoint['state_dict'])
    '''optimizer.load_state_dict(checkpoint['optimizer'])'''

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.model_fname, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.model_fname))

test()
