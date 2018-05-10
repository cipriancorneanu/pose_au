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
from dataset import Fera2017Dataset, ToTensor
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from eval import evaluate_model
from topologies import GDVM
from logger import Logger
from utils import save_checkpoint
import os

parser = argparse.ArgumentParser(description='VAE.')
parser.add_argument("--path", default='/data/data1/datasets/fera2017/',
                    help='input path for data')
parser.add_argument("--batch_size", type=int, default=64,
                    help='input batch size(default=64)')
parser.add_argument("--epochs", type=int, default=5,
                    help='number of epochs to train(default=5)')
parser.add_argument("--patch", default='faces',
                    help='type of patch to fetch from dataset(defaut=\'faces\')')
parser.add_argument("--n_folds", type=int, default=3,
                    help='defines the n-fold training scenario')
parser.add_argument("--test_fold", type=int, default=1,
                    help='defines the test')
parser.add_argument("--lr", type=float, default=0.001,
                    help='optimization learning rate')
parser.add_argument("--size_latent", type=int, default=20,
                    help='size of latent representation')
parser.add_argument("--log_interval", type=int, default=100,
                    help='how many iterations to wait before logging info')
parser.add_argument("--beta", type=float, default=1,
                    help='how much the KL weights in the final loss')
parser.add_argument("--k_beta", type=float, default=0,
                    help='Adapt how much KL weights in the final loss every epoch : beta = beta + k_beta*beta')
parser.add_argument("--quick_test", type=bool, default=False,
                    help='For quick testing use just one subject and pose.')
parser.add_argument("--model_fname",
                    help='File name for model to be loaded.')
args = parser.parse_args()

poses = [1, 6, 7]
n_classes = 10
eval_thresholds = np.arange(0.5, 0.55, 0.05)
tsfm = ToTensor()

oname = 'gdvm_beta_alr_alldata_' + str(args.beta) + \
    '_kbeta_' + str(args.k_beta) + '_lr_' + str(args.lr)

''' If just quick test reduce the training/test subjcets and poses to minimum '''
(t_subs_tr, t_subs_te) = (['F001'], ['F007']
                          ) if args.quick_test else (None, None)
poses = ([6]) if args.quick_test else [1, 2, 3, 4, 5, 6, 7, 8, 9]

dl_test, n_iter_test = [], []
for pose in poses:
    dt_test = Fera2017Dataset('/data/data1/datasets/fera2017/',
                              partition='validation', tsubs=t_subs_te,  tposes=[pose], transform=tsfm)
    n_iter_test.append(len(dt_test)/args.batch_size)
    dl_test.append(DataLoader(dt_test, batch_size=args.batch_size,
                              shuffle=False, num_workers=4))

n_iter_test_total = np.sum(n_iter_test)
print('n_iter in test per pose: {}, total: {}'.format(
    n_iter_test, n_iter_test_total))

model = GDVM()
model.cuda()


def KLD(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def BCE(pred, target):
    return F.binary_cross_entropy(pred, target, size_average=False)


logger = Logger('./logs/'+oname+'/')


def test(model, n_runs):
    model.eval()
    f1s, loss = [], 0
    for i, dl_test_pose in enumerate(dl_test):
        targets, preds = [], []
        print(
            '-----------------------------------Evaluating POSE {} ------------------------- '.format(poses[i]))
        for iter, (data, target, _) in enumerate(dl_test_pose):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True).float(
            ), Variable(target, volatile=True).float()

            outputs = [model(data) for run in range(n_runs)]

            for (pred, mu, logvar) in outputs:
                loss += (BCE(pred, target) + args.beta *
                         KLD(mu, logvar)).data[0] / (len(data)*n_runs*n_iter_test_total)

            pred = np.mean(np.asarray([p.data.cpu().numpy()
                                       for (p, mu, lvar) in outputs]), axis=0)

            preds.append(pred)
            targets.append(target.data.cpu().numpy())

        preds = np.asarray(np.concatenate(preds))
        targets = np.clip(np.rint(np.concatenate(targets)),
                          0, 1).astype(np.uint8)

        print('{}: mean:{}, var:{}'.format(i, np.mean(preds), np.var(preds)))

        for p, t in zip(preds[:5], targets[:5]):
            print('{}/{}'.format(p, t))

        ''' Evaluate model per pose'''
        f1_pose = []
        for t in eval_thresholds:
            preds_f = np.copy(preds)
            preds_f[np.where(preds_f < t)] = 0
            preds_f[np.where(preds_f >= t)] = 1

            preds_f = np.reshape(preds_f, (-1, n_classes))

            print('--------EVAL PRED------ t = {}'.format(t))
            _, _, f1, _, _ = evaluate_model(targets, preds_f, verbose=True)
            f1_pose.append(f1)

        f1s.append(f1_pose)

    f1s = np.mean(f1s, axis=0)
    print('\n====> Test loss: {:.4f}\n'.format(loss))
    print('\n====> Mean F1: {}\n'.format(f1s))


if os.path.isfile(args.model_fname):
    print("=> loading checkpoint '{}'".format(args.model_fname))
    checkpoint = torch.load(args.model_fname)
    model.load_state_dict(checkpoint['state_dict'])
    '''optimizer.load_state_dict(checkpoint['optimizer'])'''

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.model_fname, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.model_fname))


test(model, n_runs=5)
