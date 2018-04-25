from __future__ import print_function
import os
import argparse
import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from dataset import Fera2017Dataset, ToTensor
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from eval import evaluate_model
from topologies import GDVM

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
parser.add_argument("--divergence", type=float, default=10,
                    help='how high the KL must be to propagate KL regularization')

args = parser.parse_args()

prefix, oname = 'no_weighting_', os.path.basename(__file__).split('.')[0] + '_' + \
                str(args.n_folds) + 'folds_tf_' + \
                str(args.test_fold) + '_' + args.patch


model = GDVM()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
poses = [1, 6, 7]
tsfm = ToTensor()

dt_train = Fera2017Dataset('/data/data1/datasets/fera2017/',
                           partition='train', tsubs=None, tposes=[1, 6, 7], transform=tsfm)
dl_train = DataLoader(dt_train, batch_size=64, shuffle=True, num_workers=4)

dl_test, n_iter_test = [], []
for pose in poses:
    dt_test = Fera2017Dataset('/data/data1/datasets/fera2017/',
                              partition='validation', tsubs=None,  tposes=[pose], transform=tsfm)
    n_iter_test.append(len(dt_test)/args.batch_size)
    dl_test.append(DataLoader(dt_test, batch_size=64,
                              shuffle=True, num_workers=4))

n_iter_train = len(dt_train)/args.batch_size

print('n_iter in train : {}'.format(n_iter_train))
print('n_iter in test: {}'.format(n_iter_test))


def KLD(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def loss(pred, target, mu, logvar, beta):
    return F.binary_cross_entropy(pred, target) + args.beta*KLD(mu, logvar)


def train(epoch):
    model.train()
    acc_loss = 0
    for iter, (data, target, _) in enumerate(dl_train):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).float()

        optimizer.zero_grad()
        pred, mu, logvar = model(data)

        kld = KLD(mu, logvar)
        bce = F.binary_cross_entropy(pred, target)
        loss = bce + args.beta*kld

        ''' Backpropagate KL regularization just when it's too high'''
        if kld.data[0] > args.divergence:
            loss.backward()
        else:
            bce.backward()

        acc_loss += loss.data[0]
        optimizer.step()

        if iter % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.4f} + {}*{:.4f} = {:.4f}'.format(
                epoch, iter, n_iter_train, bce.data[0], args.beta, kld.data[0], loss.data[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, acc_loss / n_iter_train))


def test():
    model.eval()
    for i, dl_test_pose in enumerate(dl_test):
        acc_loss, targets, preds = 0, [], []
        print(
            '-----------------------------------Evaluating POSE {} ------------------------- '.format(poses[i]))
        for iter, (data, target, _) in enumerate(dl_test_pose):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(), Variable(target).float()

            pred, mu, logvar = model(data)

            acc_loss += loss(pred, target, mu, logvar, args.beta).data[0]
            preds.append(pred.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

        pred = np.asarray(
            np.clip(np.rint(np.concatenate(preds)), 0, 1), dtype=np.uint8)
        target = np.clip(np.rint(np.concatenate(targets)),
                         0, 1).astype(np.uint8)

        evaluate_model(target, pred)

        print('====> Test loss: {:.4f}'.format(acc_loss / n_iter_test[i]))


for epoch in range(1, args.epochs+1):
    train(epoch)
    test()
