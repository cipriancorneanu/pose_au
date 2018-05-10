from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from eval import evaluate_model
import numpy as np
import os
from dataset import Fera2017Dataset, ToTensor
from topologies import Net

parser = argparse.ArgumentParser(description='Patch classifier for DSIN.')
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
parser.add_argument("--log_interval", type=int, default=100,
                    help='how many iterations to wait before logging info')
args = parser.parse_args()

prefix, oname = 'no_weighting_', os.path.basename(__file__).split('.')[0] + '_' + \
                str(args.n_folds) + 'folds_tf_' + \
    str(args.test_fold) + '_' + args.patch

model = Net()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(n_params)
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
    dl_test.append(DataLoader(dt_test, batch_size=64,
                              shuffle=True, num_workers=4))
    n_iter_test.append(len(dt_test)/args.batch_size)

n_iter = len(dt_train)/args.batch_size

print('n_iter in train is {}'.format(n_iter))
print('n_iter in test is {}'.format(n_iter_test))


def train(epoch):
    model.train()
    mean_loss = []
    for iter, (data, target, _) in enumerate(dl_train):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(
        ), Variable(target).float()

        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if iter % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t\tLoss: {:.6f}'.format(
                epoch, iter, n_iter, loss.data[0]))

        mean_loss.append(loss.data.cpu().numpy())

    print('-----Mean loss in train : {}-----'.format(np.mean(mean_loss)))


def test():
    model.eval()
    mean_loss = []
    for i, dl_test_pose in enumerate(dl_test):
        targets, preds = [], []

        print(
            '-----------------------------------Evaluating POSE {} ------------------------- '.format(poses[i]))
        for iter, (data, target, _) in enumerate(dl_test_pose):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True).float(
            ), Variable(target, volatile=True).float()

            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            preds.append(output.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

            mean_loss.append(loss.data.cpu().numpy())

        pred = np.asarray(
            np.clip(np.rint(np.concatenate(preds)), 0, 1), dtype=np.uint8)
        target = np.clip(np.rint(np.concatenate(targets)),
                         0, 1).astype(np.uint8)

        evaluate_model(target, pred)

    print('-----Mean loss in validation : {}-----'.format(np.mean(mean_loss)))


for epoch in range(1, args.epochs+1):
    train(epoch)
    test()
