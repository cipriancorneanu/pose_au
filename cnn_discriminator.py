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
from dataset import Fera2017Dataset

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
parser.add_argument("--log_interval", type=int, default=10,
                    help='how many iterations to wait before logging info')
args = parser.parse_args()

prefix, oname = 'no_weighting_', os.path.basename(__file__).split('.')[0] + '_' + \
                str(args.n_folds) + 'folds_tf_' + \
    str(args.test_fold) + '_' + args.patch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=2)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = x.view(-1, 512)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x


model = Net()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

subs_train = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
              'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020']
subs_test = ['F021', 'F022', 'F023', 'M001']
poses = [1, 6, 7]

dt_train = Fera2017Dataset('/data/data1/datasets/fera2017/',
                           partition='train', tsubs=subs_train, tposes=[1, 6, 7])
dl_train = DataLoader(dt_train, batch_size=64, shuffle=True, num_workers=2)

dl_test = []
for pose in poses:
    dt_test = Fera2017Dataset('/data/data1/datasets/fera2017/',
                              partition='train', tsubs=subs_test,  tposes=[pose])
    dl_test.append(DataLoader(dt_test, batch_size=64,
                              shuffle=True, num_workers=2))

n_iter = len(dt_train)/args.batch_size


def train(epoch):
    model.train()

    for iter, (data, target, _) in enumerate(dl_train):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).float()

        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if iter % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t\tLoss: {:.6f}'.format(
                epoch, iter, n_iter, loss.data[0]))


def test():
    model.eval()
    targets, preds = [], []
    for i, dl_test_pose in enumerate(dl_test):
        print(
            '-----------------------------------Evaluating POSE {} ------------------------- '.format(poses[i]))
        for iter, (data, target, _) in enumerate(dl_test_pose):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(), Variable(target).float()

            preds.append(model(data).data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

        pred = np.asarray(
            np.clip(np.rint(np.concatenate(preds)), 0, 1), dtype=np.uint8)
        target = np.concatenate(targets)

        evaluate_model(target, pred)


for epoch in range(1, args.epochs+1):
    train(epoch)
    test()
