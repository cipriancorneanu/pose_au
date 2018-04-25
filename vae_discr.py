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
from eval import evaluate_model
from topologies import VAE

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
parser.add_argument("--log_interval", type=int, default=10,
                    help='how many iterations to wait before logging info')
args = parser.parse_args()

prefix, oname = 'no_weighting_', os.path.basename(__file__).split('.')[0] + '_' + \
                str(args.n_folds) + 'folds_tf_' + \
                str(args.test_fold) + '_' + args.patch


model = VAE(args.size_latent)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

subs_train = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
              'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020']
subs_test = ['F021', 'F022', 'F023', 'M001']
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

# Reconstruction + KL divergence losses summed over all elements and batch


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def KLD(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train(epoch):
    model.train()
    train_loss = 0
    for iter, (data, target, _) in enumerate(dl_train):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).float()

        optimizer.zero_grad()
        recon_data, mu, logvar, pred = model(data)

        mse = torch.div(torch.sqrt(F.mse_loss(recon_data, data)), 255)
        kld = KLD(mu, logvar)
        bce = F.binary_cross_entropy(pred, target)

        loss = mse + 0.001*kld + bce

        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if iter % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.4f} + {:.4f} + {:.4f} = {:.4f}'.format(
                epoch, iter, n_iter_train, mse.data[0], 0.001*kld.data[0], bce.data[0], loss.data[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / n_iter_train))


def test():
    model.eval()
    test_loss, targets, preds = 0, [], []
    for i, dl_test_pose in enumerate(dl_test):
        print(
            '-----------------------------------Evaluating POSE {} ------------------------- '.format(poses[i]))
        for iter, (data, target, _) in enumerate(dl_test_pose):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(), Variable(target).float()

            recon_data, mu, logvar, pred = model(data)
            test_loss += loss_function(recon_data, data, mu, logvar).data[0]
            preds.append(pred.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

            if iter == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_data[:n]])
                save_image(comparison.data.cpu()/255.,
                           'results/vae_discr/reconstruction_pose_' + str(i) +
                           '_epoch_' + str(epoch) + '_lt_'+str(args.size_latent)+'.png', nrow=n)

            preds.append(pred.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())

        pred = np.asarray(
            np.clip(np.rint(np.concatenate(preds)), 0, 1), dtype=np.uint8)
        target = np.clip(np.rint(np.concatenate(targets)),
                         0, 1).astype(np.uint8)

        evaluate_model(target, pred)

        test_loss /= n_iter_test[i]
        print('====> Test loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs+1):
    train(epoch)
    test()
