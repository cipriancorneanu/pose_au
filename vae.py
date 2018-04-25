from __future__ import print_function
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dataset import Fera2017Dataset, ToTensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=2)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=4, stride=2)

        self.fc2 = nn.Linear(512, 200)
        self.fc21 = nn.Linear(200, args.size_latent)
        self.fc22 = nn.Linear(200, args.size_latent)
        self.fc3 = nn.Linear(args.size_latent, 200)
        self.fc4 = nn.Linear(200, 512)

        self.deconv6 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 512)

        x = self.relu(self.fc2(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = x.view(-1, 512, 1, 1)

        x = F.relu(self.deconv6(x))
        x = F.relu(self.deconv5(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv1(x))

        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
print(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

'''
subs_train = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
              'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020',
              'F021', 'F022', 'F023', 'M001', 'M002']
subs_test = ['F007', 'F008']
'''
poses = [1, 6, 7]
tsfm = ToTensor()

dt_train = Fera2017Dataset('/data/data1/datasets/fera2017/',
                           partition='train', tsubs=None, tposes=[1, 6, 7], transform=tsfm)
dl_train = DataLoader(dt_train, batch_size=64, shuffle=True, num_workers=4)

dl_test, n_iter_test = [], []
for pose in poses:
    dt_test = Fera2017Dataset('/data/data1/datasets/fera2017/',
                              partition='train', tsubs=None,  tposes=[pose], transform=tsfm)
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
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def train(epoch):
    model.train()
    train_loss = 0
    for iter, (data, target, _) in enumerate(dl_train):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).float()

        optimizer.zero_grad()
        recon_data, mu, logvar = model(data)
        mse = F.mse_loss(recon_data, data)
        kld = KLD(mu, logvar)
        '''loss = loss_function(recon_data, data, mu, logvar)'''
        loss = mse + kld

        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if iter % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.5f}+{:.5f}={:.5f}'.format(
                epoch, iter, n_iter_train, mse.data[0], kld.data[0], loss.data[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / n_iter_train))


def test():
    model.eval()
    test_loss = 0
    for i, dl_test_pose in enumerate(dl_test):
        print(
            '-----------------------------------Evaluating POSE {} ------------------------- '.format(poses[i]))
        for iter, (data, target, _) in enumerate(dl_test_pose):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(), Variable(target).float()

            recon_data, mu, logvar = model(data)
            test_loss += loss_function(recon_data, data, mu, logvar).data[0]

            if iter == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_data[:n]])
                save_image(comparison.data.cpu()/255.,
                           'results/vae/reconstruction_pose_' + str(i) +
                           '_epoch_' + str(epoch) + '_lt_'+str(args.size_latent)+'.png', nrow=n)

        test_loss /= n_iter_test[i]
        print('====> Test loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs+1):
    train(epoch)
    test()
