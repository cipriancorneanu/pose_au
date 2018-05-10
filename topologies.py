from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


''' 
Implements: 
Yeh, C. K., Tsai, Y. H. H., & Wang, Y. C. F. (2017). Generative-Discriminative Variational Model for Visual Recognition. arXiv preprint arXiv:1706.02295.
'''


class GDVM(nn.Module):
    def __init__(self, dropout=0.2):
        super(GDVM, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=1)
        self.pool1 = nn.MaxPool2d(3)
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1)
        self.bn2 = nn.MaxPool2d(3)
        self.pool2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1)
        self.pool5 = nn.MaxPool2d(7)

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc21 = nn.Linear(256, 64)
        self.fc22 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 10)
        self.dropout = nn.Dropout2d(dropout)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        x = F.relu(self.conv4(x))
        x = self.dropout(x)

        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        print(x.data.size())

        x = x.view(-1, 4096)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        mu = F.relu(self.fc21(x))
        logvar = F.relu(self.fc22(x))

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        y = F.relu(self.fc3(z))
        y = self.dropout(y)
        y = self.fc4(y)

        return y, mu, logvar


'''
Same as GDVM but without reparameterization. Purely discriminative. 
'''


class GDVM_D(GDVM):
    def __init__(self, dropout):
        super(GDVM_D, self).__init__(dropout=dropout)

    def forward(self, x):
        z, _ = self.encode(x)
        y = F.relu(self.fc3(z))
        y = self.dropout(y)
        y = self.fc4(y)

        return y


class GDVM_ResNet(nn.Module):
    def __init__(self, encoder, latent=128, out=10, dropout=0.2):
        super(GDVM_ResNet, self).__init__()

        self.encoder = encoder
        self.fc1 = nn.Linear(1000, latent)
        self.fc2 = nn.Linear(latent,  1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, out)
        self.dropout = nn.Dropout2d(dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        x = self.encoder(x)

        mu = self.dropout(self.fc1(x))
        logvar = self.dropout(self.fc1(x))

        x = self.reparameterize(mu, logvar)

        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        y = self.fc4(x)

        return y, mu, logvar


'''
Variational Auto-Encoder
Implements the following paper:
Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
'''


class VAE(nn.Module):
    def __init__(self, sz):
        super(VAE, self).__init__()
        self.size_latent = sz

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=2)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(512, 200)
        self.fc21 = nn.Linear(200, args.size_latent)
        self.fc22 = nn.Linear(200, args.size_latent)
        self.fc3 = nn.Linear(args.size_latent, 200)
        self.fc4 = nn.Linear(200, 512)

        self.fcd1 = nn.Linear(args.size_latent, 512)
        self.fcd2 = nn.Linear(512, 10)

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

        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

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

        y = self.relu(self.fcd1(z))
        y = self.sigmoid(self.fcd2(y))

        return self.decode(z), mu, logvar, y
