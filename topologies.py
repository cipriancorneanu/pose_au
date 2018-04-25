from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


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


class BiggerNet(nn.Module):
    def __init__(self):
        super(BiggerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=1)
        self.pool1 = nn.MaxPool2d(3)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1)
        self.bn2 = nn.MaxPool2d(3)
        self.pool2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1)
        self.pool5 = nn.MaxPool2d(4)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        x = self.pool5(x)

        x = x.view(-1, 4096)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))

        return x


class GDVM(nn.Module):
    def __init__(self):
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
        self.pool5 = nn.MaxPool2d(4)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, 64)
        self.fc22 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 512)
        self.fc4 = nn.Linear(512, 10)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        x = x.view(-1, 4096)

        x = F.relu(self.fc1(x))
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
        y = F.sigmoid(self.fc4(y))

        return y, mu, logvar


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
