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
from dataset import Fera2017Dataset, TripletSampler, ToTensor
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from eval import evaluate_model
from topologies import GDVM, GDVM_ResNet
import torchvision.models as models
from logger import Logger
from utils import save_checkpoint
from losses import bce_loss, angular_loss, npair_loss, kld

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
parser.add_argument("--alpha", type=float, default=0,
                    help='how much the metric weights in the final loss')
parser.add_argument("--beta", type=float, default=0,
                    help='how much the regularization weights in the final loss')
parser.add_argument("--aus",  default='[0,1,2,3,4,5,6,7,8,9]',
                    help='Select specific classes to train on.')
parser.add_argument("--quick_test", type=bool, default=False,
                    help='For quick testing use just one subject and pose.')
args = parser.parse_args()

aus = map(int, args.aus.strip('[]').split(','))
print(aus)

n_classes = len(aus)

if args.encoder == 'resnet18':
    encoder = models.resnet18()
    model = GDVM_ResNet(encoder, latent=args.size_latent,
                        out=n_classes, dropout=args.dropout)
elif args.encoder == 'resnet34':
    encoder = models.resnet34()
    model = GDVM_ResNet(encoder, latent=args.size_latent,
                        out=n_classes, dropout=args.dropout)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(model)
print('Total number of params in model: {}'.format(n_params))
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
poses = [0, 1, 2, 3, 4, 5, 6, 7, 8]
eval_thresholds = np.arange(0.05, 1, 0.05)
tsfm = ToTensor()

oname = 'gdvm_' + args.criterion + args.aus + '_encoder_' + args.encoder + '_alpha_' + \
    str(args.alpha) + '_beta_' + \
    str(args.beta) + '_latent_' + \
    str(args.size_latent) + '_lr_' + str(args.lr)

print(oname)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True, threshold=0.01)
logger = Logger('./logs/post_final/'+oname+'/')

''' If just quick test reduce the training/test subjcets and poses to minimum '''
(t_subs_tr, t_subs_te) = (['F001', 'F002', 'F003'], ['F007']
                          ) if args.quick_test else (None, None)
poses = ([6]) if args.quick_test else poses

dt_train = Fera2017DatasetNew('/data/data1/datasets/fera2017/',
                              partition='train', tsubs=t_subs_tr, tposes=poses, transform=tsfm)
sampler_train = TripletSampler(
    dt_train, batch_size=args.batch_size, min_n_items=10)
dl_train = DataLoader(dt_train, sampler=sampler_train, batch_size=args.batch_size,
                      num_workers=4)

dl_test, n_iter_test, sampler_test = [], [], []
for pose in poses:
    dt_test = Fera2017DatasetNew('/data/data1/datasets/fera2017/',
                                 partition='validation', tsubs=t_subs_te,  tposes=[pose], transform=tsfm)
    n_iter_test.append(len(dt_test)/args.batch_size)
    sampler_test = TripletSampler(
        dt_test, batch_size=args.batch_size, min_n_items=10)
    dl_test.append(DataLoader(dt_test, sampler=sampler_test,
                              batch_size=args.batch_size, num_workers=4))

n_iter_train = len(dt_train)/args.batch_size
n_iter_test_total = np.sum(n_iter_test)
print('n_iter in train : {}'.format(n_iter_train))
print('n_iter in test per pose: {}, total: {}'.format(
    n_iter_test, n_iter_test_total))


def train(criterion, metric, regularization):
    model.train()
    acc_loss, acc_crit, acc_reg = [], [], []

    for iter, (data, target, _) in enumerate(dl_train):
        target = torch.clamp(target[:, aus], 0, 1)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).float()

        optimizer.zero_grad()
        pred, mu, logvar = model(data)

        crit_val = criterion(pred, target) / len(data)
        metric_val = metric(mu[0:1, :], mu[1:2, :], mu[2:, :])
        reg_val = regularization(mu, logvar) / len(data)

        loss = crit_val + args.alpha*metric_val + args.beta*reg_val
        loss.backward()
        optimizer.step()

        if iter % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.4f} + {}*{:.8f} + {}*{:.8f} = {:.4f}'.format(
                epoch, iter, n_iter_train, crit_val.data, args.alpha, metric_val.data, args.beta, reg_val.data, loss.data))

            info = {
                'loss_train': loss.data,
                'crit_train': crit_val.data,
                'reg_train': reg_val.data
            }

            for tag, value in info.items():
                logger.scalar_summary(
                    tag, value, n_iter_train*(epoch-1)+iter+1)

        acc_loss.append(loss.data.cpu().numpy())
        acc_crit.append(crit_val.data.cpu().numpy())
        acc_reg.append(reg_val.data.cpu().numpy())

    return np.mean(acc_loss), np.mean(acc_crit), np.mean(acc_reg)


def test(criterion, metric, regularization, n_runs):
    model.eval()
    f1s, acc_loss, acc_crit, acc_reg = [], [], [], []
    for i, dl_test_pose in enumerate(dl_test):
        targets, preds = [], []
        print(
            '-----------------------------------Evaluating POSE {} ------------------------- '.format(poses[i]))
        for iter, (data, target, _) in enumerate(dl_test_pose):
            target = torch.clamp(target[:, aus], 0, 1)
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data).float(
            ), Variable(target).float()

            with torch.no_grad():
                ''' Pass n_runs times through model '''
                pred, mu, logvar = model(data)

                ''' Compute loss every pass '''
                pred = F.sigmoid(pred)
                crit_val = criterion(pred, target) / len(data)
                reg_val = regularization(mu, logvar) / len(data)
                loss = crit_val + args.beta*reg_val

                acc_crit.append(crit_val.data)
                acc_reg.append(reg_val.data)
                acc_loss.append(loss.data)

                preds.append(pred)
                targets.append(target.data.cpu().numpy())

        preds = np.asarray(np.concatenate(preds))
        print('preds min:{}, max:{}, mean:{}'.format(
            preds.min(), preds.max(), np.mean(preds)))
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

    ''' Log validation loss '''
    info = {'loss_test': np.mean(acc_loss),
            'crit_test': np.mean(acc_crit),
            'reg_test': np.mean(acc_reg)}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    ''' Log F1 per threshold'''
    f1s = np.mean(f1s, axis=0)
    for i, t in enumerate(eval_thresholds):
        info = {'f1_val_t_'+str(t): f1s[i]}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

    return np.mean(acc_loss), np.mean(acc_crit), np.mean(acc_reg),  f1s


def update_params(param, crit, patience=3, update=0.5):
    var = 0.1
    if epoch > patience and np.mean(crit[-patience:])/crit[-1] < (1+var):
        param = (1+update)*param
        print('We are on a plateau. Updating parameter to: {}'.format(param))
    else:
        print('No update.')

    return param


crit_train_list = []
for epoch in range(1, args.epochs+1):
    ''' Define criterion and regularization in loss '''
    regularization = kld
    criterion = bce_loss
    metric = npair_loss

    loss_train, crit_train, reg_train = train(
        criterion, metric, regularization)

    ''' Decrease learning rate if on plateau'''
    scheduler.step(crit_train)

    loss_test, crit_test, reg_test, f1s = test(
        criterion, metric, regularization, n_runs=1)

    print('\n====> Train loss: {:.4f}'.format(loss_train))
    print('====> Train criterion: {:.4f}'.format(crit_train))
    print('====> Train regularization: {:.4f}'.format(reg_train))
    print('====> Test loss: {:.4f}'.format(loss_test))
    print('====> Test criterion: {:.4f}'.format(crit_test))
    print('====> Test regularization: {:.4f}\n'.format(reg_test))

    print('\n====> Mean F1(t=0.5): {}\n'.format(f1s[9]))

    # Save checkpoint
    '''
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, True, 'models/final/' + oname + '_epoch_' + str(epoch))
    '''
    crit_train_list.append(crit_train)
