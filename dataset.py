import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from facepp.reader.reader import read_csv
import h5py
import random


class Fera2017Dataset(Dataset):
    def __init__(self, root_dir, partition='train', tsubs=None, tposes=None, transform=None, verbose=False):
        self.verbose = verbose
        self.root_dir = root_dir + '/'
        self.partition = partition
        self.n_poses = 9
        self.partition_key = 'TR' if partition == 'train' else 'VA'

        f = h5py.File(self.root_dir+'fera_train.h5',
                      'r') if self.partition == 'train' else h5py.File(self.root_dir+'fera_test.h5', 'r')
        self.subjects = tsubs if tsubs else f[self.partition_key].keys()
        self.poses = tposes if tposes else range(self.n_poses)
        self.idxmap = self.get_idxmap_from_h5py()
        self.transform = transform

    def get_idxmap_from_csv(self):
        ''' Map 'subject/task/pose' to index _list'''
        idx_map = {}
        offset = 0
        for sub in self.subjects:
            '''Get all files from sub'''
            files = sorted([f for f in os.listdir(
                self.root_dir+self.partition+'/occ') if sub in f])

            for f in files:
                (dataset, partition, subject, task) = f.split(
                    '.')[0].split('_')

                n = np.asarray(read_csv(self.root_dir+self.partition+'/occ/'+f)
                               )[1:, 1:].shape[0]

                for pose in self.poses:
                    ''' Add entry to dictionary and update offset'''
                    idx_map[self.partition_key + '/' + sub + '/' + task + '/' +
                            str(pose)] = np.arange(offset, offset+n)
                    if self.verbose:
                        print('{}: ({}, {}), {}'.format(
                            sub+'/'+task+'/'+str(pose), offset, offset+n, n))

                    offset = offset + n

        return idx_map

    def get_idxmap_from_h5py(self):
        idx_map = {}
        offset = 0

        path = self.root_dir + \
            'fera_train.h5' if self.partition == 'train' else self.root_dir+'fera_test.h5'

        with h5py.File(path, 'r') as hf:
            for sub in self.subjects:
                for task in hf[self.partition_key+'/'+sub].keys():
                    for pose in self.poses:
                        key = self.partition_key+'/'+sub+'/'+task+'/'+str(pose)

                        if key in hf:
                            n = hf[key]['faces'].shape[0]
                            idx_map[key] = np.arange(offset, offset+n)

                            if self.verbose:
                                print('{}: ({}, {}), {}'.format(
                                    sub+'/'+task+'/'+str(pose), offset, offset+n, n))

                            offset = offset + n
        return idx_map

    def __len__(self):
        return np.sum([v.size for v in self.idxmap.itervalues()])

    def __getitem__(self, idx):
        ''' Write batch sampling code here'''
        for k, v in self.idxmap.iteritems():
            if idx >= v.min() and idx <= v.max():
                key, i = k, idx-v.min()

        path = self.root_dir + \
            'fera_train.h5' if self.partition == 'train' else self.root_dir+'fera_test.h5'

        with h5py.File(path, 'r') as hf:
            x = hf[key]['faces'][i]
            y = hf[key]['aus'][i]

        if self.transform:
            (x, y) = self.transform((x, y))

        return (x, y, key)


class Fera2017DatasetTriplet(Dataset):
    def __init__(self, root_dir, partition='train', tsubs=None, tposes=None, transform=None, verbose=False):
        self.verbose = verbose
        self.root_dir = root_dir + '/'
        self.partition = partition
        self.n_poses = 9
        self.partition_key = 'TR' if partition == 'train' else 'VA'
        self.hf = h5py.File(self.root_dir+'fera_train.h5',
                            'r') if self.partition == 'train' else h5py.File(self.root_dir+'fera_test.h5', 'r')
        self.subjects = tsubs if tsubs else self.hf[self.partition_key].keys()
        self.poses = tposes if tposes else range(self.n_poses)
        self.dataset_idx = self.get_dataset_idx()
        self.idx_pn = self.get_idx_pn(verbose)
        self.transform = transform

    def get_dataset_idx(self):
        map = {}
        offset = 0

        for sub in self.subjects:
            for task in self.hf[self.partition_key+'/'+sub].keys():
                for pose in self.poses:
                    key = self.partition_key+'/'+sub+'/'+task+'/'+str(pose)

                    if key in self.hf:
                        aus = np.array(self.hf[key]['aus'])
                        aus = aus[np.where(aus.any(axis=1))[0]]
                        n = aus.shape[0]

                        print(np.array(self.hf[key]['aus'].shape), n)

                        '''
                        PROBLEM HERE with the indexing
                        '''

                        map[key] = {'idx': np.arange(offset, offset+n),
                                    'aus': aus}

                        if self.verbose:
                            print('{}: ({}, {}), {}'.format(
                                sub+'/'+task+'/'+str(pose), offset, offset+n, n))

                        offset = offset + n

        return map

    def get_idx_pn(self, verbose=False):
        ''' For every sample in the dataset find a positive and a negative (a, p, n)'''

        map = []
        for x_a in np.arange(len(self)):
            if x_a % 100 == 0:
                print(x_a)

            _, y_a, _, _ = self.get(x_a)

            ''' Get active classes from anchor '''
            active_classes = np.where(y_a == 1)[0]

            ''' Randomly pick pivot class from one of the active classes '''
            if len(active_classes) > 0:
                pivot_class = active_classes[np.random.randint(
                    0, len(active_classes))]

                ''' Get positive '''
                x_p = self.get_positive(x_a, pivot_class, verbose=False)

                ''' Get negative '''
                x_n = self.get_negative(x_a, pivot_class, verbose=False)

            else:
                print('No active class. Return anchor as postive and negative')
                x_p, x_n = x_a, x_a

            map.append((x_p, x_n))

        return map

    def get_positive(self, idx_a, pivot, verbose=False):
        _, y_a, key_a, ridx_a = self.get(idx_a)
        if verbose:
            print('---------------------')
            print('key: {}, idx_a: {}, ridx_a:{}, pivot: {}'.format(
                key_a, idx_a, ridx_a, pivot))

        ''' Get different pose '''
        root_a, pose_a = key_a[:-1],  int(key_a.split('/')[3])

        if verbose:
            print('pose anchor: {}'.format(pose_a))

        pose_p = random.choice(list(set(self.poses) - set([pose_a])))
        key_p = root_a + str(pose_p)

        if verbose:
            print('key positive :{}'.format(key_p))

        ''' Get same samples from same subject but different pose '''
        ridx_p = ridx_a
        idx_p = self.dataset_idx[key_p]['idx'][0] + ridx_p

        if verbose:
            print('key_p: {}, idx_p: {}'.format(key_p, idx_p))

        return idx_p

    def get_negative(self, idx_a, pivot, verbose=False):
        '''
        TODO: deal with the case when no negative is located in same sequence!
        '''
        _, y_a, key_a, ridx_a = self.get(idx_a)
        if verbose:
            print('---------------------')
            print('key: {}, idx_a: {}, ridx_a:{}, pivot: {}'.format(
                key_a, idx_a, ridx_a, pivot))

        ''' Get all samples from same subject and same pose'''
        pool = np.array(self.hf[key_a]['aus'])

        if verbose:
            print('All samples: {}'.format(pool.shape))

        ''' Filter from these samples all that have pivot non-active '''
        if(pool.size > 0):
            ridxs_n = np.where(pool[:, pivot] == 0)[0]

            if ridxs_n.size > 0:
                if verbose:
                    print('Pool of potential negatives: {}'.format(
                        pool[ridxs_n].shape))

                ''' Randomply pick one '''
                ridx_n = np.random.choice(np.squeeze(ridxs_n))

                if verbose:
                    print('ridx_n: {}'.format(ridx_n))

                idx_n = self.dataset_idx[key_a]['idx'][0] + ridx_n

                if verbose:
                    print('key_n: {}, idx_n: {}'.format(key_a, idx_n))
            else:
                print('No negatives found. Return anchor.')
                idx_n = idx_a

        return idx_n

    def __len__(self):
        return np.sum([v['idx'].size for v in self.dataset_idx.itervalues()])

    def get(self, idx):
        for k, v in self.dataset_idx.iteritems():
            if idx >= v['idx'].min() and idx <= v['idx'].max():
                key, i = k, idx-v['idx'].min()

        x = self.hf[key]['faces'][i]
        y = self.hf[key]['aus'][i]

        return x, y, key, i

    def __getitem__(self, idx):
        idx_p, idx_n = self.idx_pn[idx]

        tr = [self.get(x) for x in [idx, idx_p, idx_n]]

        if self.transform:
            t_tr = [self.transform((x, y)) for x, y, _, _ in tr]

        return {'a': {'x': t_tr[0][0], 'y': t_tr[0][1]},
                'p': {'x': t_tr[1][0], 'y': t_tr[1][1]},
                'n': {'x': t_tr[2][0], 'y': t_tr[2][1]}
                }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        im, aus = sample[0], sample[1]

        # swap color axis because and range between 0 and 1
        # numpy image: H x W x C
        # torch image: C X H X W
        '''
        im = skimage.transform.resize(
            im, (28, 28), preserve_range=True).astype(np.uint8)
        '''
        im = im.transpose((2, 0, 1))
        return (torch.from_numpy(im),  torch.from_numpy(aus))


if __name__ == '__main__':
    tsfm = ToTensor()
    '''
    dt_tr = Fera2017DatasetTriplet('/data/data1/datasets/fera2017/',
                                   partition='train', tsubs=['F001', 'F002'],
                                   tposes=[1, 6, 7], transform=tsfm, verbose=True)

    for i in range(0, len(dt_tr), 100):
        t = dt_tr[i]
        print('i:{},  a_y:{}, p_y:{}, n_y:{}'.format(
            i, t['a']['y'].numpy(), t['p']['y'].numpy(), t['n']['y'].numpy()))
    '''
    dt_tr = Fera2017Dataset('/data/data1/datasets/fera2017/',
                            partition='train', tsubs=None, tposes=None, transform=tsfm, verbose=True)

    print('----------------')
    dt_val = Fera2017Dataset('/data/data1/datasets/fera2017/',
                             partition='validation', tsubs=None, tposes=None, transform=tsfm, verbose=True)
    '''
    dl_tr = DataLoader(dt_tr, batch_size=64, shuffle=True, num_workers=4)
    dl_val = DataLoader(dt_val, batch_size=64, shuffle=True, num_workers=4)
    '''
    '''
    for i, (x, y, seq) in enumerate(dl_tr):
        print('{}: {} {}, mean = {}'.format(
            i, x.shape, y.shape, torch.mean(x)))
    '''
