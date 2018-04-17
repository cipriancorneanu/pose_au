import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from facepp.reader.reader import read_csv
import h5py


class Fera2017Dataset(Dataset):
    def __init__(self, root_dir, partition='train', tsubs=None, tposes=None, transform=None):
        self.root_dir = root_dir + '/'
        self.partition = partition
        self.n_poses = 9
        self.partition_key = 'TR' if partition == 'train' else 'VA'

        '''
        all_subjects_train = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                              'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020',
                              'F021', 'F022', 'F023', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007',
                              'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']

        all_subjects_validation = ['F007', 'F008', 'F009', 'F010', 'F011', 'M001', 'M002', 'M003', 'M004',
                                   'M005', 'M006', 'rF001', 'rF002', 'rM001', 'rM002', 'rM003', 'rM004',
                                   'rM005', 'rM006', 'rM007']
        '''

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
                    '''
                    print('{}: ({}, {}), {}'.format(
                        sub+'/'+task+'/'+str(pose), offset, offset+n, n))
                    '''
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
                        n = hf[key]['faces'].shape[0]
                        idx_map[key] = np.arange(offset, offset+n)

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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        im, aus = sample[0], sample[1]

        # swap color axis because and range between 0 and 1
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1))/255.
        return (torch.from_numpy(im),  torch.from_numpy(aus))


if __name__ == '__main__':
    subjects_train = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                      'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020']
    subjects_validation = ['F021', 'F022', 'F023', 'M001']

    tsfm = ToTensor()
    dt_tr = Fera2017Dataset('/data/data1/datasets/fera2017/',
                            partition='train', tsubs=None, tposes=[1, 6], transform=tsfm)

    dt_val = Fera2017Dataset('/data/data1/datasets/fera2017/',
                             partition='validation', tsubs=None, tposes=[7], transform=tsfm)

    dl_tr = DataLoader(dt_tr, batch_size=64, shuffle=True, num_workers=4)
    dl_val = DataLoader(dt_val, batch_size=64, shuffle=True, num_workers=4)

    '''
    for i, (x, y, seq) in enumerate(dl_tr):
        print('{}: {} {}, mean = {}'.format(
            i, x.shape, y.shape, torch.mean(x)))
    '''
    '''
    for i in range(0, len(dt_tr), 10000):
        x, y, seq = dt_tr[i]
        x = x.squeeze()
        print(x.min())
        print(x.max())

        print('{}: {}'.format(i, y))

        scipy.misc.toimage(x, 0, 255).save(str(i)+'.jpg')
    '''
