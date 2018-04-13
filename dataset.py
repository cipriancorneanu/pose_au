import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from facepp.reader.reader import read_csv
import h5py


class Fera2017Dataset(Dataset):
    def __init__(self, root_dir, partition='train', tsubs=None, tposes=None):
        self.root_dir = root_dir + '/'
        self.partition = partition
        self.n_poses = 9
        self.partition_key = 'TR' if partition == 'train' else 'VA'

        all_subjects_train = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                              'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020',
                              'F021', 'F022', 'F023', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007',
                              'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']

        all_subjects_validation = ['F007', 'F008', 'F009', 'F010', 'F011', 'M001', 'M002', 'M003', 'M004',
                                   'M005', 'M006', 'rF001', 'rF002', 'rM001', 'rM002', 'rM003', 'rM004',
                                   'rM005', 'rM006', 'rM007']

        all_subjects = all_subjects_train if partition == 'train' else all_subjects_validation
        self.subjects = tsubs if tsubs else all_subjects
        self.poses = tposes if tposes else self.n_poses
        self.idxmap = self.get_idx_map()

    def get_idx_map(self):
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
            x = np.expand_dims(hf[key]['faces'][i], axis=0)
            y = hf[key]['aus'][i]

        return (x, y, key)


if __name__ == '__main__':

    dt_tr = Fera2017Dataset('/data/data1/datasets/fera2017/',
                            partition='train', tsubs=['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                                                      'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020'], tposes=[1, 6, 7])

    dt_val = Fera2017Dataset('/data/data1/datasets/fera2017/',
                             partition='train', tsubs=['F021', 'F022', 'F023', 'M001'],  tposes=[1, 6, 7])

    dl_tr = DataLoader(dt_tr, batch_size=64, shuffle=True, num_workers=4)
    dl_val = DataLoader(dt_val, batch_size=64, shuffle=True, num_workers=4)

    import scipy.misc

    print(dt_tr.__len__())
    for i in range(0, len(dt_tr), 10000):
        x, y, seq = dt_tr[i]
        x = x.squeeze()
        print(x.min())
        print(x.max())

        print('{}: {}'.format(i, y))

        scipy.misc.toimage(x, 0, 255).save(str(i)+'.jpg')
