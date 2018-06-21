import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from facepp.reader.reader import read_csv
import h5py
import random
import time
import cPickle as pkl
from operator import itemgetter


class TripletSampler(Sampler):
    def __init__(self, data_source, batch_size=32, npair=False, min_nitems=2, nclasses=None, remove_neutral=True):
        self.data_source = data_source
        self.labels = [[x[0], x[1]] for x in data_source.get_labels_idx()]
        self.batch_size = batch_size
        self.npair = npair
        self.min_nitems = min_nitems
        self.n_classes = nclasses
        self.idxs = self.generate_indxs(
            batch_size, remove_neutral, nclasses, min_nitems, npair)

    def generate_indxs(self, batch_size=32, remove_neutral=True, nclasses=None, min_nitems=2, npair=False):
        ''' Remove  neutral '''
        if remove_neutral:
            labels = [x for x in self.labels if x[0] != '0000000000']

        ''' Pick first nclasses '''
        if nclasses:
            labels = labels[:nclasses]

        idxs, idx_pool = [], []

        print([[x[0], len(x[1])] for x in labels])

        while len(labels) > self.batch_size/2:
            ''' Remove labels with fewer than 2 samples '''
            labels = [[x[0], x[1]]
                      for x in labels if len(x[1]) >= min_nitems]

            '''print([[x[0], len(x[1])] for x in labels])'''

            '''
            print('No. of remaining samples: {}'.format(
                np.sum([len(x[1]) for x in labels])))
            '''
            ''' Pick labels following label occurence distribution '''
            label_choice = np.random.choice(
                np.concatenate([[i]*len(x[1]) for i, x in enumerate(labels)]), size=batch_size/2)

            for l in label_choice:
                ''' Randomly pick 2 indices '''
                ind = np.random.choice(labels[l][1], 2)

                ''' Update indices list '''
                labels[l][1] = list(set(labels[l][1]) - set(ind))

                idx_pool.append(ind)

            ''' From the sampled pool create mini_batches of indices '''
            if npair:
                idxs.append(self._generate_npair(np.concatenate(idx_pool)))
            else:
                idxs.append(np.concatenate(idx_pool))
            idx_pool = []

        return np.concatenate(idxs)

    def __iter__(self):
        return iter(self.idxs)

    def _generate_npair(self, pool):
        heads, tails = ([[x, x+1] for x in range(len(pool))[::2]],
                        [range(0, x)+range(x+2, len(pool)) for x in range(len(pool))[::2]])
        return np.concatenate([np.hstack((pool[h], pool[t])) for h, t in zip(heads, tails)])

    def __len__(self):
        return len(self.idxs)


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

    def get_idxmap_from_h5py(self):
        idx_map = []
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

                            for i in range(n):
                                item = {'idx': offset+i, 'ridx': i,
                                        'label': ''.join(str(e) for e in hf[key]['aus'][i]),
                                        'key': key}
                                idx_map.append(item)

                                if self.verbose:
                                    print('{}:{}'.format(offset+i, item))

                            offset = offset + n
        return idx_map

    def get_label(self, label):
        '''Return items for a specific label '''
        return [item for item in self.idxmap if item['label'] == label]

    def get_label_field(self, label, field):
        ''' Return indices for a specific label'''
        return [item[field] for item in self.idxmap if item['label'] == label]

    def get_labels_occ(self):
        ''' Return a list of all individual labels sorted by number of occurences in the dataset '''
        labels = list(set([item['label'] for item in self.idxmap]))
        return sorted([(l, len(self.get_label(l)))
                       for l in labels], key=itemgetter(1), reverse=True)

    def get_labels_idx(self):
        '''
        Return a list of all individual labels and their indices
        sorted by number of occurences in the dataset
        '''
        labels_occ = self.get_labels_occ()
        return [(x[0], self.get_label_field(x[0], 'idx')) for x in labels_occ]

    def __len__(self):
        return np.sum(len(self.idxmap))

    def __getitem__(self, idx):
        ''' Write batch sampling code here'''
        item = self.idxmap[idx]

        path = self.root_dir + \
            'fera_train.h5' if self.partition == 'train' else self.root_dir+'fera_test.h5'

        with h5py.File(path, 'r') as hf:
            x = hf[item['key']]['faces'][item['ridx']]
            y = hf[item['key']]['aus'][item['ridx']]

        if self.transform:
            (x, y) = self.transform((x, y))

        return (x, y, item['key'])


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


def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def explore_labels():
    print('Total number of samples : {}'.format(len(dt_tr)))

    ''' Get complete list of labels  '''
    labels = dt_tr.get_all_labels()
    labels_occ = sorted([(l, len(dt_tr.get_label(l)))
                         for l in labels], key=itemgetter(1), reverse=True)
    print('The complete list of labels is : {}'.format(labels_occ))
    print('There are {} individual labels'.format(len(labels_occ)))

    labels_occ_100 = [x[1] for x in labels_occ if x[1] >= 100]
    print('labels_100: n_labels {}: n_samples {}'.format(
        len(labels_occ_100), np.sum(labels_occ_100)))

    ''' Get ordered list of labels according to hamming distance to anchor'''
    anchor = '0000000000'
    h_list = [(l, hamming_distance(anchor, l))
              for l in labels]
    print('List h_distance from anchor {} : {}'.format(anchor, h_list))

    ''' Get all samples for label '''
    for l in labels_occ[:10]:
        pool = dt_tr.get_label(l[0])

        '''Sample at most n_samples'''
        n_samples = 64
        pool = [pool[x] for x in np.random.choice(np.arange(len(pool)),
                                                  min(len(pool), n_samples))]
        to_save = [dt_tr[x['idx']] for x in pool]

        f = open(l[0] + '_' + str(l[1]) + '.pkl', 'wb')
        pkl.dump({'dt': to_save}, f, protocol=pkl.HIGHEST_PROTOCOL)

    ''' Get all samples  at H distance from anchor '''
    for h in range(10):
        labels_h = [x[0] for x in h_list if x[1] == h]
        print('{} labels at distance {} from anchor : {}'.format(
            len(labels_h), h, labels_h))

        pool = np.concatenate([dt_tr.get_label(l) for l in labels_h])

        print(len(pool))

        '''Sample at most n_samples'''
        n_samples = 64
        pool = [pool[x] for x in np.random.choice(np.arange(len(pool)),
                                                  min(len(pool), n_samples))]
        to_save = [dt_tr[x['idx']] for x in pool]

        f = open(anchor + '_' + str(h)+'.pkl', 'wb')
        pkl.dump({'dt': to_save}, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    tsfm = ToTensor()
    start_time = time.time()
    b_sz = 32
    dt_tr = Fera2017Dataset('/data/data1/datasets/fera2017/',
                            partition='train', tposes=[6],
                            verbose=False)
    dt_te = Fera2017Dataset('/data/data1/datasets/fera2017/',
                            partition='validation', tposes=[6],
                            verbose=False)

    sampler_tr = TripletSampler(
        dt_tr, batch_size=b_sz, nclasses=20, npair=True)
    sampler_te = TripletSampler(
        dt_te, batch_size=b_sz, nclasses=20, npair=False)

    print("--- %s seconds ---" % (time.time() - start_time))

    dl_tr = DataLoader(dt_tr, sampler=sampler_tr, batch_size=b_sz)
    dl_te = DataLoader(dt_te, sampler=sampler_te, batch_size=b_sz)

    print('len dl_tr :{} '.format(len(dl_tr)))
    print('len dl_te :{} '.format(len(dl_te)))

    '''
    for iter, (x, y, _) in enumerate(dl_tr):
        print('iter: {}, {}'.format(iter, y))
    '''
