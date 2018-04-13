__author__ = 'cipriancorneanu'

import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
import numpy as np


def evaluate_model(y_true, y_pred, verbose=True):
    n_classes = y_true.shape[1]
    print n_classes

    # show some predictions
    idxs = random.sample(xrange(0, len(y_true)), 100)
    '''
    for y, y_hat in zip(y_true[idxs], y_pred[idxs]):
        print ('gt: {} , pred: {}'.format(y, y_hat))
    '''

    # compute scores per AU
    accs, f1s, hls, recs, precs = ([], [], [], [], [])
    if n_classes == 12:
        au_map = ['01', '02', '04', '06', '07',
                  '10', '12', '14', '15', '17', '23', '24']
    elif n_classes == 10:
        au_map = ['01', '04', '06', '07', '10', '12', '14', '15', '17', '24']
    elif n_classes == 7:
        au_map = ['10', '12', '14', '15', '17', '23', '24']
    elif n_classes == 5:
        au_map = ['01', '02', '04', '06', '07']
    elif n_classes == 3:
        au_map = ['', '', '']
    elif n_classes == 1:
        au_map = ['']

    for au in range(n_classes):
        if verbose:
            print '----- AU{} ------'.format(au_map[au])

        au_test = y_true[:, au]
        au_test_hat = y_pred[:, au]

        acc = accuracy_score(au_test, au_test_hat)
        f1 = f1_score(au_test, au_test_hat, average='binary')
        hl = hamming_loss(au_test, au_test_hat)
        rec = recall_score(au_test, au_test_hat, average='binary')
        prec = precision_score(au_test, au_test_hat, average='binary')

        accs.append(acc)
        f1s.append(f1)
        hls.append(hl)
        recs.append(rec)
        precs.append(prec)

        if verbose:
            print '\t\tAccuracy = {}'.format(acc)
            print '\t\tF1 = {}'.format(f1)
            print '\t\tHamming Loss = {}'.format(hl)
            print '\t\tPrecision = {}'.format(prec)
            print '\t\tRecall = {}'.format(rec)

    # compute average scores
    print 'Exact Match = {}'.format(accuracy_score(y_true, y_pred))
    print 'Accuracy = {}'.format(np.mean(accs))
    print 'Hamming Loss = {}'.format(np.mean(hls))
    print 'F1 = {}'.format(np.mean(f1s))
    print 'Precision = {}'.format(np.mean(precs))
    print 'Recall = {}'.format(np.mean(recs))

    # filter exact matches
    exact_matches = []
    for sample, sample_hat in zip(y_true, y_pred):
        if np.array_equal(sample, sample_hat):
            exact_matches.append(sample_hat)

    # show some random exact matches
    '''
    idxs = random.sample(xrange(0,len(exact_matches)), 100)
    for x in idxs:
        print 'Exact match sample: {}'.format(exact_matches[x])
    '''
