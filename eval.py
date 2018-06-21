__author__ = 'cipriancorneanu'

import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
import numpy as np


def evaluate_model(y_true, y_pred, verbose=True):
    n_classes = y_true.shape[1]

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
    else:
        au_map = ['']*y_true.shape[1]

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
    em, acc, hl, f1, prec, rec = accuracy_score(y_true, y_pred), np.mean(
        accs), np.mean(hls), np.mean(f1s), np.mean(precs), np.mean(recs)

    if verbose:
        print 'Exact Match = {}'.format(em)
        print 'Accuracy = {}'.format(acc)
        print 'Hamming Loss = {}'.format(hl)
        print 'F1 = {}'.format(f1)
        print 'Precision = {}'.format(prec)
        print 'Recall = {}'.format(rec)

    return em, hl, f1, prec, rec
