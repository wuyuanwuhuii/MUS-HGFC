# -*- coding utf-8 -*-
# 作者: SMF
# 时间: 2022.08.11
import csv

import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import vstack
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.preprocessing import minmax_scale
from torch.cuda.random import manual_seed


def scaley(ymat):
    return (ymat - ymat.min()) / ymat.max()


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        manual_seed(seed)


def load_data(data, cuda):
    path = 'Dataset' + str(data)
    gdi = np.loadtxt(path + '/known_gene_disease_interaction.txt')
    ldi = np.loadtxt(path + '/known_lncRNA_disease_interaction.txt')
    rnafeat = np.loadtxt(path + '/rnafeat.txt', delimiter=',')
    rnafeat = minmax_scale(rnafeat, axis=0)
    gdit = torch.from_numpy(gdi).float()
    ldit = torch.from_numpy(ldi).float()
    rnafeatorch = torch.from_numpy(rnafeat).float()
    gl = norm_adj(rnafeat)
    gd = norm_adj(gdi.T)
    if cuda:
        gdit = gdit.cuda()
        ldit = ldit.cuda()
        rnafeatorch = rnafeatorch.cuda()
        gl = gl.cuda()
        gd = gd.cuda()

    return gdit, ldit, rnafeatorch, gl, gd


def neighborhood(feat, k):
    # compute C
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C


def normalized(wmat):
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def norm_adj(feat):
    C = neighborhood(feat.T, k=10)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        # md_data_new = ToTensor(md_data)
        # return torch.FloatTensor(md_data)
        return numpy.array(md_data)


def show_auc(ymat, data):
    path = 'Dataset' + str(data)
    # ldi = np.loadtxt('../datasets/m-d.txt', delimiter=',')
    ldi = read_csv('../datasets/MDAData/data(383-495)/m-d.csv')
    y_true = ldi.flatten()
    ymat = ymat.flatten()  # (189585)
    fpr, tpr, rocth = roc_curve(y_true, ymat)
    auroc = auc(fpr, tpr)
    plt.rc('font', family='Arial Unicode MS', size=14)
    plt.plot(fpr, tpr, label="AUC={:.2f}".format(auroc), marker='o', color='b', linestyle='--')
    plt.legend(loc=4, fontsize=10)
    plt.title('ROC曲线', fontsize=20)
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.show()
    fpr = np.array(fpr, dtype=np.int32)
    tpr = np.array(tpr, dtype=np.int32)
    np.savetxt('../save/roc.txt', vstack((fpr, tpr)), fmt='%10.5f', delimiter=',')

    precision, recall, prth = precision_recall_curve(y_true, ymat)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()  # 生成网格
    plt.plot(recall, precision)
    plt.figure("P-R Curve")
    plt.show()
    aupr = auc(recall, precision)
    np.savetxt('../save/pr.txt', np.vstack((recall, precision)), fmt='%10.5f', delimiter=',')
    print('AUROC= %.4f | AUPR= %.4f' % (auroc, aupr))
    rocdata = np.loadtxt('../save/roc.txt', delimiter=',')
    prdata = np.loadtxt('../save/pr.txt', delimiter=',')
    dataPre = np.around(ymat, 0).astype(int)
    f1_weighted = f1_score(y_true, dataPre, average='weighted')
    print("f1-score: {}".format(f1_weighted))
    # plt.figure()
    # plt.plot(rocdata[0], rocdata[1])
    # plt.plot(prdata[0], prdata[1])
    # precision, recall, thresholds = precision_recall_curve(dataAct, dataPre)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.grid()  # 生成网格
    # plt.plot(recall, precision)
    # plt.figure("P-R Curve")
    # plt.show()
    return auroc, aupr
