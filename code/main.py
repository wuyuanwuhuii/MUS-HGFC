from torch import nn, optim, save
from prepareData import prepare_data
from model import Model
from trainData import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
import torch


import os, sys

os.chdir(sys.path[0])
  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Config(object):
    def __init__(self):
        self.data_path = './datasets/'
        self.validation = 1
        self.save_path = './datasets/'
        self.epoch = 500
        self.alpha = 0.1


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1-opt.alpha)*loss_sum[one_index].sum()+opt.alpha*loss_sum[zero_index].sum()


class Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['data'].size(0)
        self.d = dataset['dd']['data'].size(0)
        self.fg = 256
        self.fd = 256
        self.k = 32


def train(model, train_data, optimizer, opt, cha_index, cha_index0):
    model.train()
    regression_crit = Myloss()
    one_index = train_data[2][0].cuda().to(device).tolist()
    zero_index = train_data[2][1].cuda().to(device).tolist()
    SUM=[]

    def train_epoch():
        model.zero_grad()
        scores = model(train_data)
        loss = regression_crit(one_index, zero_index, train_data[4].cuda(), scores)
        loss.backward()
        optimizer.step()
        return loss, scores
    for epoch in range(1, opt.epoch+1):
        if len(SUM) == 0:
            save(model.state_dict(), 'HGNN_MDA_all5.pt')
        train_reg_loss, scores = train_epoch()
        score1=scores
        print(train_reg_loss.item() / (len(one_index[0]) + len(zero_index[0])))
        if epoch == 100:
            dataPre = []
            dataAct = []
            # for ind in zero_index:
            #     dataPre.append(scores[ind[0], ind[1]].data.cpu().numpy())
            # for ind in one_index:
            #     dataAct.append(scores[ind[0], ind[1]].data.cpu().numpy())
            for ind in cha_index:
                dataAct.append(1)
                dataPre.append(scores[ind[0], ind[1]].data.cpu().numpy())
            for ind0 in cha_index0:
                dataAct.append(0)
                dataPre.append(scores[ind0[0], ind0[1]].data.cpu().numpy())
            # 绘制ROC/AUC
            act = np.array(dataAct)
            pre = np.array(dataPre)
            # print(pre)
            FPR, TPR, thresholds = roc_curve(act, pre)

            AUC = auc(FPR, TPR)
            print('AUC:', AUC)

            # 严格定义计算方法
            precision, recall, thresholds = precision_recall_curve(dataAct, dataPre)
            PR = auc(recall, precision)
            print("PR: ", PR)

            dataPre = np.around(dataPre, 0).astype(int)
            f1_weighted = f1_score(dataAct, dataPre, average='weighted')
            f1_macro = f1_score(dataAct, dataPre, average='macro')
            print("f1-score: 考虑类别的不平衡性为{}, 不考虑类别的不平衡性为{}".format(f1_weighted, f1_macro))
            Sum = f1_weighted + AUC + PR
            Sum = Sum / 3

            SUM.append(Sum)
            print(SUM)

    return scores


opt = Config()


def main():
    global score_all, score1
    dataset, cha_index, cha_index0, cha_index1, cha_index2 = prepare_data(opt)
    sizes = Sizes(dataset)
    # train_data = Dataset(opt, dataset)

    #model = Model(sizes)

    #model = Model()
    for i in range(opt.validation):
        print('-'*50)
        model = Model(sizes)
        model.cuda()
        train_data = Dataset(opt, dataset)
        train_data[0][0]['data'] = torch.Tensor(train_data[0][0]['data']).to(device)
        train_data[0][1]['data'] = torch.Tensor(train_data[0][1]['data']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        score1=train(model, train_data[i], optimizer, opt, cha_index, cha_index0)
    score2=score1
    m_state_dict = torch.load('HGNN_MDA_all5.pt')
    model.load_state_dict(m_state_dict)
    #scores = model(train_data)

    print("________________________________________________")
    dataPre = []
    dataAct = []
    for ind in cha_index:
        dataAct.append(1)
        dataPre.append(score1[ind[0], ind[1]].data.cpu().numpy())
    for ind0 in cha_index0:
        dataAct.append(0)
        dataPre.append(score1[ind0[0], ind0[1]].data.cpu().numpy())
    # 绘制ROC/AUC
    act = np.array(dataAct)
    pre = np.array(dataPre)
    # print(pre)
    FPR, TPR, thresholds = roc_curve(act, pre)

    AUC = auc(FPR, TPR)
    print('AUC:', AUC)

    # 严格定义计算方法
    precision, recall, thresholds = precision_recall_curve(dataAct, dataPre)
    PR = auc(recall, precision)
    print("PR: ", PR)

    dataPre = np.around(dataPre, 0).astype(int)
    f1_weighted = f1_score(dataAct, dataPre, average='weighted')
    f1_macro = f1_score(dataAct, dataPre, average='macro')
    print("f1-score: 考虑类别的不平衡性为{}, 不考虑类别的不平衡性为{}".format(f1_weighted, f1_macro))


if __name__ == "__main__":
    main()
