# -*- coding utf-8 -*-
# 作者: SMF
# 时间: 2022.07.21
import torch
from torch import nn

import torch.nn.functional as F

from models import HGNN_conv


class HGNN(nn.Module):
    #def __init__(self, in_ch_m=1208, in_ch_d=895, n_class=64, n_hid=128, dropout=0.5):
    def __init__(self, in_ch_m=940, in_ch_d=1412, n_class=64, n_hid=125, dropout=0.5):  # original set
    #def __init__(self, in_ch_m=577, in_ch_d=336, n_class=64, n_hid=128, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc_m1 = HGNN_conv(in_ch_m, n_hid)
        self.hgc_m2 = HGNN_conv(n_hid, n_class)
        self.hgc_d1 = HGNN_conv(in_ch_d, n_hid)
        self.hgc_d2 = HGNN_conv(n_hid, n_class)
        
        
        
        
    def forward(self, input, G1_Sum, G2_Sum):
        # 1. 获取所有的 K 值（即字典的键），并排序以保证顺序一致
        k_keys = sorted(G1_Sum.keys())
        
        # 2. 处理第一个 K 值，初始化 x_new 和 y_new
        first_k = k_keys[0]
        G1_first = G1_Sum[first_k]
        G2_first = G2_Sum[first_k]

        # 计算第一个超图的分量 (x)
        x = input[1]['data']
        x = F.relu(self.hgc_m1(x, G1_first))
        x = F.dropout(x, self.dropout)
        x_new = self.hgc_m2(x, G1_first)

        # 计算第一个超图的分量 (y)
        y = input[0]['data']
        y = F.relu(self.hgc_d1(y, G2_first))
        y = F.dropout(y, self.dropout)
        y_new = self.hgc_d2(y, G2_first)

        # 3. 遍历剩余的 K 值并进行拼接 (torch.cat)
        for i in k_keys[1:]:
            G1 = G1_Sum[i]
            G2 = G2_Sum[i]

            # 处理 x 方向 (m)
            x_temp = input[1]['data']
            x_temp = F.relu(self.hgc_m1(x_temp, G1))
            x_temp = F.dropout(x_temp, self.dropout)
            temp_x = self.hgc_m2(x_temp, G1)
            x_new = torch.cat((x_new, temp_x), 1)

            # 处理 y 方向 (d)
            y_temp = input[0]['data']
            y_temp = F.relu(self.hgc_d1(y_temp, G2))
            y_temp = F.dropout(y_temp, self.dropout)
            temp_y = self.hgc_d2(y_temp, G2)
            y_new = torch.cat((y_new, temp_y), 1)

        # 4. 返回最终的矩阵乘法结果
        return x_new.mm(y_new.t())   
        
        
        
        
        
        
    

    def forward1(self, input, G1_Sum, G2_Sum):
        G1 = G1_Sum[1]
        G2 = G2_Sum[1]

        x = input[1]['data']
        x = F.relu(self.hgc_m1(x, G1))
        x = F.dropout(x, self.dropout)
        x_new = self.hgc_m2(x, G1)

        y = input[0]['data']
        y = F.relu(self.hgc_d1(y, G2))
        y = F.dropout(y, self.dropout)
        y_new = self.hgc_d2(y, G2)

        for i in range(2,8): 
            G1 = G1_Sum[i]
            G2 = G2_Sum[i]

            x = input[1]['data']
            x = F.relu(self.hgc_m1(x, G1))
            x = F.dropout(x, self.dropout)
            temp_x = self.hgc_m2(x, G1)
            x_new = torch.cat((x_new, temp_x), 1)

            y = input[0]['data']
            y = F.relu(self.hgc_d1(y, G2))
            y = F.dropout(y, self.dropout)
            temp_y = self.hgc_d2(y, G2)
            y_new = torch.cat((y_new, temp_y), 1)
        return x_new.mm(y_new.t())

# class HGNN_d(nn.Module):
#     def __init__(self, in_ch=383, n_class=64, n_hid=128, dropout=0.5):
#         super(HGNN_d, self).__init__()
#         self.dropout = dropout
#         self.hgc1 = HGNN_conv(in_ch, n_hid)
#         self.hgc2 = HGNN_conv(n_hid, n_class)
#
#     def forward(self, input, G):
#         x = input[0]['data']
#         x = F.relu(self.hgc1(x, G))
#         x = F.dropout(x, self.dropout)
#         x = self.hgc2(x, G)
#         return x
