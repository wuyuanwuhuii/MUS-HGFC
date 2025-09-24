import csv
import torch as t
import random
import copy

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def prepare_data(opt):
    dataset = dict()
    dataset['md_p'] = read_csv(opt.data_path + '\\m-d.csv')
    dataset['md_true'] = read_csv(opt.data_path + '\\m-d.csv')

    zero_index_old = []
    one_index_old = []
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                zero_index_old.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index_old.append([i, j])
    random.shuffle(one_index_old)
    random.shuffle(zero_index_old)

    one_index = copy.deepcopy(one_index_old)
    zero_index = copy.deepcopy(zero_index_old)

    cha_index = {}
    cha_index0 = {}
    cha_index1 = {}
    cha_index2 = {}

    i = 0
    # 超参数
    length = int(len(one_index_old) * 0.2)  # 20% 的 “1” :1086

    cha_index1[i] = one_index[i * length:(i + 1) * length]  # 分配入测试集 “1” 的量
    cha_index2[i] = zero_index[i * length:(i + 1) * length]  # 分配入测试集 “0” 的量
    zero_index.extend(one_index[i * length:(i + 1) * length])
    # one_index = one_index[2 * length:len(one_index)]
    del one_index[i * length:(i + 1) * length]

    # 超参数
    length1 = int(len(one_index) * 0.2)  # 分配入验证集的量：434
    cha_index[i] = one_index[i * length1:(i + 1) * length1]  # 分配入验证集 “1” 的量
    cha_index0[i] = zero_index[i * length1:(i + 1) * length1]  # 分配入验证集 “0” 的量
    zero_index.extend(one_index[i * length1:(i + 1) * length1])
    del one_index[i * length1:(i + 1) * length1]

    for ind in cha_index[i]:
        if dataset['md_p'][ind[0], ind[1]] != 0:
            dataset['md_p'][ind[0], ind[1]] = 0
        else:
            print("md矩阵有错！")

    zero_tensor = t.LongTensor(zero_index)
    one_tensor = t.LongTensor(one_index)
    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]

    dd_matrix = read_csv(opt.data_path + '\\d-d.csv')
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}

    mm_matrix = read_csv(opt.data_path + '\\m-m.csv')
    mm_edge_index = get_edge_index(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}
    return dataset, cha_index, cha_index0, cha_index1, cha_index2

