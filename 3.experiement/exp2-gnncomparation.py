import time
import gc
from copy import deepcopy as dcopy
import warnings
import torch.nn.functional as F
import torch.nn as nn
import torch
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from random import randint, uniform
from sklearn import preprocessing
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd 
import sys
import os
import pickle
import random
import lightgbm as lgb
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
#import import_ipynb
from LightGBMClassification import lgb_train_model_with_split, print_res
from saveFile import load_pickle, save_pickle
import pickle
import os

RAWDATA_PATH = './dataset/rawdata/'
PUBLICDATA_PATH = './dataset/publicdata/'

SAMPLE_SIZE = 50000
RANDOM_SEED = 2019
DAYS, HOURS, MINUTES, SECOND = 30, 24, 60, 60
INTERVAL = DAYS*HOURS * MINUTES * SECOND

interPathName = "./snapshotdata/graph_%d/interval_%d" % (SAMPLE_SIZE, INTERVAL)
mulgPathName = interPathName + "/mulG"
muldigPathName = interPathName+"/mulDiG"
featurePathName = interPathName+"/features"
edgePathName = interPathName+"/edges"
weightedEdgePathName = interPathName+"/weighted_edges"
deltaFeaturePathName = interPathName+"/delta_features"
tolFeaturePathName = interPathName + "/tol_features"

embPathName = "./embeddings/graph_%d/interval_%d" % (SAMPLE_SIZE, INTERVAL)
GCNPathName = embPathName+"/GCNembedding_test"
LINEPathName = embPathName+"/Line"
N2VPathName= embPathName+"/N2V"
DeepwalkPathName= embPathName+"/Deepwalk"
DeepwalkPathName_new= embPathName+"/DeepWalkemb"
LINEPathName_new = embPathName+"/Lineemb"
N2VPathName_new= embPathName+"/N2Vemb"
subSum = sum([len(files) for root, dirs, files in os.walk(mulgPathName)])
print(subSum)
testNum = 30
# 8 for 2 layer gcn
# 30 for 
FeatureSize = 16
fSize=8

def read_embeds(fname):
    with open(fname, 'r') as f:
        """skip the first row， first col"""
        data = f.readline()
        data = f.readlines()
        npdata = np.loadtxt(data, float, delimiter=' ')
        argsort = np.argsort(npdata[:, 0])
        npdata = npdata[argsort].tolist()
        npdata = np.delete(npdata, 0, axis=1)
        return npdata

def normalize(A):
    lena = A.shape[0]
    row = [i for i in range(lena)]
    col, data = row.copy(), [1 for i in range(lena)]
    eye_mat = coo_matrix((data, (row, col)), shape=(lena, lena))
    A = A + eye_mat
    d = np.array(A.sum(1))
    d = np.power(A.sum(1), -0.5)
    d = np.ravel(d)
    i = [j for j in range(lena)]
    D = coo_matrix((d, (i, i)), shape=(lena, lena))

    scipy_mat = (D * A * D).tocoo()
    return scipy_mat


def scipy_tensor(scipy_mat):
    row, col, data = scipy_mat.row, scipy_mat.col, scipy_mat.data
    lena = scipy_mat.shape[0]
    indice, data = torch.LongTensor([row, col]), torch.FloatTensor(data)
    torch_mat = torch.sparse.FloatTensor(
        indice, data, torch.Size([lena, lena]))
    return torch_mat


def get_input_vars():
    global train_x, train_y, scipy_adj_matrix
    adj_matrix = dcopy(scipy_adj_matrix)
    # print(adj_matrix)
    A_normed = normalize(adj_matrix)
    A_normed = scipy_tensor(A_normed)
    adj_mat = scipy_tensor(adj_matrix)
    X, Y = dcopy(train_x), dcopy(train_y)
    X, Y = torch.from_numpy(X.values).float(
    ), torch.from_numpy(Y.values).long()
    return X, Y, A_normed, adj_mat



def get_scipy_adj_matrix(sp_muldG):
    row, col, data = [], [], []
    for ind, edge in enumerate(nx.edges(sp_muldG)):
        (u, v) = edge
        row.append(u)
        col.append(v)
        data.append(1)
    adj_matrix_coo = coo_matrix((np.array(data), (np.array(
        row), np.array(col))), shape=(SAMPLE_SIZE, SAMPLE_SIZE))
    # print(adj_matrix_coo)
    return adj_matrix_coo


def prob_res(pred):
    pred_np = pred.detach().numpy()
    return np.argmax(pred_np, axis=1)


class GCN(nn.Module):
    def __init__(self, DAD, dim_in, dim_out, random_seed):
        super(GCN, self).__init__()
        torch.manual_seed(random_seed)
        self.DAD = DAD
        self.fc1 = nn.Linear(dim_in, dim_out,  bias=False)

    def forward(self, X, flag=1):
        # print("X:",X.shape[0],X.shape[1])
        #print("DAD:",self.DAD.shape[0], self.DAD.shape[1])
        X = torch.tanh(self.fc1(self.DAD.mm(X)))
        # X是返回的embedding结果
        return X


def gcn_train(X, Y, A_normed, A, epoch, lr, weight_decay, esize, random_seed):
    dim_n, dim_f = X.shape[0], X.shape[1]
    #print("dim_f:", X.shape[1])
    #print(A_normed.shape[0], A_normed.shape[1])
    gcn_model = GCN(A_normed, dim_f, esize, random_seed=random_seed)
    optim = torch.optim.Adam(gcn_model.parameters(),
                             lr=lr, weight_decay=weight_decay)
    for i in range(epoch):
        Z = gcn_model(X, flag=1)
        adj_dec = Z.mm(Z.t())

        loss = torch.norm(adj_dec - A, p='fro')
        loss = torch.pow(loss, 2) / (dim_n)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return Z


def get_GCN_embedding(epoch=10, lr=0.02, weight_decay=2e-6, esize=8, random_seed=RANDOM_SEED):
    global train_x, train_y
    
    X_new, Y_new, A_normed_new, adj_mat_new = get_input_vars()
    print("start training")
    X, Y, A_normed, adj_mat = dcopy(X_new), dcopy(
        Y_new), dcopy(A_normed_new), dcopy(adj_mat_new)

    embed_gcn = gcn_train(X, Y, A_normed, adj_mat, epoch=epoch, lr=lr,
                          weight_decay=weight_decay, esize=esize, random_seed=random_seed)
    embed_feas = embed_gcn.detach().numpy()
    return embed_feas

def get_input_data(epathName, efileName, tolFlag, splitFlag, lastFlag):
    tolX = np.zeros((SAMPLE_SIZE, subSum, FeatureSize))
    tolY = np.zeros((SAMPLE_SIZE, subSum, 1))
    if(lastFlag == True):
        start = subSum-1
    else:
        start = 0
    for idx in range(start, subSum):
        embedding = load_pickle(epathName, efileName+"_%d.pkl" % idx)
        embedding_df = pd.DataFrame(embedding)
        deltaFeature_df = load_pickle(
            deltaFeaturePathName, "/deltaFeature_%d.pkl" % idx)
        tolFeature_df = load_pickle(
            tolFeaturePathName, "/tolFeature_%d.pkl" % idx)
        # 合并两个特征
        if(tolFlag == True):
            aggreFeature = pd.concat([embedding_df, tolFeature_df], axis=1)
        else:
            aggreFeature = pd.concat([embedding_df, deltaFeature_df], axis=1)
        y_cols_name = ['label']
        x_cols_name = [x for x in aggreFeature.columns if x not in y_cols_name]
        subX = dcopy(aggreFeature[x_cols_name]).values
        subY = dcopy(aggreFeature[y_cols_name]).values
        tolX[:, idx, :] = subX
        tolY[:, idx, :] = subY
    if(splitFlag == True):
        trainX, testX, trainY, testY = train_test_split(
            tolX, tolY, test_size=0.2, random_state=RANDOM_SEED)
    else:  # False用于无监督情况
        trainX, trainY, testX, testY = tolX, tolY, 0, 0
    return trainX, trainY, testX, testY

def GCN_tol_embedding():
    for idx in range(0, subSum):
        print(idx)
        tolFeature = load_pickle(tolFeaturePathName, "/tolFeature_%d.pkl" % idx)
        tolFeature_df = pd.DataFrame(
            tolFeature, columns=['label', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8'])
            
        sp_muldG = load_pickle(muldigPathName, "/G_%d.pkl" % idx)
        y_cols_name = ['label']
        x_cols_name = [x for x in tolFeature_df.columns if x not in y_cols_name]
        global scipy_adj_matrix,train_x,train_y
        train_x = dcopy(tolFeature_df[x_cols_name])
        train_y = dcopy(tolFeature_df[y_cols_name])
        pos_cnt, neg_cnt = int(train_y.sum()), int(len(train_y) - train_y.sum())
            
        scipy_adj_matrix = get_scipy_adj_matrix(sp_muldG)
        print('pos node cnts:', pos_cnt)
        print('neg node cnts:', neg_cnt, 'pos/all ratio:',
            pos_cnt / (pos_cnt + neg_cnt))

        embSize = 8
        fGCNembedding = get_GCN_embedding(epoch=6, lr=0.005, weight_decay=1e-6,
                                    esize=embSize, random_seed=7)
        print("finish calculate embedding data!")
        save_pickle(fGCNembedding, GCNPathName+"%d_%d" %
                    (embSize,testNum), "/fGCNembedding_%d.pkl" % idx)
        
print(subSum)        
GCN_tol_embedding()

#GCN classification
def GCN_classfication():
    gcn_res, rcnt = [0, 0, 0, 0], 5
    rs=7
    for i in range(rcnt):
        for idx in range(subSum-1, subSum):
            print(idx)
            print(tolFeaturePathName+"/tolFeature_%d.pkl" % idx)
            tolFeature = load_pickle(tolFeaturePathName, "/tolFeature_%d.pkl" % idx)
            tolFeature_df = pd.DataFrame(
                tolFeature, columns=['label', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8'])
            
            sp_muldG = load_pickle(muldigPathName, "/G_%d.pkl" % idx)
            #sp_mulG = load_pickle(mulgPathName+"/G_%d.pkl" % idx)
            print(mulgPathName+"/G_%d.pkl" % idx)
            y_cols_name = ['label']
            x_cols_name = [x for x in tolFeature_df.columns if x not in y_cols_name]
            global scipy_adj_matrix,train_x,train_y
            train_x = dcopy(tolFeature_df[x_cols_name])
            train_y = dcopy(tolFeature_df[y_cols_name])
            pos_cnt, neg_cnt = int(train_y.sum()), int(len(train_y) - train_y.sum())
            
            scipy_adj_matrix = get_scipy_adj_matrix(sp_muldG)
            print('pos node cnts:', pos_cnt)
            print('neg node cnts:', neg_cnt, 'pos/all ratio:',
                pos_cnt / (pos_cnt + neg_cnt))

            embSize = 8
            fGCNembedding = get_GCN_embedding(epoch=6, lr=0.0035, weight_decay=1e-6,
                                        esize=embSize, random_seed=rs+i)
            print("finish calculate embedding data!")
            save_pickle(fGCNembedding, GCNPathName+"%d" %
                        testNum, "/fGCNembedding_%d.pkl" % idx)
            trainX, trainY, testX, testY = get_input_data(
                GCNPathName+"%d" % testNum, "/fGCNembedding", True, False, True)
            trainX_2D, trainY_1D = trainX[:, subSum-1, :], trainY[:, subSum-1, 0]
        
        lgb_res = lgb_train_model_with_split(
            pd.DataFrame(trainX_2D), pd.DataFrame(trainY_1D), 2011)
        print_res(lgb_res)
        for j in range(len(gcn_res)):
            gcn_res[j] += lgb_res[j]
    gcn_res = [i / rcnt for i in gcn_res]
    gc.collect()
    return gcn_res


def get_trainx_trainy():
    idx=subSum-1
    tolFeature = load_pickle(tolFeaturePathName, "/tolFeature_%d.pkl" % idx)
    tolFeature_df = pd.DataFrame(
        tolFeature, columns=['label', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8'])
    sp_muldG = load_pickle(muldigPathName, "/G_%d.pkl" % idx)
    #print(mulgPathName+"/G_%d.pkl" % idx)
    y_cols_name = ['label']
    x_cols_name = [x for x in tolFeature_df.columns if x not in y_cols_name]
    global train_x,train_y
    train_x = dcopy(tolFeature_df[x_cols_name])
    train_y = dcopy(tolFeature_df[y_cols_name])
    pos_cnt, neg_cnt = int(train_y.sum()), int(len(train_y) - train_y.sum())


print("start!")
get_trainx_trainy()
gcn_res=GCN_classfication()
print_res(gcn_res)


# Deepwalk_classification
def Deepwalk_classification():
    global train_x, train_y
    dw_res, rcnt = [0, 0, 0, 0], 5
    for i in tqdm(range((rcnt))):
        node_feas, labels = dcopy(train_x.values), dcopy(train_y.values)
        embe_feas = read_embeds(DeepwalkPathName_new + '/embeds_dw_%d.dat' % i)
        np_fea_lab = np.hstack((node_feas, embe_feas, labels))
        columns_name = ['f%02d' % i for i in range(
            np_fea_lab.shape[1] - 1)] + ['label']
        df_fea_lab = pd.DataFrame(
            data=np_fea_lab, columns=columns_name, dtype=float)
        df_fea_lab['label'] = df_fea_lab['label'].astype(int)

        y_cols_name = ['label']
        x_cols_name = [x for x in df_fea_lab.columns if x not in y_cols_name]

        df_x = df_fea_lab[x_cols_name]
        df_y = df_fea_lab[y_cols_name]

        lgb_res = lgb_train_model_with_split(df_x, df_y, RANDOM_SEED+1)
        #print(lgb_res)
        for i in range(len(dw_res)):
            dw_res[i] += lgb_res[i]
    dw_res = [i / rcnt for i in dw_res]
    gc.collect()
    return dw_res

dw_res=Deepwalk_classification()
print_res(dw_res)


# LINE classification
def LINE_classification():
    line_res, rcnt = [0, 0, 0, 0], 5
    for i in tqdm(range(rcnt)):
        embe_feas = read_embeds(LINEPathName_new + '/embeds_line_%d.dat' % i)

        node_feas, labels = dcopy(train_x.values), dcopy(train_y.values)
        np_fealab = np.hstack((node_feas, embe_feas, labels))
        columns_name = ['f%02d' % i for i in range(
            np_fealab.shape[1] - 1)] + ['label']
        df_fealab = pd.DataFrame(
            data=np_fealab, columns=columns_name, dtype=float)
        df_fealab['label'] = df_fealab['label'].astype(int)

        y_cols_name = ['label']
        x_cols_name = [x for x in df_fealab.columns if x not in y_cols_name]

        linedf_x = df_fealab[x_cols_name]
        linedf_y = df_fealab[y_cols_name]

        lgb_res = lgb_train_model_with_split(linedf_x, linedf_y, RANDOM_SEED)
        for i in range(len(line_res)):
            line_res[i] += lgb_res[i]
    line_res = [i / rcnt for i in line_res]
    gc.collect()
    return line_res

line_res=LINE_classification()
print_res(line_res)


# node2vec Classification
def n2v_classification():
    global train_x, train_y
    n2v_res, rcnt = [0, 0, 0, 0], 5
    for i in tqdm(range(rcnt)):
        node_feas, labels = dcopy(train_x.values), dcopy(train_y.values)
        embe_feas = read_embeds(N2VPathName_new + '/embeds_n2v_%d.dat' % i)
        np_fealab = np.hstack((node_feas, embe_feas, labels))
        columns_name = ['f%02d' % i for i in range(
            np_fealab.shape[1] - 1)] + ['label']
        df_fealab = pd.DataFrame(
            data=np_fealab, columns=columns_name, dtype=float)
        df_fealab['label'] = df_fealab['label'].astype(int)

        y_cols_name = ['label']
        x_cols_name = [x for x in df_fealab.columns if x not in y_cols_name]

        n2vdf_x = df_fealab[x_cols_name]
        n2vdf_y = df_fealab[y_cols_name]
        print(n2vdf_x.shape,n2vdf_y.shape)
        lgb_res = lgb_train_model_with_split(n2vdf_x, n2vdf_y, RANDOM_SEED)
        for i in range(len(n2v_res)):
            n2v_res[i] += lgb_res[i]

    n2v_res = [i / rcnt for i in n2v_res]
    gc.collect()
    return n2v_res

n2v_res=n2v_classification()
print_res(n2v_res)


fSize=8
# pure feature
def get_pure_feature(splitFlag):
    tolX = np.zeros((SAMPLE_SIZE, subSum, fSize))
    tolY = np.zeros((SAMPLE_SIZE, subSum, 1))
    idx = subSum-1
    tolFeature_df = load_pickle(tolFeaturePathName, "/tolFeature_%d.pkl" % idx)
    # print(tolFeature_df.info())
    # print(tolFeature_df.head())
    # print(tolFeature_df.sum())
    y_cols_name = ['label']
    x_cols_name = [x for x in tolFeature_df.columns if x not in y_cols_name]

    subX = dcopy(tolFeature_df[x_cols_name]).values
    subY = dcopy(tolFeature_df[y_cols_name]).values
    tolX[:, idx, :] = subX
    tolY[:, idx, :] = subY
    if(splitFlag == True):
        trainX, testX, trainY, testY = train_test_split(
            tolX, tolY, test_size=0.2, random_state=RANDOM_SEED)
    else:
        trainX, trainY, testX, testY = tolX, tolY, 0, 0
    return trainX, trainY, testX, testY


def feature_classification():
    global train_x, train_y
    res, rcnt = [0, 0, 0, 0], 5
    for i in tqdm(range((rcnt))):
        trainX, trainY, testX, testY = get_pure_feature(False)
        trainX_2D, trainY_1D = trainX[:, subSum-1, :], trainY[:, subSum-1, :]
        print(trainX_2D.shape,trainY_1D.shape)
        f_res = lgb_train_model_with_split(pd.DataFrame(trainX_2D), pd.DataFrame(trainY_1D), RANDOM_SEED+i)
        print(f_res)
        for j in range(len(f_res)):
            res[j] += f_res[j]
    res = [j/ rcnt for j in res]
    gc.collect()
    return res

f_res=feature_classification()
print_res(f_res)