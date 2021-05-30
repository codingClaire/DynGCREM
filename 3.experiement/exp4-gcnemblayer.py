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
import import_ipynb
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
LINEPathName = embPathName+"/LINE"
N2VPathName= embPathName+"/N2V"
DeepwalkPathName= embPathName+"/Deepwalk"
DeepwalkPathName_new= embPathName+"/DeepWalkemb"
LINEPathName_new = embPathName+"/LINEemb"
N2VPathName_new= embPathName+"/N2Vemb"
subSum = sum([len(files) for root, dirs, files in os.walk(mulgPathName)])
print(subSum)
testNum = 11
# 11 for gcn embedding size exp
# 7 gcn layer experiment
#FeatureSize = 16
fSize = 8
print("warning testNum is:",testNum)

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
        self.fc1 = nn.Linear(dim_in, dim_in,  bias=False)
        self.fc2 = nn.Linear(dim_in,dim_out,bias=False)


    def forward(self, X, flag=1):
        # print("X:",X.shape[0],X.shape[1])
        #print("DAD:",self.DAD.shape[0], self.DAD.shape[1])
        X = torch.tanh(self.fc1(self.DAD.mm(X)))
        # X是返回的embedding结果
        X = F.tanh(self.fc2(self.DAD.mm(X)))
        return X


def gcn_train(X, Y, A_normed, A, epoch, lr, weight_decay, esize, random_seed):
    dim_n, dim_f = X.shape[0], X.shape[1]
    #print("dim_f:", X.shape[1])
    #print(A_normed.shape[0], A_normed.shape[1])
    print("esize is:",esize)
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


def get_GCN_embedding(epoch=10, lr=0.02, esize=4,weight_decay=2e-6, random_seed=RANDOM_SEED):
    global train_x, train_y
    X_new, Y_new, A_normed_new, adj_mat_new = get_input_vars()
    print("start training")
    X, Y, A_normed, adj_mat = dcopy(X_new), dcopy(
        Y_new), dcopy(A_normed_new), dcopy(adj_mat_new)

    embed_gcn = gcn_train(X, Y, A_normed, adj_mat, epoch=epoch, lr=lr,
                          weight_decay=weight_decay, esize=esize, random_seed=random_seed)
    embed_feas = embed_gcn.detach().numpy()
    return embed_feas

def get_input_data(epathName, efileName, esize,tolFlag, splitFlag, lastFlag):
    tolX = np.zeros((SAMPLE_SIZE, subSum, esize+8))
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

def GCN_classfication(emsize):
    gcn_res, rcnt = [0, 0, 0, 0], 5
    rs=7
    for i in range(rcnt):
        for idx in range(subSum-1, subSum):
            #print(idx)
            #print(tolFeaturePathName+"/tolFeature_%d.pkl" % idx)
            tolFeature = load_pickle(tolFeaturePathName, "/tolFeature_%d.pkl" % idx)
            tolFeature_df = pd.DataFrame(
                tolFeature, columns=['label', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8'])
            
            sp_muldG = load_pickle(muldigPathName, "/G_%d.pkl" % idx)
            #sp_mulG = load_pickle(mulgPathName+"/G_%d.pkl" % idx)
            #print(mulgPathName+"/G_%d.pkl" % idx)
            y_cols_name = ['label']
            x_cols_name = [x for x in tolFeature_df.columns if x not in y_cols_name]
            global scipy_adj_matrix,train_x,train_y
            train_x = dcopy(tolFeature_df[x_cols_name])
            train_y = dcopy(tolFeature_df[y_cols_name])
            pos_cnt, neg_cnt = int(train_y.sum()), int(len(train_y) - train_y.sum())
            
            scipy_adj_matrix = get_scipy_adj_matrix(sp_muldG)
            #print('pos node cnts:', pos_cnt)
            #print('neg node cnts:', neg_cnt, 'pos/all ratio:',
            #    pos_cnt / (pos_cnt + neg_cnt))

            fGCNembedding = get_GCN_embedding(epoch=6, lr=0.005, weight_decay=1e-6,
                                        esize=emsize, random_seed=rs+i)
            print("finish calculate embedding data!")
            save_pickle(fGCNembedding, GCNPathName+"%d" %testNum, "/fGCNembedding_%d.pkl" % idx)
            trainX, trainY, testX, testY = get_input_data(
                GCNPathName+"%d" %testNum, "/fGCNembedding",esize,True, False, True)
            #ftrainX, ftrainY, ftestX, ftestY = get_pure_feature(False)
            #print(ftrainX[:,subSum-1,:].size)
            #print(trainX[:, subSum-1, :])
            inputX=trainX[:, subSum-1, :]
            #inputX=np.concatenate((ftrainX[:,subSum-1,:], trainX[:, subSum-1, :]), axis=1)
            #inputX=trainX
            trainX_2D, trainY_1D =inputX, trainY[:,subSum-1, 0]
            lgb_res= lgb_train_model_with_split(
                pd.DataFrame(trainX_2D), pd.DataFrame(trainY_1D), 2011)
            #print_res(lgb_res)
        for j in range(len(gcn_res)):
            gcn_res[j] += lgb_res[j]
    gcn_res = [i / rcnt for i in gcn_res]
    print(gcn_res)
    gc.collect()
    return gcn_res

e_res = []
for esize in [4,8,16,32,64]:
    print(esize)
    tree_res = GCN_classfication(esize)
    e_res.append(tree_res)
    print(tree_res)
    
df_res = pd.DataFrame(e_res)
df_res.head()

print(df_res[0].tolist())
print(df_res[1].tolist())
print(df_res[2].tolist())
print(df_res[3].tolist())