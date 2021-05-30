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

SAMPLE_SIZE = 40000
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
modelPathName="./model/graph_%d/interval_%d" % (SAMPLE_SIZE, INTERVAL)
subSum = sum([len(files) for root, dirs, files in os.walk(mulgPathName)])
print(subSum)
testNum = 60
# 7 gcn layer experiment
# 10 gcn layer for 3layer
# 11 gcn layer for 2layer
# 20 for exp-30000 2layer
# 21 for exp-40000 1layer
# 40 double check for 30000
# 50 lstm Layer
# 60 lstm embSize experiment
FeatureSize = 16
fSize = 8
embSize=8

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
        #self.fc2 = nn.Linear(dim_in ,dim_out,bias=False)

    def forward(self, X, flag=1):
        # print("X:",X.shape[0],X.shape[1])
        #print("DAD:",self.DAD.shape[0], self.DAD.shape[1])
        X = torch.tanh(self.fc1(self.DAD.mm(X)))
        # X是返回的embedding结果
        #X = F.tanh(self.fc2(self.DAD.mm(X)))

        
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
    #print("start training")
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
            aggreFeaturde = pd.concat([embedding_df, deltaFeature_df], axis=1)
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

def GCN_tol_embedding(rs):
    for idx in tqdm(range(0, subSum)):
        #print(idx)
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
        #print('pos node cnts:', pos_cnt)
        #print('neg node cnts:', neg_cnt, 'pos/all ratio:',
        #    pos_cnt / (pos_cnt + neg_cnt))

        embSize = 8
        fGCNembedding = get_GCN_embedding(epoch=6, lr=0.005, weight_decay=1e-6,
                                    esize=embSize, random_seed=7+rs)
        #print("finish calculate embedding data!")
        save_pickle(fGCNembedding, GCNPathName+"%d_%d" %
                    (embSize,testNum), "/fGCNembedding_%d.pkl" % idx)
        
print(subSum)        

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from copy import deepcopy as dcopy
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.layers import LSTM, GRU
from keras.layers import RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from keras.losses import mse
from keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from keras.models import load_model
import import_ipynb
from LightGBMClassification import lgb_train_model_with_split,print_res
from saveFile import load_pickle, save_pickle



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

    # return dcopy(tolFeature_df[x_cols_name]), dcopy(tolFeature_df[y_cols_name])


def get_input_data(epathName, efileName, tolFlag, splitFlag):
    tolX = np.zeros((SAMPLE_SIZE, subSum, FeatureSize))
    tolY = np.zeros((SAMPLE_SIZE, subSum, 1))
    for idx in range(subSum-1,subSum):
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


def RNN_model_2layer(savePath,saveName,emsize):
    ###### model setting ######
    batchSize = 128
    epochsNum = 10

    lr_reduce = ReduceLROnPlateau(
        monitor='loss', factor=0.1, epsilon=0.00001, patience=1, verbose=1)
    checkpoint = ModelCheckpoint(
        "checkpointdata.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')

    tb = TensorBoard(log_dir='./logs',
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
    trainX, trainY, testX, testY = get_input_data(
        GCNPathName+"%d_%d" %(embSize,testNum), "/fGCNembedding", True, True) 
    #False表示不切分训练测试集
    #trainX= trainX[:, :, 8:] # 没注释表示只选取单纯特征进行训练
    #testX=testX[:,:,8:]
    print(trainX.shape, trainY.shape)
    featuresNum = trainX.shape[2]
    timestamp = trainX.shape[1]

    ###### model structure ######
    model = Sequential()
    model.add(LSTM(units=32, activation='tanh', return_sequences=True,
                   input_shape=(timestamp, featuresNum)))
    model.add(LSTM(units=emsize, activation='relu', return_sequences=False,
                  input_shape=(timestamp, featuresNum)))
    model.add(RepeatVector(timestamp))  # none,timestamp,8
    model.add(LSTM(units=emsize, activation='tanh', return_sequences=True))
    model.add(LSTM(units=32, activation='relu', return_sequences=True,
                   input_shape=(timestamp, featuresNum)))
    model.add(TimeDistributed(Dense(featuresNum)))

    print(model.summary())

    model.compile(loss='mse', optimizer=Adam(
        lr=0.0001), metrics=['mean_squared_error'], run_eagerly=True)
    history = model.fit(trainX, trainX,
                        epochs=epochsNum,
                        batch_size=batchSize,
                        callbacks=[checkpoint, lr_reduce],
                        # callbacks=[tb],
                        shuffle=True,
                        validation_data=(testX, testX),
                        verbose=1
                        )
    ###### model save ######
    h5file = savePath+saveName+".h5"
    jsonfile = savePath+saveName+".json"
    print(h5file, jsonfile)

    model_json = model.to_json()
    with open(jsonfile, "w") as json_file:
        json_file.write(model_json)
    model.save(h5file)
    print("Saved model to disk!")

    with open(savePath+saveName+'_trainHistoryDict.txt', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

        
def load_RNNmodel(fileName):
    new_model = load_model(fileName)
    return new_model


def get_autoEncoder_Embedding_Layer(trainX, testX, model):
    train_output1 = model.layers[0](trainX)
    train_output2 = model.layers[1](train_output1)
    train_output3 = model.layers[2](train_output2)

    test_output1 = model.layers[0](testX)
    test_output2 = model.layers[1](test_output1)
    test_output3 = model.layers[2](test_output2)
    
    return np.array(train_output3), np.array(test_output3)

def GCN_LSTM_classification(modelFileName,es):
    new_model = load_RNNmodel(modelFileName)
   
    trainX, trainY, testX, testY = get_input_data(
        GCNPathName+"%d_%d" %(embSize,testNum), "/fGCNembedding", True, False)
    #trainX= trainX[:, :, :8]
    ftrainX, ftrainY, ftestX, ftestY = get_pure_feature(False)
    print(trainX.shape,trainY.shape)
    trainX_emb, testX_emb = get_autoEncoder_Embedding_Layer(
        trainX, trainX, new_model)
    print("trainX_emb.shape:",trainX_emb.shape)
    for i in range(subSum-1,subSum):
        #print(str(i)+":------")
        inputX=np.concatenate((ftrainX[:,i,:], trainX_emb[:, i, :]), axis=1)
        #inputX=trainX_emb[:, i, :]
        print(inputX.shape)
        trainX_2D, trainY_1D =inputX, trainY[:,subSum-1, 0]
        result = lgb_train_model_with_split(
            pd.DataFrame(trainX_2D), pd.DataFrame(trainY_1D), es)
        print_res(result)
        return result

def LSTM_classification(modelFileName):
    # 不切分训练测试集
    new_model = load_RNNmodel(modelFileName)
    trainX, trainY, testX, testY = get_pure_feature(False)
    print(trainX.shape,trainY.shape)
    
    trainX_emb, testX_emb = get_autoEncoder_Embedding_Layer(
        trainX, trainX, new_model)  
    #实际上传入的test也是trainX 因为在这里不切分数据集
    trainXemb_2D = trainX_emb[:, subSum-1, :]
    trainY_1D = trainY[:, subSum-1, 0]
    inputX=np.concatenate((trainX[:,subSum-1,:], trainX_emb[:, subSum-1, :]), axis=1)
    result = lgb_train_model_with_split(
        pd.DataFrame(inputX), pd.DataFrame(trainY_1D), 2011)
    print_res(result)
    return result

def GCN_classfication():
    trainX, trainY, testX, testY = get_input_data(
        GCNPathName+"%d_%d" %(embSize,testNum), "/fGCNembedding",True, False)
    trainX_2D, trainY_1D = trainX[:, subSum-1, :], trainY[:, subSum-1, 0]
    ftrainX, ftrainY, ftestX, ftestY = get_pure_feature(False)
    #inputX=np.concatenate((ftrainX[:,subSum-1,:], trainX[:, subSum-1, :]), axis=1)
    inputX=trainX_2D
    result = lgb_train_model_with_split(
        pd.DataFrame(inputX), pd.DataFrame(trainY_1D), 2011)
    print_res(result)
    return result
    
def pure_feature_classification():
    trainX, trainY, testX, testY = get_pure_feature(False)
    trainX_2D, trainY_1D = trainX[:, subSum-1, :], trainY[:, subSum-1, 0]
    result = lgb_train_model_with_split(
        pd.DataFrame(trainX_2D), pd.DataFrame(trainY_1D), 2011)
    print_res(result)
    return result

top5=[2,4,9,11,13] # for 40000 experiement
if not os.path.exists(modelPathName):
    os.makedirs(modelPathName)
elist=[4,8,16,32,64]
m,m1=[],[]
for i in top5[1:]:
    for e in elist:
        #GCN_tol_embedding(i)
        modelName="/model_t60_%d_e%d" %(i,e)
        print(modelPathName+modelName)
        RNN_model_2layer(modelPathName,modelName,e)

        modelName="/model_t60_%d_e%d.h5" %(i,e)

        print("GCN_LSTM_classification for esize=",e)
        mres=GCN_LSTM_classification(modelPathName+modelName,)
        m.append(mres)
        print("GCN classification")
        m1res=GCN_classfication()
        m1.append(m1res)
        print(pd.DataFrame(m))
        print(pd.DataFrame(m1))
        print(pd.DataFrame(m).mean())
        print(pd.DataFrame(m1).mean())

print(pd.DataFrame(m))
print(pd.DataFrame(m1))
print(pd.DataFrame(m).mean())
print(pd.DataFrame(m1).mean())
