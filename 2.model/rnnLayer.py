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
#import import_ipynb
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


def RNN_model(savePath,saveName):
    ###### model setting ######
    batchSize = 128
    epochsNum = 20

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
    model.add(LSTM(units=8, activation='relu', return_sequences=False,
                  input_shape=(timestamp, featuresNum)))
    model.add(RepeatVector(timestamp))  # none,timestamp,8
    model.add(LSTM(units=8, activation='tanh', return_sequences=True))
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