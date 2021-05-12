import pandas as pd
import numpy as np
import os
import pickle
import networkx as nx
import time
import datetime
from tqdm import tqdm
from sklearn import preprocessing


SAMPLE_SIZE = 40000
DAYS, HOURS, MINUTES, SECOND = 60, 24, 60, 60
INTERVAL = DAYS*HOURS * MINUTES * SECOND
interPathName = "./snapshotdata/graph_%d/interval_%d" % (SAMPLE_SIZE, INTERVAL)
featurePathName = interPathName+'/features'
e_path = interPathName+'/edges'
wei_path = interPathName+'/weighted_edges'
deltaFeaturePathName = interPathName+'/delta_features'
tolFeaturePathName = interPathName+'/tol_features'
mulgpathName = interPathName+"/mulG"
muldigPathName = interPathName+"/mulDiG"
sp_mulgPath = './dataset/publicdata/graph_%d/SP_MulGs.pkl' % SAMPLE_SIZE
sp_muldigPath = './dataset/publicdata/graph_%d/SP_MulDiGs.pkl' % SAMPLE_SIZE

labels = [] # 所有sample_size的labe


subSum = sum([len(files) for root, dirs, files in os.walk(mulgpathName)])
featureSize = 9 # 特征+label

if not os.path.exists(featurePathName):
    os.makedirs(featurePathName)
if not os.path.exists(deltaFeaturePathName):
    os.makedirs(deltaFeaturePathName)
if not os.path.exists(tolFeaturePathName):
    os.makedirs(tolFeaturePathName)
if not os.path.exists(e_path):
    os.makedirs(e_path)
if not os.path.exists(wei_path):
    os.makedirs(wei_path)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


mul_dG = load_pickle(sp_muldigPath)
for i, nd in tqdm(enumerate(mul_dG.nodes())):
    labels.append(int(mul_dG.nodes[nd]['isp']))

def save_pickle(narray, pathName, fileName):
    if not os.path.exists(pathName):
        os.makedirs(pathName)
    fname = pathName+fileName
    with open(fname, 'wb') as f:
        pickle.dump(narray, f)


def get_edges_features(mul_G, mul_dG, idx):
    print(idx)
    print('Start writing edges ...')
    e_file = e_path+'/edges_%d.dat' % idx
    with open(e_file, 'w') as f:
        lines, weighted_edges = [], []
        for ind, edge in enumerate(nx.edges(mul_G)):
            (u, v) = edge
            eg = mul_G[u][v][0]
            amo, tim = eg['amount'], eg['timestamp']
            weighted_edges.append([u, v, amo, tim])
            lines.append(''.join([str(u), ' ', str(v)]))
        f.writelines('%s\n' % l for l in lines)

        df_wei = pd.DataFrame(weighted_edges, columns=[
                              'node1', 'node2', 'amount', 'timestamp'])
        norm_columns = ['amount', 'timestamp']
        df_wei[norm_columns] = preprocessing.minmax_scale(df_wei[norm_columns])
        df_wei['hybrid_fea'] = (df_wei['amount'] + df_wei['timestamp']) / 2
        df_wei = df_wei.drop(['amount', 'timestamp'], axis=1)
        wei_file = wei_path+'/weighted_edges_%d.dat' % idx
        df_wei.to_csv(wei_file, sep=' ', index=False, header=False)
        print('Edges write done.')

    print('Start getting features ...')
    df_data = []
    for i, nd in enumerate(mul_dG.nodes()):
        label = int(mul_dG.nodes[nd]['isp'])
        AF1 = mul_dG.in_degree[nd]
        AF2 = mul_dG.out_degree[nd]
        AF3 = AF1 + AF2
        AF4 = mul_dG.in_degree(nd, weight='amount')
        AF5 = mul_dG.out_degree(nd, weight='amount')
        AF6 = AF4 + AF5

        neighbors, timestamps = set(), []
        for ta, tb in mul_dG.in_edges(nd):
            neighbors.add(ta)
            timestamps.append(mul_dG[ta][tb][0]['timestamp'])
            timestamps.append(
                mul_dG[ta][tb][mul_dG.number_of_edges(ta, tb)-1]['timestamp'])
        for ta, tb in mul_dG.out_edges(nd):
            neighbors.add(tb)
            timestamps.append(mul_dG[ta][tb][0]['timestamp'])
            timestamps.append(
                mul_dG[ta][tb][mul_dG.number_of_edges(ta, tb)-1]['timestamp'])
        timestamps = list(map(float, timestamps))

        AF7 = len(neighbors)
        AF8 = (max(timestamps) - min(timestamps)) / AF3
        df_data.append([label, AF1, AF2, AF3, AF4, AF5, AF6, AF7, AF8])

    df = pd.DataFrame(df_data, columns=[
                      'label', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8'])
    df[['label']] = df[['label']].astype(int)
    norm_columns = ['AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8']
    df[norm_columns] = preprocessing.minmax_scale(df[norm_columns])
    f_file = featurePathName + '/features_%d.pkl' % idx
    df.to_pickle(f_file)  # , sep=' ', index=False, header=False)
    #save_pickle(df, idx,featurePathName)
    print('Features ready.')


"""[summary]
获取图的时间戳的最值
"""

def get_min_and_max_timestamp(G):
    time_list = []
    for ind, edge in enumerate(nx.edges(G)):
        (u, v) = edge
        # 因为两个点之间不只有一条边，需要统计所有的时间戳
        # 时间戳是按照大小排序的，因此只需要加入最大的和最小的
        # for i in range(0, G.number_of_edges(u, v)):
        #   time_list.append(G[u][v][i]['timestamp'])
        time_list.append(G[u][v][G.number_of_edges(u, v)-1]['timestamp'])
        time_list.append(G[u][v][0]['timestamp'])
    maxTimestamp = max(time_list)
    minTimestamp = min(time_list)
    #maxT = timestamp2time(maxTimestamp)
    #minT = timestamp2time(minTimestamp)
    return minTimestamp, maxTimestamp


def get_all_features():
    for idx in range(0, subSum):
        mulgFName = mulgpathName+"/G_%d.pkl" % idx
        muldigFName = muldigPathName+"/G_%d.pkl" % idx
        subMulG = load_pickle(mulgFName)
        subMuldG = load_pickle(muldigFName)
        get_edges_features(subMulG, subMuldG, idx)


"""[summary]
通过get_all_featues生成feature之后
生成从0时刻到当前时间段之间的tol_feature矩阵 每一个矩阵的尺寸全部相同
"""


def get_tol_features():
    for idx in range(0, subSum):
        # for idx in range(0, 1):
        fTolFeature = np.zeros((SAMPLE_SIZE, featureSize))
        sp_muldG_idx = load_pickle(muldigPathName+"/G_%d.pkl" % idx)
        df_idx = load_pickle(featurePathName+"/features_%d.pkl" % idx)
        node_list = list(sp_muldG_idx.nodes())
        #print(type(df_idx))
        #print(df_idx.shape[0], df_idx.shape[1])
        for i in range(len(node_list)):
            #print(df_idx.iloc[i, ].values)
            #print(type(df_idx.iloc[i, ].values))
            fTolFeature[node_list[i], :] = df_idx.iloc[i].values # 扩展到30000行的版本
        # print(fTolFeature)
        # print(np.info(fTolFeature))
        # for i in range(0,30000):
        #    print(fTolFeature[i])

        fTolFeature_df = pd.DataFrame(fTolFeature, columns=[
                                      'label', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8'])
        fTolFeature_df['label']=labels
        #print(fTolFeature_df.sum())
        save_pickle(fTolFeature_df, tolFeaturePathName,
                    "/tolFeature_%d.pkl" % idx)


"""[summary]
通过get_all_featues生成feature之后
生成时间段之间的delta_feature矩阵 每一个矩阵的尺寸全部相同
"""


def get_delta_features():
    idx=0
    alldf_idx_0 = load_pickle(
        tolFeaturePathName+"/tolFeature_%d.pkl" % idx)
    save_pickle(alldf_idx_0, deltaFeaturePathName,
                "/deltaFeature_%d.pkl" % idx)
    for idx in range(1, subSum):
        fdeltaFeature = np.zeros((SAMPLE_SIZE, featureSize))
        
        alldf_idx_1 = load_pickle(
            tolFeaturePathName+"/tolFeature_%d.pkl" % (idx-1))
        #print(alldf_idx_1.sum())
        alldf_idx_2 = load_pickle(
            tolFeaturePathName+"/tolFeature_%d.pkl" % idx)
        fdeltaFeature = alldf_idx_2.values-alldf_idx_1.values
        fdeltaFeature_df = pd.DataFrame(fdeltaFeature, columns=[
            'label', 'AF1', 'AF2', 'AF3', 'AF4', 'AF5', 'AF6', 'AF7', 'AF8'])
        #print(fdeltaFeature_df.sum())
        fdeltaFeature_df['label'] = labels
        save_pickle(fdeltaFeature_df, deltaFeaturePathName,
                    "/deltaFeature_%d.pkl" % idx)
 

if __name__ == '__main__':

    
    get_all_features()
    get_tol_features()
    get_delta_features()
