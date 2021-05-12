import pandas as pd
import numpy
import os
import pickle
import networkx as nx
import time
import datetime
from tqdm import tqdm

SAMPLE_SIZE = 40000
DAYS, HOURS, MINUTES, SECOND = 45, 24, 60, 60
INTERVAL = DAYS*HOURS * MINUTES * SECOND
mulpathName = "./snapshotdata/graph_%d/interval_%d/mulG" % (
    SAMPLE_SIZE, INTERVAL)
muldigpathName = "./snapshotdata/graph_%d/interval_%d/mulDiG" % (
    SAMPLE_SIZE, INTERVAL)
statisticpathName = "./snapshotdata/graph_%d/interval_%d/statistic" % (
    SAMPLE_SIZE, INTERVAL)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_statistic_file(sPathName,gPathName):

    subMuldigSum = sum([len(files)
                        for root, dirs, files in os.walk(gPathName)])
    info_df = pd.DataFrame(columns=['index',
                           'nodesNum', 'edgeNum', 'posNum'])
    print(subMuldigSum)
    #print(info_df)
    for i in range(0,subMuldigSum):
        fName = gPathName+"/G_%d.pkl" % i
        G = load_pickle(fName)
        print(fName)
        nodesNum=len(list(G.nodes()))
        edgesNum = G.size() # 计算总边数 len(list(G.edges()))
        posNum=0
        for j, nd in enumerate(G.nodes()):
            posNum+=G.nodes[nd]['isp']
        #info_df[i]['index'] = i
        #info_df[i]['nodesNum'] = nodesNum
        #info_df[i]['edgeNum'] = edgesNum
        subInfo=[]
        subInfo.append(i)
        subInfo.append(nodesNum)
        subInfo.append(edgesNum)
        subInfo.append(posNum)
        #print(subInfo)
        info_df.loc[i]=subInfo
        #print(str(i)+":")
        #print(info_df.loc[i])
        #print(info_df)
    if not os.path.exists(sPathName):  # 如果路径不存在
        os.makedirs(sPathName)
    csvfileName = str(sPathName)+"/"+str(gPathName.split('/')[-1])+".csv"
    print(csvfileName)
    info_df.to_csv(csvfileName)

get_statistic_file(statisticpathName, muldigpathName)
get_statistic_file(statisticpathName, mulpathName)