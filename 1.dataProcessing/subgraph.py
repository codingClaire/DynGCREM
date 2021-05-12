import pandas
import numpy
import os
import pickle
import networkx as nx
import time
import datetime
from tqdm import tqdm
import gc

SAMPLE_SIZE = 40000
DAYS, HOURS, MINUTES, SECOND = 60, 24, 60, 60
INTERVAL = DAYS*HOURS * MINUTES * SECOND


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pickle(pname, idx, data):
    if not os.path.exists(pname):  # 如果路径不存在
        os.makedirs(pname)
    fname = pname+"/G_%d.pkl" % idx
    print(fname)
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def timestamp2time(timeStamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeStamp))


def get_snapshotdata(spPath,subgPath,diFlag):
    #print(muldigpathName)
    G = load_pickle(spPath)  # 1140091
    print(nx.info(G))
    # 求timestamp的最值
    time_list = []
    for ind, edge in enumerate(nx.edges(G)):
        (u, v) = edge
        time_list.append(G[u][v][G.number_of_edges(u, v)-1]['timestamp'])
        time_list.append(G[u][v][0]['timestamp'])
    maxTimestamp = max(time_list)
    minTimestamp = min(time_list)
    maxT = timestamp2time(maxTimestamp)
    minT = timestamp2time(minTimestamp)
    G_list = []
    total = int((maxTimestamp-minTimestamp)//INTERVAL+1)
    print(total)
    for g in range(total):
        if(diFlag==True):
            G_tmp = nx.MultiDiGraph()
        else:
            G_tmp=nx.MultiGraph()
        G_list.append(G_tmp)

    flag={}
    for ind, edge in tqdm(enumerate(nx.edges(G))):
        (u, v) = edge
        if u not in flag:
            flag[u]={}
        if v not in flag[u]:
            flag[u][v]=1
            for m in range(0, G.number_of_edges(u, v)):
                t_idx = int((G[u][v][m]['timestamp']-minTimestamp)//INTERVAL)
                for j in range(t_idx, total):
                    nextE = G_list[j].number_of_edges(u, v)
                    G_list[j].add_edge(u, v)
                    G_list[j].nodes[u]['isp'] = G.nodes[u]['isp']
                    G_list[j].nodes[v]['isp'] = G.nodes[v]['isp']
                    G_list[j][u][v][nextE]['timestamp'] = G[u][v][m]['timestamp']
                    G_list[j][u][v][nextE]['amount'] = G[u][v][m]['amount']
        else:
            continue
    for g in range(total):
        print(g)
        print(nx.info(G_list[g]))
        save_pickle(subgPath, g, G_list[g])
        gc.collect()

if __name__ == '__main__':
    print("start!")
    mulgPath = './dataset/publicdata/graph_%d/SP_MulGs.pkl' % SAMPLE_SIZE
    muldigPath = './dataset/publicdata/graph_%d/SP_MulDiGs.pkl' % SAMPLE_SIZE
    mulgpathName = "./snapshotdata/graph_%d/interval_%d/mulG" % (SAMPLE_SIZE, INTERVAL)
    muldigpathName = "./snapshotdata/graph_%d/interval_%d/mulDiG" % (SAMPLE_SIZE, INTERVAL)
    get_snapshotdata(muldigPath, muldigpathName,True)
    get_snapshotdata(mulgPath, mulgpathName,False)