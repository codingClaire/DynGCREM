from baseline.dynAE import dyngem_embedding
import pandas as pd
import numpy as np
import os
import pickle
import networkx as nx

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_formatted_data():
    path='./data/graph_40000/0.ori/mulDiG'
    output_path='./data/graph_40000/1.format/'
    subSum = sum([len(files) for root, dirs, files in os.walk(path)])
    print(subSum)
    for idx in range(0,subSum):
        print(idx)
        df_graph=pd.DataFrame(columns = ['from_id', 'to_id', 'weight'])
        df_idx=0
        G = load_pickle(path+"/G_%d.pkl" %idx)
        for ind, edge in enumerate(nx.edges(G)):
            (u, v) = edge
            row=[]
            #print(type(u))
            row.append(int(u))
            row.append(int(v))
            row.append(G[u][v][0]['amount'])
            df_graph.loc[df_idx]=row
            #print(row[0])
            df_idx+=1
        if(idx<10):
            name="0"+str(idx)
        else:
            name=str(idx)
        df_graph.to_csv(output_path+name+".csv",sep='\t', index=False)
        
def get_node_data(node_num):
    node_dir='./data/graph_%d/nodes_set/' %node_num
    node_list=[i for i in range(1,node_num)]
    df_node = pd.DataFrame(node_list, columns=['node'])
    df_node.to_csv(os.path.join(node_dir, 'nodes.csv'), sep='\t', index=False)



args={
    'base_path':'/data/graph_40000',
    'origin_folder':'1.format',
    'embed_folder':'2.emb/dynGEM',
    'model_folder':'CTGCN/model',
    'model_file':'nodes_set/nodes.csv',
    'file_sep':"\t",
    'start_idx': 0,
    'end_idx': -1,
    'duration': 1,
    'embed_dim': 16,
    'has_cuda':False,
    'epoch':50,
    'lr':1e-3,
    'batch_size':256,
    'load_model':True,
    'shuffle':True,
    'export':True,
    'record_time':False,
    "n_units": [500, 300],
    "alpha": 1e-5,
    "beta": 10,
    "nu1": 1e-4,
    "nu2": 1e-4,
    "bias": True
}


get_node_data(40000)
get_formatted_data()
dyngem_embedding('DynGEM', args)