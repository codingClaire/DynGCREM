import pickle
import os

def load_pickle(pathName, fileName):
    fname = pathName+fileName
    # print(fname)
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pickle(narray, pathName, fileName):
    if not os.path.exists(pathName):
        os.makedirs(pathName)
    fname = pathName+fileName
    with open(fname, 'wb') as f:
        pickle.dump(narray, f)