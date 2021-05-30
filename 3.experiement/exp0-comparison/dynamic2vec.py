from baseline.dynAE import dyngem_embedding

args={
      'base_path': './data/graph_50000',
      "origin_folder": "1.format",
      "embed_folder": "2.embedding/DynAE2",
      "model_folder": "CTGCN/model",
      "model_file": "dynae",
      "node_file": "nodes_set/nodes.csv",
      "file_sep": "\t",
      "start_idx": 4,
      "end_idx": -1,
      "duration": 4,
      "embed_dim": 8,
      "has_cuda": False,
      "thread_num": 30,
      "epoch": 5,
      "lr": 1e-3,
      "batch_size": 5120,
      "load_model": False,
      "shuffle": True,
      "export": True,
      "record_time": False,
      "n_units": [200, 100],
      "look_back": 3,
      "beta": 5,
      "nu1": 1e-6,
      "nu2": 1e-6,
      "bias": True
}
dyngem_embedding('DynAE', args)


args={
      "base_path": './data/graph_40000',
      "origin_folder": "1.format",
      "embed_folder": "2.embedding/DynAE2",
      "model_folder": "CTGCN/model",
      "model_file": "dynae",
      "node_file": "nodes_set/nodes.csv",
      "file_sep": "\t",
      "start_idx": 4,
      "end_idx": -1,
      "duration": 4,
      "embed_dim": 8,
      "has_cuda": False,
      "thread_num": 30,
      "epoch": 5,
      "lr": 1e-3,
      "batch_size": 5120,
      "load_model": False,
      "shuffle": True,
      "export": True,
      "record_time": False,
      "n_units": [200, 100],
      "look_back": 3,
      "beta": 5,
      "nu1": 1e-6,
      "nu2": 1e-6,
      "bias": true
    }
dyngem_embedding('DynAE', args)



args={
      "base_path": './data/graph_30000',
      "origin_folder": "1.format",
      "embed_folder": "2.embedding/DynAE2",
      "model_folder": "CTGCN/model",
      "model_file": "dynae",
      "node_file": "nodes_set/nodes.csv",
      "file_sep": "\t",
      "start_idx": 4,
      "end_idx": -1,
      "duration": 4,
      "embed_dim": 8,
      "has_cuda": False,
      "thread_num": 30,
      "epoch": 5,
      "lr": 1e-3,
      "batch_size": 5120,
      "load_model": False,
      "shuffle": True,
      "export": True,
      "record_time": False,
      "n_units": [200, 100],
      "look_back": 3,
      "beta": 5,
      "nu1": 1e-6,
      "nu2": 1e-6,
      "bias": True
    }
dyngem_embedding('DynAE', args)