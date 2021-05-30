# Part 1.Data Processing

The dataset for the project is from the Ethereum transaction network, which can be downloaded on [Kaggle](https://www.kaggle.com/xblock/ethereum-phishing-transaction-network).
I use a random walk algorithm to get three subgraphs on the original dataset with node number: 30000,40000,50000. We change the static graph into a graph snapshot sequence with 30 days as a time interval.
Since the nodes are lack of their features, eight features are defined and extracted from the structure of the dataset.
