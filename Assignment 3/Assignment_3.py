import torch
import torch_scatter
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

import networkx as nx
import matplotlib.pyplot as plt

torch.manual_seed(666)

dataset = MoleculeNet(root='data/MoleculeNet', name='ESOL')  # download dataset

# let's look at some dataset statistics
print(f'Dataset name: {dataset}')
print('==============================================')
print(f'Number of graphs: {len(dataset)}')  # nr graphs in dataset
print(f'Number of node features: {dataset.num_node_features}')
print(f'Number of edge features: {dataset.num_edge_features}')
print('==============================================')
print()

# take a look at the first two graphs
graph_index = 0  # first graph
print(f'Graph {graph_index}:')
print('---------------------')
print(f'Number of nodes: {dataset[graph_index].num_nodes}')
print(f'Number of edges: {dataset[graph_index].num_edges}')
print()

# visualize the graph
G = to_networkx(dataset[graph_index], to_undirected=True)
nx.draw(G, with_labels=True)
plt.show()
print()

graph_index = 1  # second graph
print(f'Graph {graph_index}:')
print('---------------------')
print(f'Number of nodes: {dataset[graph_index].num_nodes}')
print(f'Number of edges: {dataset[graph_index].num_edges}')

# visualize the graph
G = to_networkx(dataset[graph_index], to_undirected=True)
nx.draw(G, with_labels=True)
plt.show()

# print a data object
graph_index = 0
print(f"Dataset index: {graph_index}\n")

print(f"Dataset item:\n{dataset[graph_index]}")

# Complete the code below for Question 2.1
graph_index = 0  # first graph

x = ...  # TODO: get node features
edge_index = ...    # TODO: get edge indices
edge_features = ...     # TODO: get edge features
target = ...    # TODO: get target property

# print out the attributes of the data object
print(f"Node features: {x}\n")
print(f"Edge indices: {edge_index}\n")
print(f"Edge features: {edge_features}\n")
print(f"Target property: {target}\n")

batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f'Number of batches: {len(loader)}')

first_batch = next(iter(loader))
print(first_batch)  # print the first batch

batch = next(iter(loader))
batch_attr = ...    # TODO: write your code to Question 3.1 here!

print(f'Minibatch .batch attribute:\n {batch_attr}\n')

class DummyModel(torch.nn.Module):
    ''' 
    A dummy model to illustrate how to use the `batch` attribute.
    '''
    def __init__(self, n_features=9):
        super(DummyModel, self).__init__()

        # here you can add cooler layers, e.g., graph convolutions or similar, but not this is not needed right now,
        # we keep it simple for this example

        self.proj = torch.nn.Linear(n_features, 1)  # project node features to a single value

    def forward(self, batch):
        node_feats = batch.x.float()
        x = ... # TODO: pass node features through the projection layer
        x_shape = ...
        print(f"Shape before aggregation: {x_shape}")

        out = ... # TODO: aggregate through summation
        out_shape = ...
        print(f"Shape after aggregation: {out_shape}")

        # NOTE: here we could add an activation function, but let's keep it simple for now
        return out

f = DummyModel()
batch = next(iter(loader))
out = f(batch)

