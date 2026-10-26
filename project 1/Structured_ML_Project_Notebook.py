# -*- coding: utf-8 -*-
"""
Converted from: Structured_ML_Project_Notebook.ipynb
Formatted with # %% cell markers for VS Code / PyCharm / Jupyter interactive execution.
"""

# %%
import sys
import os

# %% [markdown]
# <a href="https://colab.research.google.com/github/MahdiTheGreat/Structured-machine-learning/blob/main/project%201/Structured_ML_Project_Notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Usage of AI
#
# AI was mainly used to formalize my own arguments and also to generate latex formulas. I also used AI for research and to brush up on some previously learned topics e.g. axioms of Inner product.

# %% [markdown]
# # Project 1: Comparing Graph Neural Networks for Molecular Property Prediction
#
# ### Goal
# The goal of this project is to train **Graph Neural Network (GNN) models** that predict 
# molecular properties directly from the chemical structure of a molecule, represented as a graph.
#
# In chemistry, computing molecular properties from first principles (e.g., using the Boltzmann 
# distribution $p_M(\mathbf{x}) \propto \exp(-\beta u(\mathbf{x}))$) is extremely expensive 
# and often completely impractical. The alternative investigated here is to use large volumes 
# of experimental data together with the natural graph structure of molecules to learn 
# fast, direct predictors of those properties.
#
# ### The Two Datasets & Tasks
# Two molecular benchmark datasets from **MoleculeNet** are used, each representing a 
# different learning task:
#
# 1. **ESOL** *(Regression)*: Predicts the **solubility** of a molecule — how strongly it 
#    thermodynamically favors being dissolved in water over being in a vacuum. Evaluated with 
#    **MSE** and **Pearson's Correlation Coefficient**.
#
# 2. **BBBP** *(Binary Classification)*: Predicts whether a molecule can cross the 
#    **blood-brain barrier** — a critically important property for developing drugs that 
#    target the central nervous system. Evaluated with the **Area Under the ROC Curve (AUC)**.
#
# Each molecule is stored as a graph where nodes are atoms (with node features) and edges 
# represent chemical bonds. Data is split by **chemical scaffold** to avoid models accidentally 
# overfitting to structurally similar molecules.
#
# ### The Three Architectures Compared
# 1. **DeepSet (Baseline)** — A simple MLP that completely ignores the graph (edge) structure 
#    of the molecule. It treats the molecule as an unordered bag of atom features. This 
#    establishes the performance floor.
#
# 2. **GCN** (Graph Convolutional Network) — Aggregates information from neighboring atoms 
#    by computing a simple weighted average of neighbor features at each message-passing step.
#
# 3. **GAT** (Graph Attention Network / Graph Transformer) — An advanced architecture that 
#    learns to assign different **attention weights** to different neighbors, allowing the model 
#    to dynamically focus on the most chemically relevant atoms during message passing.
#
# ### How It Works
# - **Node features**: Integer atom features are converted into continuous vectors via 
#   `torch.nn.Embedding`. Binary features are kept as-is.
# - **Training**: Each model is trained **3 times with different random seeds** to produce 
#   reliable error estimates.
# - **Loss**: MSE for regression (ESOL) and Binary Cross-Entropy for classification (BBBP).
#
# ### The Central Question
# Does incorporating the **graph structure** of a molecule improve predictions over ignoring 
# it (DeepSet)? And between the two graph models, does the **attention mechanism** in GAT 
# give it an edge over the simpler aggregation in GCN?

# %%
# !pip install torch
# !pip install torch_geometric
# !pip install rdkit
# !pip install torcheval
# !pip install matplotlib
# !git clone https://github.com/olsson-group/sml_project1.git

# %%
from sml_project1 import metrics, data
from sml_project1.data import ScaffoldSplitter
import torch
import torch_geometric as geom
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import random
import numpy as np
import wandb
import time


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f'Using device: {device}')

# %%
# Filter out graphs with fewer than 2 nodes
def filter_small_graphs(dataset):
    """Remove molecules with fewer than 2 nodes (atoms) and return a new dataset."""
    initial_len = len(dataset)
    # filtered_data_list = [data_obj for data_obj in dataset if data_obj.x.size(0) >= 2]

    # To maintain compatibility with ScaffoldSplitter, we must return a true Dataset object
    # rather than just a list of Data objects. We do this by extracting the valid
    # indices and using PyTorch Geometric's built-in index_select() function to create a Subset.

    valid_indices = [i for i, data_obj in enumerate(dataset) if data_obj.x.size(0) >= 2]
    filtered_dataset = dataset.index_select(valid_indices)
    print(f"Filtered {initial_len - len(filtered_dataset)} molecules with < 2 nodes")
    return filtered_dataset


# Load and filter datasets
esol_raw = MoleculeNet(root='./storage/', name='ESOL')
esol = filter_small_graphs(esol_raw)

bbbp_raw = MoleculeNet(root='./storage/', name='BBBP')
bbbp = filter_small_graphs(bbbp_raw)


# Split datasets with scaffold stratification (Train: 80%, Val: 10%, Test: 10%)
# --- EXPLANATION OF SPLITTING METHOD ---
# We use a ScaffoldSplitter rather than a random split.
# In chemistry, a "scaffold" refers to the core structural framework of a molecule.
# Scaffold splitting groups molecules by their scaffold and completely isolates
# these groups into either Train, Validation, or Test sets.
# This ensures that the Test set contains entirely novel chemical families
# that the model has never seen before. It prevents data leakage and ensures
# that we evaluate the model's true ability to generalize to new drugs,
# rather than just memorizing structures it saw during training.
# ---------------------------------------
seed = 42
torch.manual_seed(seed)
splitter = ScaffoldSplitter()

esol_smiles = [d.smiles for d in esol]
esol_train_idx, esol_val_idx, esol_test_idx = splitter.train_valid_test_split(esol, esol_smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)
esol_train_dataset = torch.utils.data.Subset(esol, esol_train_idx)
esol_val_dataset = torch.utils.data.Subset(esol, esol_val_idx)
esol_test_dataset = torch.utils.data.Subset(esol, esol_test_idx)

bbbp_smiles = [d.smiles for d in bbbp]
bbbp_train_idx, bbbp_val_idx, bbbp_test_idx = splitter.train_valid_test_split(bbbp, bbbp_smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)
bbbp_train_dataset = torch.utils.data.Subset(bbbp, bbbp_train_idx)
bbbp_val_dataset = torch.utils.data.Subset(bbbp, bbbp_val_idx)
bbbp_test_dataset = torch.utils.data.Subset(bbbp, bbbp_test_idx)

print(f"ESOL: {len(esol)} molecules")
print(f"  Train: {len(esol_train_dataset)}, Val: {len(esol_val_dataset)}, Test: {len(esol_test_dataset)}")
print(f"BBBP: {len(bbbp)} molecules")
print(f"  Train: {len(bbbp_train_dataset)}, Val: {len(bbbp_val_dataset)}, Test: {len(bbbp_test_dataset)}")

def print_data_info(dataset, name, visualize=False):
    print("\n" + "="*50)
    print(f"Dataset: {name}")
    print("="*50)
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}")

    if len(dataset) > 0:
        graph = dataset[0]
        print(f"Example Graph 0: {graph}")
        print(f"Example Node Features (x): \n{graph.x[:5]} ... (truncated)")
        print(f"Example Target (y): {graph.y}")

        if visualize:
            from torch_geometric.utils import to_networkx
            import networkx as nx
            import matplotlib.pyplot as plt

            print(f"Visualizing Graph 0 from {name}...")
            G = to_networkx(graph, to_undirected=True)
            nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
            plt.show()
    print("="*50 + "\n")

print_data_info(esol, "ESOL", visualize=False)
print_data_info(bbbp, "BBBP", visualize=False)

# %%
from torch_geometric.nn import GraphConv, GATConv, global_mean_pool
import torch.nn as nn

class MolecularGraphNet(torch.nn.Module):
  # This is the parent class for both the regression and classification task

  def __init__(self, embedding, net, onehot_sizes, name="MolecularGraphNet"):
    super().__init__()
    self.embedding = embedding
    self.onehot_sizes = onehot_sizes  # List of num_classes for features 1-8
    self.net = net
    self.name = name
    self.setup_optimizer()

  def training_step(self, loss):
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

  def setup_optimizer(self):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

  def forward(self, batch):

    # Implement in children
    # We Setup the forward pass of the model for a classification task
    # this includes embedding the input data, applying the graph network,
    # pooling the results, and predicting an output.
    # where batch.x denotes the node features,
    # batch.batch denotes which graph each node belongs to

    """
    FEATURE PRE-PROCESSING:
    Neural networks cannot directly process raw categorical integers (like atomic numbers).
    Here, we convert the discrete features into continuous mathematical vectors:
    
    1. Feature 0 (Atomic Number): Mapped to a dense 32-dimensional vector via a learned embedding.
    2. Features 1-8 (Other Atom Properties): Converted into sparse one-hot encoded vectors.
    3. Concatenation: We glue the 32D embedding and all one-hot vectors together horizontally 
       to create the final, continuous feature vector for each atom.
    """

    #x = batch.x.to(torch.float32)  # prevent a type mismatch
    #print('batch is ', batch)
    x=batch.x
    atomic_embedded = self.embedding(x[:, 0])  # [num_nodes, 32]
    # Features 1-8: One-hot encode
    onehot_features = []
    for i, num_classes in enumerate(self.onehot_sizes, start=1):
     #print(f"Feature index: {i}, num_classes: {num_classes}")
     #print(f"Values for feature {i}: {x[:, i]}")
     # Add 1 to num_classes because values are 0-indexed and can go up to num_classes - 1
     # We cannot cast `batch.x` to float32 early because it contains discrete
     # categorical integers (like atomic numbers). Both `nn.Embedding` and
     # `nn.functional.one_hot` strictly require integer indices. We cast the
     # features to float32 *after* encoding them (see line 190).

     onehot = nn.functional.one_hot(x[:, i], num_classes=num_classes + 1).float()
     onehot_features.append(onehot)

    # Concatenate: embedded atomic number + all one-hot features
    x = torch.cat([atomic_embedded] + onehot_features, dim=1)

    #print("x shape after embedding is ",x.shape)

    # Apply GNN layers
    x = self.net(x, batch.edge_index, batch.batch)
    # Pool graph-level representation
    #print("net out shape: ",x.shape)
    x = geom.nn.global_mean_pool(x, batch.batch)
    return x.squeeze()

  def loss(self, y, y_hat):
    # Implement in children
    raise NotImplementedError

  def evaluate(self, y, y_hat):
    # Implement in children
    raise NotImplementedError


from torch_geometric.nn import GraphConv, GATConv, global_mean_pool
import torch.nn as nn

class MolecularGraphClassification(MolecularGraphNet):
  def __init__(self, embedding, net,onehot_sizes, name="MolecularGraphClassification"):
    super().__init__(embedding, net,onehot_sizes, name=name)
    self.sigmoid = nn.Sigmoid()
    self.bce_loss = nn.BCELoss()

  def forward(self, batch):
   return self.sigmoid(super().forward(batch))

  def loss(self, y, y_hat):
    return self.bce_loss(y_hat, y.squeeze().float())

  def evaluate(self, y, y_hat):
    roc_auc = metrics.get_auc_roc(y, y_hat)
    return {"roc_auc": roc_auc}

class MolecularGraphRegression(MolecularGraphNet):
  def __init__(self, embedding, net, onehot_sizes, name="MolecularGraphRegression", mean=0.0, std=1.0):
    super().__init__(embedding, net, onehot_sizes, name=name)
    # Target standardization parameters (mean and standard deviation)
    self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
    self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

  def forward(self, batch):
    return super().forward(batch)

  def loss(self, y, y_hat):
    # Standardize ground truth y to normalized space for MSE loss
    mean = self.mean.to(y.device)
    std = self.std.to(y.device)
    y_norm = (y.squeeze().float() - mean) / std if std > 0 else y.squeeze().float()
    return torch.nn.functional.mse_loss(y_norm, y_hat)

  def evaluate(self, y, y_hat):
    # Revert normalized model predictions back to original physical scale before computing PCC and RMSE
    mean = self.mean.to(y_hat.device)
    std = self.std.to(y_hat.device)
    y_hat_orig = y_hat * std + mean if std > 0 else y_hat
    pcc = metrics.get_pearson_corr(y, y_hat_orig)[0]
    rmse = metrics.get_rmse(y, y_hat_orig)
    return {"pcc": pcc, "rmse": rmse}

  def predict(self, batch):
    """Predicts values and returns them in original physical units."""
    self.eval()
    with torch.no_grad():
      y_hat = self.forward(batch)
      mean = self.mean.to(y_hat.device)
      std = self.std.to(y_hat.device)
      return y_hat * std + mean if std > 0 else y_hat

class DeepSet(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(in_features, hidden_features))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_features, out_features))


    def forward(self, x, edge_index, batch):
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
              out = layer(out)
            else: # Assuming ReLU
              out = layer(out)
        return out

from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

class GCN(torch.nn.Module):
# We use 1 Message Passing Layer (the GCNConv layer) followed by a ReLU activation, and then a final Linear layer to make the prediction.
# While this might sound "shallow", shallow networks are actually a massive advantage for GCNs.
# GCNs suffer heavily from a well-documented mathematical flaw called "Oversmoothing".
# If you use too many GCN layers (e.g., 3 or 4+), the features of every atom in the molecule become averaged out and mathematically identical,
# destroying the model's predictive power. shallow prevents oversmoothing while still leveraging the layer-wise propagation rule from Kipf & Welling.
    def __init__(self,in_features, hidden_features, out_features=1, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GCNConv(in_features, hidden_features))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_features, hidden_features))

        # Output layer
        self.layers.append(nn.Linear(hidden_features, out_features))

    def forward(self, x, edge_index, batch):
        # Apply GCN layers with ReLU
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = torch.relu(x)

        # Apply final linear layer
        x = self.layers[-1](x)
        return x


class GAT(torch.nn.Module):
# We use 3 Message Passing Layers to fairly compare against GCN and DeepSet.
# To maintain a matching parameter count, we use single-head attention (heads=1).
# The layers use ELU activations (as proposed in the foundational GAT paper), except the final GAT layer which uses ReLU.


    def __init__(self, in_features, hidden_features, out_features=1, num_layers=3, heads=1, concat=True):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads # Number of attention heads for the first layer
        self.layers = nn.ModuleList()

        # First GAT layer (with multiple heads)

        # Note: concat=False is used as a safety measure. Since heads=1, concatenating or averaging
        # is mathematically identical. If heads > 1 is ever used, False prevents the output dimension
        # from exploding and crashing the downstream Linear layer.

        self.layers.append(GATConv(in_features, hidden_features, heads=self.heads, concat=concat))
        self.layers.append(nn.ELU()) # Activation after first GAT layer

        # Hidden GAT layers (collapsing heads if needed, or keeping multi-head)
        # For simplicity here, let's add more multi-head GAT layers if num_layers > 2
        # A more complex implementation might alternate between multi-head and single-head
        for _ in range(num_layers - 2):
             self.layers.append(GATConv(hidden_features, hidden_features, heads=self.heads, concat=concat))
             self.layers.append(nn.ELU())


        # Final layer (either GAT or Linear). Let's use a final GAT layer that collapses heads
        # self.layers.append(GATConv(hidden_features * self.heads, hidden_features, heads=1, concat=False))
        # self.layers.append(nn.ReLU()) # Activation after final GAT layer


        # Final linear layer for prediction
        self.layers.append(nn.Linear(hidden_features, out_features))


    def forward(self, x, edge_index, batch):
        out = x
        # Apply GAT and activation layers
        for layer in self.layers[:-1]: # Apply all layers except the final linear layer
             if isinstance(layer, GATConv):
                out = layer(out, edge_index)
             else: # Activation layer
                out = layer(out)

        # Apply final linear layer
        out = self.layers[-1](out)
        return out

def plot_results(results, model_name):
    epochs = range(1, len(results['train_loss']) + 1)
    metrics_to_plot = list(results['train_metrics'][0].keys())
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics + 1, figsize=(6 * (num_metrics + 1), 5))

    # Plot Loss
    axes[0].plot(epochs, results['train_loss'], label='Train Loss')
    axes[0].plot(epochs, results['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Metrics
    for i, metric_name in enumerate(metrics_to_plot):
        train_metric_values = [m[metric_name] for m in results['train_metrics']]
        val_metric_values = [m[metric_name] for m in results['val_metrics']]
        axes[i+1].plot(epochs, train_metric_values, label=f'Train {metric_name.upper()}')
        axes[i+1].plot(epochs, val_metric_values, label=f'Validation {metric_name.upper()}')
        axes[i+1].set_title(f'{model_name} - {metric_name.upper()} over Epochs')
        axes[i+1].set_xlabel('Epoch')
        axes[i+1].set_ylabel(metric_name.upper())
        axes[i+1].legend()
        axes[i+1].grid(True)


    plt.tight_layout()
    plt.show()

def train(model, train_dataset, batch_size, val_dataset, test_dataset, n_epochs, val_factor=2, use_wandb=False, patience=15):
    import copy
    print(f"Training {model.name} with {batch_size} batch size for {n_epochs} epochs (Early Stopping Patience: {patience})...")

    if use_wandb:
     # Generate a dynamic project name
     name = f"{model.name}_epochs{n_epochs}_batch{batch_size}_layers{model.net.num_layers}"
     # Initialize a wandb run
     wandb.init(project=model.name, name=name)

    train_dataloader = geom.loader.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataloader = geom.loader.DataLoader(
        val_dataset, batch_size=batch_size*val_factor, drop_last=False # Fixed ZeroDivisionError
    )

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # === TRAINING ===
        model.train()
        epoch_train_loss = 0
        all_train_y = []
        all_train_y_hat = []

        for i, batch in enumerate(train_dataloader):
            batch = batch.to(device)

            # Forward pass
            y_hat = model(batch)

            if epoch == 0 and i == 0:
                print(f"\nArchitecture of {model.name} after first forward pass:")
                print(model)
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total Trainable Parameters: {total_params:,}\n")

            # Calculate loss
            loss = model.loss(batch.y, y_hat)

            # Backward pass
            model.training_step(loss)

            epoch_train_loss += loss.item()

            # Safely move predictions to CPU RAM to prevent GPU Out-of-Memory leaks
            # (sklearn metrics require CPU anyway). .view(-1) flattens the tensor
            # to prevent torch.cat crashes if a batch only has 1 molecule left.
            all_train_y.append(batch.y.cpu().view(-1))
            all_train_y_hat.append(y_hat.detach().cpu().view(-1))

        # Calculate epoch train metrics properly
        all_train_y = torch.cat(all_train_y, dim=0)
        all_train_y_hat = torch.cat(all_train_y_hat, dim=0)
        try:
            avg_train_metrics = model.evaluate(all_train_y, all_train_y_hat)
        except Exception:
            avg_train_metrics = {"error": 0.0}

        # === VALIDATION ===
        model.eval()
        epoch_val_loss = 0
        all_val_y = []
        all_val_y_hat = []

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = val_batch.to(device)

                # Forward pass
                y_hat = model(val_batch)

                # Calculate loss
                loss = model.loss(val_batch.y, y_hat)
                epoch_val_loss += loss.item()

                # Offload to CPU to prevent memory leaks and flatten to 1D
                all_val_y.append(val_batch.y.cpu().view(-1))
                all_val_y_hat.append(y_hat.cpu().view(-1))

        # Calculate epoch val metrics properly
        all_val_y = torch.cat(all_val_y, dim=0)
        all_val_y_hat = torch.cat(all_val_y_hat, dim=0)
        try:
            avg_val_metrics = model.evaluate(all_val_y, all_val_y_hat)
        except Exception:
            avg_val_metrics = {"error": 0.0}

        # Average metrics
        avg_train_loss = epoch_train_loss / max(1, len(train_dataloader))
        avg_val_loss = epoch_val_loss / max(1, len(val_dataloader))


        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_metrics.append(avg_train_metrics)
        val_metrics.append(avg_val_metrics)
        if (epoch + 1) % 5 == 0:
         print(f"Epoch {epoch+1}/{n_epochs}")
         print(f"  Train - Loss: {avg_train_loss:.4f}, {', '.join([f'{k}: {v:.4f}' for k, v in avg_train_metrics.items()])}")
         print(f"  Val   - Loss: {avg_val_loss:.4f}, {', '.join([f'{k}: {v:.4f}' for k, v in avg_val_metrics.items()])}")

        if use_wandb:
         # Log metrics to wandb
         wandb.log({
             "epoch": epoch + 1,
             "train_loss": avg_train_loss,
             "val_loss": avg_val_loss,
             **{f"train_{k}": v for k, v in avg_train_metrics.items()},
             **{f"val_{k}": v for k, v in avg_val_metrics.items()}
         })

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break

    # Restore the best model weights before testing
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # === TEST EVALUATION ===
    test_dataloader = geom.loader.DataLoader(test_dataset, batch_size=batch_size*val_factor, drop_last=False)
    model.eval()
    all_test_y = []
    all_test_y_hat = []

    with torch.no_grad():
        for test_batch in test_dataloader:
            test_batch = test_batch.to(device)
            y_hat = model(test_batch)
            # Offload to CPU to prevent memory leaks and flatten to 1D
            all_test_y.append(test_batch.y.cpu().view(-1))
            all_test_y_hat.append(y_hat.cpu().view(-1))

    all_test_y = torch.cat(all_test_y, dim=0)
    all_test_y_hat = torch.cat(all_test_y_hat, dim=0)
    final_test_metrics = model.evaluate(all_test_y, all_test_y_hat)

    print(f"  Final Test Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in final_test_metrics.items()])}")


    hist= {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": final_test_metrics
    }

    plot_results(hist, model.name)

    # Finish the wandb run
    if use_wandb:
        wandb.finish()

    return hist

def get_max_values(dataset):
    # Dynamically get the number of columns (features) from the first molecule
    num_features = dataset[0].x.shape[1]

    max_vals = []
    # Loop over the actual dynamic length
    for i in range(num_features):
        max_val = int(max([data.x[:, i].max().item() for data in dataset]))
        max_vals.append(max_val)
    return max_vals


from collections import defaultdict

def reset_weights(m):
    """Recursively resets the parameters of a PyTorch module to new random values."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def run_experiment(model_class, model_kwargs, train_ds, val_ds, test_ds, batch_size, n_epochs, repeat_factor=3):
    print(f"\n{'='*50}")
    print(f"Running Experiment: {model_kwargs['name']} ({repeat_factor} replicates)")
    print(f"{'='*50}")

    all_test_metrics = defaultdict(list)
    start_time = time.time()

    # We instantiate the model architecture here...
    model = model_class(**model_kwargs).to(device)

    for i in range(repeat_factor):
        print(f"\n--- Replicate {i+1}/{repeat_factor} ---")
        # ...but we MUST force PyTorch to assign completely new random weights for each replicate!
        # Otherwise, the loop just continues training the exact same model!
        model.apply(reset_weights)

        hist = train(model, train_ds, batch_size, val_ds, test_ds, n_epochs)

        for k, v in hist['test_metrics'].items():
            all_test_metrics[k].append(v)

    end_time = time.time()
    print(f"\n>>> FINAL AGGREGATED RESULTS FOR {model_kwargs['name']} <<<")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    final_metrics_summary = {}
    for k, v_list in all_test_metrics.items():
        mean_val = np.mean(v_list)
        std_val = np.std(v_list)
        print(f"  Test {k.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        final_metrics_summary[k] = (mean_val, std_val)

    return final_metrics_summary


# %%
esol_max_vals = get_max_values(esol)


repeat_factor=3
reg_batch_size = 32
reg_n_epochs = 1000
reg_onehot_sizes=esol_max_vals[1:]

import sys
DEBUG_MODE = '--test' in sys.argv
if DEBUG_MODE:
    print("\n[WARNING] Running in FAST DEBUG MODE. Disabling large epochs and replicates.")
    reg_n_epochs = 1
    repeat_factor = 1
    # Shrink datasets
    esol_train_dataset = torch.utils.data.Subset(esol_train_dataset, range(min(10, len(esol_train_dataset))))
    esol_val_dataset = torch.utils.data.Subset(esol_val_dataset, range(min(5, len(esol_val_dataset))))
    esol_test_dataset = torch.utils.data.Subset(esol_test_dataset, range(min(5, len(esol_test_dataset))))

# Setup
hidden_features = 64
embedding_dim=32

# Total dimension: 32 (atomic) + sum of one-hot dimensions
# Add 1 to each size in esol_max_vals[1:] for one-hot encoding
# There are 9 features in the ESOL dataset
in_features = embedding_dim + sum([size + 1 for size in esol_max_vals[1:]])
print(f"in features: {in_features}")

# Target standardization parameters (computed strictly on the training set to prevent data leakage)
esol_train_targets = torch.cat([d.y.view(-1) for d in esol_train_dataset]).float()
esol_mean = esol_train_targets.mean().item()
esol_std = esol_train_targets.std().item()
print(f"Target Standardization (ESOL) -> Train Mean: {esol_mean:.4f}, Train Std: {esol_std:.4f}")

# --- REGRESSION EXPERIMENTS (ESOL) ---

reg_results = {}
reg_results["DeepSet"] = run_experiment(MolecularGraphRegression,
               {"embedding": torch.nn.Embedding(num_embeddings=esol_max_vals[0]+1, embedding_dim=embedding_dim),
                "net": DeepSet(in_features, hidden_features),
                "onehot_sizes": reg_onehot_sizes,
                "mean": esol_mean,
                "std": esol_std,
                "name": "DeepSet Regression"},
               esol_train_dataset, esol_val_dataset, esol_test_dataset, reg_batch_size, reg_n_epochs)

# %%
reg_results["GCN"] = run_experiment(MolecularGraphRegression,
               {"embedding": torch.nn.Embedding(num_embeddings=esol_max_vals[0]+1, embedding_dim=embedding_dim),
                "net": GCN(in_features, hidden_features),
                "onehot_sizes": reg_onehot_sizes,
                "mean": esol_mean,
                "std": esol_std,
                "name": "GCN Regression"},
               esol_train_dataset, esol_val_dataset, esol_test_dataset, reg_batch_size, reg_n_epochs)

# %%
reg_results["GAT"] = run_experiment(MolecularGraphRegression,
               {"embedding": torch.nn.Embedding(num_embeddings=esol_max_vals[0]+1, embedding_dim=embedding_dim),
                "net": GAT(in_features, hidden_features),
                "onehot_sizes": reg_onehot_sizes,
                "mean": esol_mean,
                "std": esol_std,
                "name": "GAT Regression"},
               esol_train_dataset, esol_val_dataset, esol_test_dataset, reg_batch_size, reg_n_epochs)

# %%
# Setup for BBBP (Classification)
# There are 9 features in the BBBP dataset
bbbp_max_vals = get_max_values(bbbp)
class_batch_size = 32
class_n_epochs = 200
class_onehot_sizes = bbbp_max_vals[1:]
in_features_classification = embedding_dim + sum([size + 1 for size in bbbp_max_vals[1:]])

if DEBUG_MODE:
    class_n_epochs = 1
    bbbp_train_dataset = torch.utils.data.Subset(bbbp_train_dataset, range(min(10, len(bbbp_train_dataset))))
    bbbp_val_dataset = torch.utils.data.Subset(bbbp_val_dataset, range(min(5, len(bbbp_val_dataset))))
    bbbp_test_dataset = torch.utils.data.Subset(bbbp_test_dataset, range(min(5, len(bbbp_test_dataset))))

# --- CLASSIFICATION EXPERIMENTS (BBBP) ---
cls_results = {}
cls_results["DeepSet"] = run_experiment(MolecularGraphClassification,
               {"embedding": torch.nn.Embedding(num_embeddings=bbbp_max_vals[0]+1, embedding_dim=embedding_dim),
                "net": DeepSet(in_features_classification, hidden_features),
                "onehot_sizes": class_onehot_sizes,
                "name": "DeepSet Classification"},
               bbbp_train_dataset, bbbp_val_dataset, bbbp_test_dataset, class_batch_size, class_n_epochs)

# %%
cls_results["GCN"] = run_experiment(MolecularGraphClassification,
               {"embedding": torch.nn.Embedding(num_embeddings=bbbp_max_vals[0]+1, embedding_dim=embedding_dim),
                "net": GCN(in_features_classification, hidden_features),
                "onehot_sizes": class_onehot_sizes,
                "name": "GCN Classification"},
               bbbp_train_dataset, bbbp_val_dataset, bbbp_test_dataset, class_batch_size, class_n_epochs)

# %%
cls_results["GAT"] = run_experiment(MolecularGraphClassification,
               {"embedding": torch.nn.Embedding(num_embeddings=bbbp_max_vals[0]+1, embedding_dim=embedding_dim),

                "net": GAT(in_features_classification, hidden_features, heads=4),
                "onehot_sizes": class_onehot_sizes,
                "name": "GAT Classification"},
               bbbp_train_dataset, bbbp_val_dataset, bbbp_test_dataset, class_batch_size, class_n_epochs)

# %%

def plot_final_comparison(results_dict, task_name, metrics):
    print(f"\n{'='*50}\nFINAL RESULTS FOR {task_name.upper()}\n{'='*50}")
    for model_name, metrics_data in results_dict.items():
        print(f"\nModel: {model_name}")
        for metric in metrics:
            mean, std = metrics_data[metric]
            print(f"  Test {metric.upper()}: {mean:.4f} ± {std:.4f}")

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    models = list(results_dict.keys())

    for i, metric in enumerate(metrics):
        means = [results_dict[m][metric][0] for m in models]
        stds = [results_dict[m][metric][1] for m in models]

        bars = axes[i].bar(models, means, yerr=stds, capsize=5, alpha=0.7)
        axes[i].set_title(f'{task_name} - {metric.upper()}')
        axes[i].set_ylabel(metric.upper())
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        # Add numeric labels on top of the bars for easy reading
        for bar, mean, std in zip(bars, means, stds):
            yval = bar.get_height()
            # Position the text slightly above the error bar
            axes[i].text(bar.get_x() + bar.get_width()/2, yval + std + (0.02 * abs(yval)),
                         f'{mean:.3f}±{std:.3f}',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{task_name.lower()}_comparison.png")
    plt.show()

print("\nGenerating comparison plots...")
plot_final_comparison(reg_results, "Regression", ["rmse", "pcc"])
plot_final_comparison(cls_results, "Classification", ["roc_auc"])

# %% [markdown]
# Here's a brief outline of what has been done:
#
# 1. **Installed necessary libraries:** Installed `torch`, `torch_geometric`, `rdkit`, `torcheval`, and `matplotlib`, and cloned a GitHub repository containing relevant code.
# 2. **Set up the device:** Determined whether to use CUDA, MPS, or CPU for training.
# 3. **Loaded and filtered datasets:** Loaded the ESOL and BBBP datasets using `torch_geometric.datasets.MoleculeNet`. A function `filter_small_graphs` was implemented and used to remove molecules with fewer than 2 nodes from both datasets.
# 4. **Split datasets:** Split the filtered ESOL and BBBP datasets into training and validation sets using a scaffold split.
# 5. **Defined Model Architectures:** To keep the code highly modular, the models were built using object-oriented inheritance:
#    * **The Parent Class (`MolecularGraphNet`):** Handles all shared boilerplate (feature embedding, one-hot encoding, passing data through the GNN, and graph pooling).
#    * **The Task Children (`MolecularGraphClassification` & `MolecularGraphRegression`):** Inherit the heavy lifting from the parent and only define the final task-specific logic (e.g., loss functions like BCELoss vs MSELoss, and evaluation metrics like ROC-AUC vs Pearson). For regression (ESOL), target standardization (z-score normalization) is integrated: targets are standardized using training set statistics during training/loss calculation to stabilize gradients, and predictions are reverted back to original units during evaluation to report authentic RMSE and PCC. 
#
#    Because the Task classes are separated from the Graph architectures, we can mix-and-match any architecture into any task like Lego bricks. We defined three distinct architectures (the `net` passed to the models) to evaluate how node connections impact performance:
#    * **DeepSet (Baseline):** Explicitly ignores the graph structure (bypassing the adjacency matrix / `edge_index`), treating the molecule as an unconnected set of atoms. This determines if edges provide meaningful signal beyond raw node features alone.
#    * **GCN:** A Message Passing Neural Network (MPNN) based on Kipf & Welling [2] that treats all neighbors equally, using the static layer-wise propagation rule:
#
#      $$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$
#
#      where $\tilde{A} = A + I_N$ is the adjacency matrix with self-connections and $\tilde{D}$ is the degree matrix used to normalize incoming messages so highly-connected nodes don't explode in value.
#    * **GAT:** A second MPNN that replaces GCN's fixed, degree-based normalization with a learned attention mechanism, allowing nodes to dynamically weigh the importance of incoming messages from different neighbors.
#
#    *(Note: We also adjusted batch sizes across these models to improve stability and prevent undefined metrics like Pearson correlation during training).*
# 6. **Defined Training and Plotting Functions:** Implemented functions for plotting training results (`plot_results`) and training the models (`train`), including logging with Weights & Biases. A helper function `get_max_values` was created to determine feature dimensions for embedding.
# 7. **Instantiated and Trained Models:** Initialized and trained instances of `MolecularGraphRegression` (using DeepSet, GCN, and GAT) on the ESOL dataset and `MolecularGraphClassification` (again using DeepSet, GCN and GAT) on the BBBP dataset.

# %% [markdown]
# ## Model Comparison
#
# Below is the comparison of the different models on the ESOL (Regression) and BBBP (Classification) tasks:
#
# 1. **DeepSet (Baseline)**: This model applies a per-node MLP and then pools the results globally. It does not use the graph structure (edges) at all. It serves as a baseline to see how much can be learned purely from atomic properties without structural context.
# 2. **GCN (Graph Convolutional Network)**: GCN aggregates features from immediate neighbors using an isotropic weighting scheme based on node degrees. It can capture local structural motifs but struggles to distinguish between neighbors of different importance.
# 3. **GAT (Graph Attention Network)**: GAT uses an attention mechanism to learn dynamic weights for different neighbors. To maintain a fair and equal parameter count against the GCN baseline, we constrained the model to a single attention head. It theoretically can attend to specific chemical bonds that are most relevant to the target property.
#
# **Discussion**:
# Initially, it was hypothesized that GCN and GAT would easily outperform the DeepSet baseline, as molecular properties generally depend heavily on the structural arrangement of atoms (e.g., rings, functional groups). However, our empirical results revealed a more nuanced reality depending on the specific physical domain of the dataset.
#
# For the **ESOL regression task**, the DeepSet baseline surprisingly outperformed both graph networks. One can assume that because solubility is largely an additive property driven by the atoms within a molecule, identifying atomic features without topological context is highly effective. The graph models likely overcomplicated the problem by attempting to map complex pathways, resulting in slight overfitting and degraded performance.
#
# Conversely, for the **BBBP classification task**, GCN significantly outperformed DeepSet. One can assume that passing through the blood-brain barrier is a highly selective biological process that demands a specific structural shape and geometry. Merely knowing the atomic composition is not enough and the model must understand the molecular graph connectivity, which allowed the GCN to excel.
#
# Finally, when comparing the two graph architectures, GCN outperformed GAT. While GAT's attention mechanism is theoretically more expressive, one can assume that attention networks are highly data-hungry. Because the subsets of the MoleculeNet datasets used here are relatively tiny, the GAT struggled to learn meaningful, dynamic attention weights. In contrast, the GCN's fixed, mathematically normalized averaging rule proved to be a much more robust and reliable inductive bias for small datasets.

# %% [markdown]
# ### Analysis of Training vs. Validation Metrics
#
# Throughout the training logs for both tasks, an interesting paradox emerges: the validation metrics (PCC and ROC-AUC) consistently outperform the training metrics, even when the validation loss (MSE/BCE) is strictly worse than the training loss. We assume this phenomenon is driven by two primary factors:
#
# First, the discrepancy highlights the fundamental mathematical difference between absolute error and correlation. For example, in the ESOL regression task, a model might predict values with a massive, consistent offset on the validation set, resulting in a terrible Mean Squared Error (Loss). However, because those predictions still move perfectly in sync with the true values, the Pearson Correlation Coefficient (PCC) remains exceptionally high.
#
# Second, the dataset was partitioned using a `ScaffoldSplitter`, meaning the validation set is not a random sample, but rather consists of distinct chemical families. If the validation set was assigned a chemical scaffold that is structurally simpler or has a more straightforward, linear relationship with the target property, the model will naturally achieve a higher correlation or ROC-AUC score on those "easy" molecules compared to the highly complex, diverse training set. Furthermore, because training metrics are calculated as a running average over the epoch while the model is still learning, the validation metrics, calculated at the very end of the epoch with fully updated weights, naturally receive a slight advantage.

