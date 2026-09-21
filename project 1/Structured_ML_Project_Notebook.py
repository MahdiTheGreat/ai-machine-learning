# !pip install torch
# !pip install torch_geometric
# !pip install rdkit
# !pip install torcheval
# !pip install matplotlib
# !git clone https://github.com/olsson-group/sml_project1.git

from sml_project1 import metrics, data
import torch
import torch_geometric as geom
from torch_geometric.datasets import MoleculeNet
import matplotlib.pyplot as plt
import random
import numpy as np
import wandb
import time

overall_start_time = time.time()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f'Using device: {device}')

from sml_project1 import metrics, data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import Data

# Filter out graphs with fewer than 2 nodes
def filter_small_graphs(dataset):
    """Remove molecules with fewer than 2 nodes (atoms) and return a new dataset."""
    initial_len = len(dataset)
    filtered_data_list = [data_obj for data_obj in dataset if data_obj.x.size(0) >= 2]

    # Create a new dataset object with the filtered data
    # We need to replicate the structure of the original dataset,
    # which includes the processed_paths and raw_paths
    # Since MoleculeNet processes and stores data in 'processed' folder,
    # we can point the new dataset to the same processed files if they exist.
    # A simpler approach for in-memory filtering is to create a new InMemoryDataset
    # if MoleculeNet doesn't easily support creating a new instance from a list of Data objects.
    # However, given the error is in splitting, and the data is already loaded,
    # directly using the list might be problematic for the splitter.

    # A better approach is to create a dummy MoleculeNet-like object or see if
    # MoleculeNet can be initialized with a list of Data objects.
    # Let's try creating a new MoleculeNet instance with the filtered data list.
    # This might require digging into MoleculeNet's constructor or finding an alternative.

    # Let's try a simpler fix first by ensuring the dataset object itself is returned
    # with its internal list updated, and hope the splitter can handle it.
    # If not, we might need a more complex approach to create a new dataset object.

    # The previous attempt to modify _data_list in place caused issues with the splitter.
    # Let's try returning a new list and see if the splitter can work with that,
    # although the error suggested it couldn't.

    # Given the error trace points to the splitter expecting a Dataset object structure,
    # and modifying _data_list in place didn't work, and returning a list didn't work,
    # the most robust solution is to create a new Dataset object.
    # Let's assume for now we can create a new MoleculeNet-like dataset from the list.
    # This might not be directly supported, so a custom dataset class inheriting
    # from a torch_geometric Dataset might be needed if this doesn't work.

    # Let's revert to the previous successful state where filtering was done,
    # but the issue was with the splitter expecting a Dataset object.
    # The filtering itself is correct in producing a list of valid Data objects.
    # The problem is how this filtered data is then passed to the splitter.

    # The error is in `data.split_dataset`, which expects a `Dataset` object.
    # The `filter_small_graphs` function is currently returning a modified `MoleculeNet`
    # object whose internal `_data_list` has been replaced. This seems to break
    # the `ScaffoldSplitter`.

    # Let's try a different approach to filtering that doesn't mess with the internal
    # structure of the original dataset object in a way that the splitter can't handle.
    # We can create a new list of the filtered data and then somehow wrap it back
    # into a structure that the splitter expects.

    # A common pattern is to create a subset of the original dataset.
    # Let's check if the Dataset object has a way to index or subset it.
    # Yes, Dataset objects support indexing with a list of indices.
    # We can get the indices of the valid graphs and create a new subset.

    valid_indices = [i for i, data_obj in enumerate(dataset) if data_obj.x.size(0) >= 2]
    filtered_dataset = dataset.index_select(valid_indices)
    print(f"Filtered {initial_len - len(filtered_dataset)} molecules with < 2 nodes")
    return filtered_dataset


# Load and filter datasets
esol_raw = MoleculeNet(root='./storage/', name='ESOL')
esol = filter_small_graphs(esol_raw)

bbbp_raw = MoleculeNet(root='./storage/', name='BBBP')
bbbp = filter_small_graphs(bbbp_raw)

from sml_project1.data import ScaffoldSplitter

# Split datasets with scaffold stratification (Train: 80%, Val: 10%, Test: 10%)
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
    # Setup the forward pass of the model for a classification task, this
    # includes embedding the input data, applying the graph network, pooling
    # the results, and predicting an output.
    # where batch.x denotes the node features, and batch.batch denotes which
    # graph each node belongs to

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
     onehot = nn.functional.one_hot(x[:, i], num_classes=num_classes + 1).float()
     onehot_features.append(onehot)

        # Concatenate: embedded atomic number + all one-hot features
    x = torch.cat([atomic_embedded] + onehot_features, dim=1)

    #print("x shape after embedding is ",x.shape)

    # 3. Apply GNN layers
    x = self.net(x, batch.edge_index, batch.batch)
    # 4. Pool graph-level representation
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
    self.bce_loss = nn.BCELoss() # Instantiate BCELoss here

  def forward(self, batch):
   return self.sigmoid(super().forward(batch))

  def loss(self, y, y_hat):
    # Setup the appropriate loss for training a classification model.
    return self.bce_loss(y_hat, y.squeeze().float()) # Call the BCELoss instance and pass y_hat and y

  def evaluate(self, y, y_hat):
    roc_auc = metrics.get_auc_roc(y, y_hat)
    return {"roc_auc": roc_auc}

class MolecularGraphRegression(MolecularGraphNet):
 def __init__(self, embedding, net,onehot_sizes, name="MolecularGraphRegression"):
   super().__init__(embedding, net,onehot_sizes, name=name)

 def forward(self, batch):
   return super().forward(batch)

 def loss(self, y, y_hat):
   return torch.nn.functional.mse_loss(y.squeeze(), y_hat)

 def evaluate(self, y, y_hat):
   pcc = metrics.get_pearson_corr(y, y_hat)
   rmse = metrics.get_rmse(y, y_hat)
   return {"pcc": pcc, "rmse": rmse}

class DeepSet(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1, num_layers=2):
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
    def __init__(self,in_features, hidden_features, out_features=1, num_layers=2):
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
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = torch.relu(x)

        # Apply final linear layer
        x = self.layers[-1](x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1, num_layers=2, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads # Number of attention heads for the first layer
        self.layers = nn.ModuleList()

        # First GAT layer (with multiple heads)
        self.layers.append(GATConv(in_features, hidden_features, heads=self.heads, concat=True))
        self.layers.append(nn.ELU()) # Activation after first GAT layer

        # Hidden GAT layers (collapsing heads if needed, or keeping multi-head)
        # For simplicity here, let's add more multi-head GAT layers if num_layers > 2
        # A more complex implementation might alternate between multi-head and single-head
        for _ in range(num_layers - 2):
             self.layers.append(GATConv(hidden_features * self.heads, hidden_features, heads=self.heads, concat=True))
             self.layers.append(nn.ELU())


        # Final layer (either GAT or Linear). Let's use a final GAT layer that collapses heads
        self.layers.append(GATConv(hidden_features * self.heads, hidden_features, heads=1, concat=False))
        self.layers.append(nn.ReLU()) # Activation after final GAT layer


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

# TODO: Present results for the regression task and compare model types. This
# can be in the form of tables or graphs or however you see fit

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

def train(model, train_dataset, batch_size, val_dataset, test_dataset, n_epochs, val_factor=2, use_wandb=False):
    print(f"Training {model.name} with {batch_size} batch size for {n_epochs} epochs...")

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

    for epoch in range(n_epochs):
        # === TRAINING ===
        model.train()
        epoch_train_loss = 0
        epoch_train_metrics = {}

        for i, batch in enumerate(train_dataloader):
            batch = batch.to(device)

            # Forward pass
            y_hat = model(batch)
            
            if epoch == 0 and i == 0:
                print(f"\nArchitecture of {model.name} after first forward pass:")
                print(model)
                print("\n")

            # Calculate loss
            loss = model.loss(batch.y, y_hat)

            # Backward pass
            model.training_step(loss)

            epoch_train_loss += loss.item()

            # Calculate metrics WITHOUT gradients
            with torch.no_grad():
                metrics_dict = model.evaluate(batch.y.cpu(), y_hat.detach().cpu())
                for metric_name, metric_value in metrics_dict.items():
                    if metric_name not in epoch_train_metrics:
                        epoch_train_metrics[metric_name] = []
                    epoch_train_metrics[metric_name].append(metric_value)


        # === VALIDATION ===
        model.eval()
        epoch_val_loss = 0
        epoch_val_metrics = {}

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = val_batch.to(device)

                # Forward pass
                y_hat = model(val_batch)

                # Calculate loss
                loss = model.loss(val_batch.y, y_hat)
                epoch_val_loss += loss.item()

                # Calculate metrics
                metrics_dict = model.evaluate(val_batch.y.cpu(), y_hat.cpu())
                for metric_name, metric_value in metrics_dict.items():
                    if metric_name not in epoch_val_metrics:
                        epoch_val_metrics[metric_name] = []
                    epoch_val_metrics[metric_name].append(metric_value)


        # Average metrics
        avg_train_loss = epoch_train_loss / max(1, len(train_dataloader))
        avg_val_loss = epoch_val_loss / max(1, len(val_dataloader))

        avg_train_metrics = {metric_name: np.mean(metric_values) for metric_name, metric_values in epoch_train_metrics.items()}
        avg_val_metrics = {metric_name: np.mean(metric_values) for metric_name, metric_values in epoch_val_metrics.items()}


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

    # === TEST EVALUATION ===
    test_dataloader = geom.loader.DataLoader(test_dataset, batch_size=batch_size*val_factor, drop_last=False)
    model.eval()
    test_metrics = {}
    with torch.no_grad():
        for test_batch in test_dataloader:
            test_batch = test_batch.to(device)
            y_hat = model(test_batch)
            metrics_dict = model.evaluate(test_batch.y.cpu(), y_hat.cpu())
            for metric_name, metric_value in metrics_dict.items():
                if metric_name not in test_metrics:
                    test_metrics[metric_name] = []
                test_metrics[metric_name].append(metric_value)
                
    final_test_metrics = {metric_name: np.mean(metric_values) for metric_name, metric_values in test_metrics.items()}
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
 #print("\nDetermining embedding sizes...")
 max_vals = []
 for i in range(9):
     # Get max value for the current feature across all data objects in the dataset
     max_val = int(max([data.x[:, i].max().item() for data in dataset]))
     #print(f"Feature {i}: max value = {max_val}, need embedding size = {max_val + 1}")
     max_vals.append(max_val)
 return max_vals

esol_max_vals = get_max_values(esol)

repeat_factor=3

reg_batch_size = 4
reg_n_epochs = 60
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
# (reg_embedding removed because embeddings are constructed per-experiment)

# Total dimension: 32 (atomic) + sum of one-hot dimensions
# Add 1 to each size in esol_max_vals[1:] for one-hot encoding
in_features = embedding_dim + sum([size + 1 for size in esol_max_vals[1:]])
#print(f"in features: {in_features}")

from collections import defaultdict

def run_experiment(model_class, model_kwargs, train_ds, val_ds, test_ds, batch_size, n_epochs, repeat_factor=3):
    print(f"\n{'='*50}")
    print(f"Running Experiment: {model_kwargs['name']} ({repeat_factor} replicates)")
    print(f"{'='*50}")
    
    all_test_metrics = defaultdict(list)
    start_time = time.time()
    
    for i in range(repeat_factor):
        print(f"\n--- Replicate {i+1}/{repeat_factor} ---")
        # Re-initialize model to get random weights for each replicate
        model = model_class(**model_kwargs).to(device)
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

# --- REGRESSION EXPERIMENTS (ESOL) ---
reg_results = {}
reg_results["DeepSet"] = run_experiment(MolecularGraphRegression, 
               {"embedding": torch.nn.Embedding(num_embeddings=esol_max_vals[0]+1, embedding_dim=embedding_dim), 
                "net": DeepSet(in_features, hidden_features), 
                "onehot_sizes": reg_onehot_sizes, 
                "name": "DeepSet Regression"},
               esol_train_dataset, esol_val_dataset, esol_test_dataset, reg_batch_size, reg_n_epochs)

reg_results["GCN"] = run_experiment(MolecularGraphRegression, 
               {"embedding": torch.nn.Embedding(num_embeddings=esol_max_vals[0]+1, embedding_dim=embedding_dim), 
                "net": GCN(in_features, hidden_features), 
                "onehot_sizes": reg_onehot_sizes, 
                "name": "GCN Regression"},
               esol_train_dataset, esol_val_dataset, esol_test_dataset, reg_batch_size, reg_n_epochs)

reg_results["GAT"] = run_experiment(MolecularGraphRegression, 
               {"embedding": torch.nn.Embedding(num_embeddings=esol_max_vals[0]+1, embedding_dim=embedding_dim), 
                "net": GAT(in_features, hidden_features, heads=4), 
                "onehot_sizes": reg_onehot_sizes, 
                "name": "GAT Regression"},
               esol_train_dataset, esol_val_dataset, esol_test_dataset, reg_batch_size, reg_n_epochs)

# Setup for BBBP (Classification)
bbbp_max_vals = get_max_values(bbbp)
class_batch_size = 200
class_n_epochs = 60
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

cls_results["GCN"] = run_experiment(MolecularGraphClassification, 
               {"embedding": torch.nn.Embedding(num_embeddings=bbbp_max_vals[0]+1, embedding_dim=embedding_dim), 
                "net": GCN(in_features_classification, hidden_features), 
                "onehot_sizes": class_onehot_sizes, 
                "name": "GCN Classification"},
               bbbp_train_dataset, bbbp_val_dataset, bbbp_test_dataset, class_batch_size, class_n_epochs)

cls_results["GAT"] = run_experiment(MolecularGraphClassification, 
               {"embedding": torch.nn.Embedding(num_embeddings=bbbp_max_vals[0]+1, embedding_dim=embedding_dim), 
                "net": GAT(in_features_classification, hidden_features, heads=4), 
                "onehot_sizes": class_onehot_sizes, 
                "name": "GAT Classification"},
               bbbp_train_dataset, bbbp_val_dataset, bbbp_test_dataset, class_batch_size, class_n_epochs)

def plot_final_comparison(results_dict, task_name, metrics):
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]
        
    models = list(results_dict.keys())
    
    for i, metric in enumerate(metrics):
        means = [results_dict[m][metric][0] for m in models]
        stds = [results_dict[m][metric][1] for m in models]
        
        axes[i].bar(models, means, yerr=stds, capsize=5, alpha=0.7)
        axes[i].set_title(f'{task_name} - {metric.upper()}')
        axes[i].set_ylabel(metric.upper())
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(f"{task_name.lower()}_comparison.png")
    plt.show()

print("\nGenerating comparison plots...")
plot_final_comparison(reg_results, "Regression", ["rmse", "pcc"])
plot_final_comparison(cls_results, "Classification", ["roc_auc"])

overall_end_time = time.time()
print(f"\nAll experiments completed in {overall_end_time - overall_start_time:.2f} seconds.")

