import os
import sys
import subprocess

def run_cmd(cmd):
    subprocess.check_call(cmd, shell=True)

if not os.path.exists("sml_project2"):
    print("Cloning sml_project2 utilities...")
    run_cmd("git clone https://github.com/olsson-group/sml_project2.git")
    print("Installing requirements...")
    run_cmd(f"{sys.executable} -m pip install -r sml_project2/requirements.txt")
    run_cmd(f"{sys.executable} -m pip install e3nn nglview")

try:
    import torch_scatter
except ImportError:
    print("Installing torch-scatter & torch-cluster (pre-built wheels)...")
    import torch
    torch_version = torch.__version__.split('+')[0]
    cuda_version = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html"
    run_cmd(f"{sys.executable} -m pip install torch-scatter -f {url}")
    run_cmd(f"{sys.executable} -m pip install torch-cluster -f {url}")

import random

import e3nn
import nglview
from scipy.spatial.transform import Rotation
import torch
import torch_geometric as geom
from torch_scatter import scatter

from sml_project2 import data
from sml_project2 import utils

# DATA: Download all the datasets
data.download_all(path='data/MD17')

# UNITTEST: Setup test data

def get_test_batch(batch_size=2, n_atoms=9):
  atom_types = [1,6,7,8]

  test_batch = geom.data.Batch.from_data_list([
        geom.data.Data(
            pos=torch.randn(n_atoms, 3),
            z=torch.tensor(random.choices(atom_types, k=n_atoms)),
            energy = torch.randn(1),
            force = torch.randn(n_atoms, 3)
        )
        for _ in range(batch_size)
  ])
  return test_batch

# UNITTEST: Test model can forward and forces and energies have the correct shapes
# TODO Make Sure E3MPNN can forward and output correct dimensions of data

test_batch = get_test_batch()
print(test_batch)

# MODEL: Setup the MLFF baseclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
class MLFF(torch.nn.Module):
  def __init__(self, net):
    super().__init__()
    self.net = net
    self.setup_optimizer()

  def training_step(self, loss):
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    if self.lr_scheduler is not None:
      # HINT: If using lr scheduler you can consider monitoring the learning
      # rate
      self.lr_scheduler.step()

  def setup_optimizer(self):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # Adam has an adaptive mechanism (per-parameter scaling).
    # Schedulers control the global LR schedule over time.
    # Both affect learning rates, but at different levels (global vs. per-parameter adaptation).
    # Check https://pytorch.org/docs/stable/optim.html#module-torch.optim.lr_scheduler
    # for available learning rate schedulers

  def forward(self, batch, return_force=False):
    # HINT: requiring grad on positions is necessary for computing forces.
    if return_force:
      batch.pos.requires_grad = True

    energy = self.net(batch)

    if return_force:
      force = self.get_force(batch, energy)
      return energy, force

    return energy

  def get_force(self, batch, energy):
    force = - torch.autograd.grad(
        energy,
        batch.pos,
        grad_outputs=torch.ones_like(energy),
        retain_graph=True,
        create_graph=True,
    )[0]
    return force

  def loss_fn(self, y, y_hat):
    # TODO: Implement an appropriate loss.
    return torch.nn.MSELoss()(y, y_hat)


while True:
  pass

# MODEL: Setup the Baseline class

class Baseline(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    # Invariant features only (lmax=0)
    self.mp = e3nn.nn.models.v2103.gate_points_message_passing.MessagePassing(
        irreps_node_input="1x0e",
        irreps_node_hidden="16x0e",
        irreps_node_output="1x0e",
        irreps_node_attr="1x0e",
        irreps_edge_attr="1x0e", 
        layers=3,
        fc_neurons=[10, 32],
        num_neighbors=10.0
    )

  def forward(self, batch):
    node_features = torch.ones(batch.pos.shape[0], 1, device=batch.pos.device)
    node_attr = torch.ones(batch.pos.shape[0], 1, device=batch.pos.device)
    
    edge_src, edge_dst = [], []
    for i in range(batch.num_graphs):
        nodes = (batch.batch == i).nonzero(as_tuple=True)[0]
        n = len(nodes)
        src = nodes.repeat_interleave(n)
        dst = nodes.repeat(n)
        edge_src.append(src)
        edge_dst.append(dst)
    
    edge_src = torch.cat(edge_src)
    edge_dst = torch.cat(edge_dst)
    
    edge_vec = batch.pos[edge_dst] - batch.pos[edge_src]
    edge_length = edge_vec.norm(dim=1)
    
    edge_scalars = e3nn.math.soft_one_hot_linspace(
        edge_length,
        start=0.0,
        end=10.0,
        number=10,
        basis='cosine',
        cutoff=True
    )
    
    edge_attr = e3nn.o3.spherical_harmonics("0e", edge_vec, normalize=True, normalization='component')
    node_outputs = self.mp(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)
    energy = scatter(node_outputs, batch.batch, dim=0, reduce="sum")
    return energy

# MODEL: Setup the E3MPNN network

class E3MPNN(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    # Equivariant features (lmax>0)
    irreps_hidden = e3nn.o3.Irreps("16x0e + 8x1o")
    self.mp = e3nn.nn.models.v2103.gate_points_message_passing.MessagePassing(
        irreps_node_input="1x0e",
        irreps_node_hidden=irreps_hidden,
        irreps_node_output="1x0e",
        irreps_node_attr="1x0e",
        irreps_edge_attr=e3nn.o3.Irreps.spherical_harmonics(1),
        layers=3,
        fc_neurons=[10, 32],
        num_neighbors=10.0
    )

  def forward(self, batch):
    node_features = torch.ones(batch.pos.shape[0], 1, device=batch.pos.device)
    node_attr = torch.ones(batch.pos.shape[0], 1, device=batch.pos.device)
    
    edge_src, edge_dst = [], []
    for i in range(batch.num_graphs):
        nodes = (batch.batch == i).nonzero(as_tuple=True)[0]
        n = len(nodes)
        src = nodes.repeat_interleave(n)
        dst = nodes.repeat(n)
        edge_src.append(src)
        edge_dst.append(dst)
    
    edge_src = torch.cat(edge_src)
    edge_dst = torch.cat(edge_dst)
    
    edge_vec = batch.pos[edge_dst] - batch.pos[edge_src]
    edge_length = edge_vec.norm(dim=1)
    
    edge_scalars = e3nn.math.soft_one_hot_linspace(
        edge_length,
        start=0.0,
        end=10.0,
        number=10,
        basis='cosine',
        cutoff=True
    )
    
    edge_attr = e3nn.o3.spherical_harmonics(self.mp.irreps_edge_attr, edge_vec, normalize=True, normalization='component')
    node_outputs = self.mp(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)
    energy = scatter(node_outputs, batch.batch, dim=0, reduce="sum")
    return energy

# UNITTEST: Set up test_model

test_model = MLFF(net=E3MPNN())
energy, force = test_model(test_batch, return_force=True)

assert energy.shape == (2, 1)
assert force.shape == (18, 3)

# UNITTEST: Test translational symmetries
# TODO: Make sure E3MPNN pass tests for translational symmetries
# NOTE: The baseline model doesn't have to pass these tests

test_batch = get_test_batch()
translated_test_batch = test_batch.clone()
translated_test_batch.pos += 1

energy1, force1 = test_model(test_batch, return_force=True)
energy2, force2 = test_model(translated_test_batch, return_force=True)

# Energy and force are invariant to translations
assert torch.allclose(energy1, energy2)
assert torch.allclose(force1, force2, atol=1e-4)

# UNITTEST: Test rotational symmetries
# TODO: Make sure E3MPNN pass tests for rotational symmetries

test_batch = get_test_batch()
rotated_test_batch = test_batch.clone()

rotation_matrix = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float32)
rotated_test_batch.pos = torch.matmul(rotated_test_batch.pos, rotation_matrix.T)
rotated_test_batch.force = torch.matmul(rotated_test_batch.force, rotation_matrix.T)

energy1, force1 = test_model(test_batch, return_force=True)
energy2, force2 = test_model(rotated_test_batch, return_force=True)
rotated_force1 = torch.matmul(force1, rotation_matrix.T)

# Energy is invariant to rotation and forces are equivariant to rotations
assert torch.allclose(energy1, energy2)
assert torch.allclose(rotated_force1, force2, atol=1e-4)

# UNITTEST: Test for permutational symmetries
# TODO: Make sure E3MPNN pass tests for rotational symmetries

test_batch = get_test_batch(1)
permuted_test_batch = test_batch.clone()

permutation = torch.randperm(test_batch.z.shape[0])

permuted_test_batch.z = test_batch.z[permutation]
permuted_test_batch.pos = test_batch.pos[permutation]
permuted_test_batch.force = test_batch.force[permutation]

energy1, force1 = test_model(test_batch, return_force=True)
energy2, force2 = test_model(permuted_test_batch, return_force=True)

assert torch.allclose(energy1, energy2)
assert torch.allclose(force1[permutation], force2, atol=1e-4)

# DATA: Setup transfom

# We subtract the groundstate energy from each conformation such that the model
# only has to calculate the deviation from the ground state energy.


class Transform():
  def __init__(self, ground_state_energy):
    self.ground_state_energy = ground_state_energy

  def __call__(self, batch):
    # Here we set up a transformation for the dataset. Any operation in here wil
    # be applied to each datapoint before the dataloader collects them and
    # aggregates them into a batch. We simply subtract the ground_state energy.
    batch.energy = batch.energy - self.ground_state_energy
    return batch

# TRAIN: Ethanol

# Setup Ethanol data
dataset = data.get_dataset('ethanol')
ground_state_energy = data.get_ground_state_energy(dataset)
transform = Transform(ground_state_energy)
dataset.transform = transform

n_train = 900
n_val = 50
seed = 42
train_batch_size = 32
val_batch_size = 50

train_dataset, val_dataset = data.split_dataset(
  dataset, n_train=n_train, n_val=n_val, seed=seed
)
train_dataloader = geom.loader.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = geom.loader.DataLoader(val_dataset, batch_size=val_batch_size)

n_epochs = 10

def train_eval_model(model, name):
    print(f"\n--- Training {name} ---")
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            energy = model(batch)
            loss = model.loss_fn(batch.energy, energy)
            model.training_step(loss)
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            batch = batch.to(device)
            energy = model(batch)
            loss = model.loss_fn(batch.energy, energy)
            val_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss/len(train_dataloader):.4f} | Val Loss: {val_loss/len(val_dataloader):.4f}")

    # Evaluate forces on validation set
    print(f"\nEvaluating Forces for {name}:")
    model.eval()
    all_pred_forces = []
    all_true_forces = []
    for batch in val_dataloader:
        batch = batch.to(device)
        _, pred_force = model(batch, return_force=True)
        all_pred_forces.append(pred_force.detach().cpu())
        all_true_forces.append(batch.force.cpu())
    
    pred_f = torch.cat(all_pred_forces)
    true_f = torch.cat(all_true_forces)
    msd = torch.mean((pred_f - true_f)**2)
    rmsd = torch.sqrt(msd)
    print(f"Force MSD: {msd:.4f} | Force RMSD: {rmsd:.4f}\n")

# Train baseline
baseline_model = MLFF(Baseline())
train_eval_model(baseline_model, "Baseline (lmax=0)")

# Train E3MPNN
e3mpnn_model = MLFF(E3MPNN())
train_eval_model(e3mpnn_model, "E3MPNN (lmax=1)")
# HINT: This can code can help you render a molecule - it wont work in colab as
# colab doesn't allow for third party widgets

batch = dataset[0]
pos = batch.pos.detach().numpy()
z = batch.z.detach().numpy()

# HINT: You can also render a trajectory if you decide to experiment with
# integrating langevin dynamics by applying something like
#     pos = torch.stack([dataset[idx].pos for idx in range(20)])

traj = utils.get_mdtraj(pos, z)
view = nglview.show_mdtraj(traj)
view.clear_representations()
view.add_representation('hyperball')
view

