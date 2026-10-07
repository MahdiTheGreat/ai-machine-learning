import os
import sys
import subprocess

def run_cmd(cmd):
    subprocess.check_call(cmd, shell=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
scratch_dir = os.path.abspath(os.path.join(script_dir, "..", "scratch"))

os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if scratch_dir not in sys.path:
    sys.path.insert(0, scratch_dir)

installed_anything = False

scratch_proj2 = os.path.join(scratch_dir, "sml_project2")

if not os.path.exists(scratch_proj2):
    print("Cloning sml_project2 utilities to scratch...")
    os.makedirs(scratch_dir, exist_ok=True)
    run_cmd(f"git clone https://github.com/olsson-group/sml_project2.git {scratch_proj2}")
    print("Installing sml_project2 requirements...")
    run_cmd(f"{sys.executable} -m pip install -r {os.path.join(scratch_proj2, 'requirements.txt')}")
    installed_anything = True

try:
    import e3nn
    import nglview
    import torch_geometric
except ImportError:
    print("Installing e3nn, nglview, and torch_geometric...")
    run_cmd(f"{sys.executable} -m pip install e3nn nglview torch_geometric mdtraj")
    installed_anything = True
except AttributeError as e:
    print("\n" + "="*60)
    print("CRITICAL: PyTorch Geometric circular import detected!")
    print("You MUST restart the Colab Runtime (Runtime -> Restart session) right now.")
    print(f"Error details: {e}")
    print("="*60 + "\n")
    sys.exit(1)

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
    installed_anything = True
if installed_anything:
    print("\n" + "="*60)
    print("CRITICAL: New packages were installed!")
    print("You MUST restart the Colab Runtime (Runtime -> Restart session) before running this script again.")
    print("This prevents PyTorch circular import errors.")
    print("="*60 + "\n")
    sys.exit(1)

import random
import e3nn
from e3nn import o3
from e3nn import math
from e3nn.nn.models.v2103 import gate_points_message_passing
import nglview
from scipy.spatial.transform import Rotation
import torch
import torch_geometric as geom
from torch_geometric.utils import scatter

from sml_project2 import data
from sml_project2 import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

  def setup_optimizer(self):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
    
    # Scheduler Arguments:
    self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=50, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

  def forward(self, batch, return_force=False):
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
    return torch.nn.MSELoss()(y.view(-1), y_hat.view(-1))



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
    # 1. Node Features (The Atoms)
    # We do not use nn.Embedding lookup tables for atom types because MD17 simulates 
    # the exact same molecule in every frame. The network learns purely from spatial geometry,
    # so we hardcode the input to a constant 1.0 for every atom.
    node_features = torch.ones(batch.pos.shape[0], 1, device=batch.pos.device)
    node_attr = torch.ones(batch.pos.shape[0], 1, device=batch.pos.device)
    
    edge_src, edge_dst = [], []
    for i in range(batch.num_graphs):
        nodes = (batch.batch == i).nonzero(as_tuple=True)[0]
        n = len(nodes)
        src = nodes.repeat_interleave(n)
        dst = nodes.repeat(n)
        
        # Remove self-loops to prevent corrupted direction vectors [0,0,0]
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        
        edge_src.append(src)
        edge_dst.append(dst)
    
    edge_src = torch.cat(edge_src)
    edge_dst = torch.cat(edge_dst)
    
    edge_vec = batch.pos[edge_dst] - batch.pos[edge_src]
    edge_length = edge_vec.norm(dim=1)
    
    # 2a. Edge Scalars (Distances)
    # We embed the raw scalar distance between atoms into a 10-dimensional vector
    # using a Cosine Radial Basis Function (RBF).
    edge_scalars = e3nn.math.soft_one_hot_linspace(
        edge_length,
        start=0.0,
        end=10.0,
        number=10,
        basis='cosine',
        cutoff=True
    )
    
    # 2b. Edge Attributes (Directions)
    # To make the network E(3) equivariant (understanding 3D rotation), we embed the 
    # raw 3D direction vectors into Spherical Harmonics representations.
    edge_attr = e3nn.o3.spherical_harmonics(self.mp.irreps_edge_attr, edge_vec, normalize=True, normalization='component')
    node_outputs = self.mp(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)
    energy = scatter(node_outputs, batch.batch, dim=0, reduce="sum")
    return energy

# MODEL: Setup the E3MPNN network

class E3MPNN(Baseline):
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
# 
# 1. Numerical Stability: Absolute quantum energies are massive (e.g. -120,000 kcal/mol) 
#    but fluctuate by tiny amounts (e.g. 1 kcal/mol). Subtracting the baseline prevents 
#    these tiny fluctuations from getting lost in floating-point math errors.
# 2. Physics: Forces are the derivative of energy (F = -dE/dx). Subtracting a massive 
#    constant baseline from the energy doesn't change the derivative or the forces at all!


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
import time
overall_start_time = time.time()
dataset = data.get_dataset('ethanol')
print(f"Loaded Ethanol Dataset: {len(dataset)} molecules")
ground_state_energy = data.get_ground_state_energy(dataset)
transform = Transform(ground_state_energy)
dataset.transform = transform

import sys
DEBUG_MODE = '--test' in sys.argv

n_train = 10 if DEBUG_MODE else 900
n_val = 10 if DEBUG_MODE else 50
seed = 42
train_batch_size = 32
val_batch_size = 50

if DEBUG_MODE:
    print("\n[WARNING] FAST DEBUG MODE")

n_test = len(dataset) - n_train - n_val
train_dataset, val_dataset, test_dataset = data.split_dataset(
  dataset, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed
)
print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
train_dataloader = geom.loader.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_dataloader = geom.loader.DataLoader(val_dataset, batch_size=val_batch_size)

n_epochs = 1 if DEBUG_MODE else 200

def train_eval_model(model, name, train_dl, val_dl, test_dl, epochs=None):
    if epochs is None:
        epochs = n_epochs
    print(f"\n--- Training {name} for {epochs} epochs ---")
    start_time = time.time()
    model = model.to(device)
    model.setup_optimizer() # Re-initialize optimizer so it attaches to the GPU parameters!
    
    best_val_loss = float('inf')
    best_weights = None
    early_stop_patience = 150
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_dl):
            batch = batch.to(device)
            energy, force = model(batch, return_force=True)
            
            if epoch == 0 and i == 0:
                print(f"\nArchitecture for {name} after first forward pass:")
                print(model)
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total Trainable Parameters: {total_params:,}\n")
                
            loss_e = model.loss_fn(batch.energy, energy)
            loss_f = model.loss_fn(batch.force, force)
            loss = 0.01 * loss_e + 0.99 * loss_f
            
            model.training_step(loss)
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        # We need gradients for inputs to compute forces, but not for weights
        torch.set_grad_enabled(True) 
        for batch in val_dl:
            batch = batch.to(device)
            energy, force = model(batch, return_force=True)
            loss_e = model.loss_fn(batch.energy, energy)
            loss_f = model.loss_fn(batch.force, force)
            loss = 0.01 * loss_e + 0.99 * loss_f
            val_loss += loss.item()
            
        avg_train_loss = train_loss/len(train_dl)
        avg_val_loss = val_loss/len(val_dl)
        
        if model.lr_scheduler is not None:
            model.lr_scheduler.step(avg_val_loss)
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            import copy
            best_weights = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
            
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1 or early_stop_counter >= early_stop_patience:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        if early_stop_counter >= early_stop_patience:
            print(f"Traditional Early stopping triggered at Epoch {epoch+1}! Restoring best model weights...")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    end_time = time.time()
    print(f"Training {name} completed in {end_time - start_time:.2f} seconds")

    # ---------------------------------------------------------
    # ASSIGNMENT REQUIREMENT SATISFIED: "Average test errors"
    # Evaluate on the unseen TEST set and compute uncertainties.
    # ---------------------------------------------------------
    print(f"\nEvaluating Forces and Energies on TEST set for {name}:")
    model.eval()
    all_pred_forces = []
    all_true_forces = []
    all_pred_energies = []
    all_true_energies = []
    
    for batch in test_dl:
        batch = batch.to(device)
        pred_energy, pred_force = model(batch, return_force=True)
        all_pred_forces.append(pred_force.detach().cpu())
        all_true_forces.append(batch.force.cpu())
        all_pred_energies.append(pred_energy.detach().cpu())
        all_true_energies.append(batch.energy.cpu())
    
    pred_f = torch.cat(all_pred_forces)
    true_f = torch.cat(all_true_forces)
    f_msd = torch.mean((pred_f - true_f)**2)
    f_rmsd = torch.sqrt(f_msd)
    
    pred_e = torch.cat(all_pred_energies).view(-1)
    true_e = torch.cat(all_true_energies).view(-1)
    e_msd = torch.mean((pred_e - true_e)**2)
    e_rmsd = torch.sqrt(e_msd)
    
    # Standard deviation across the test set for uncertainties
    f_err_std = torch.std(torch.sqrt(torch.sum((pred_f - true_f)**2, dim=1)))
    e_err_std = torch.std(torch.abs(pred_e - true_e))
    
    print(f"  Energy MSD: {e_msd:.4f} | Energy RMSD: {e_rmsd:.4f} ± {e_err_std:.4f}")
    print(f"  Force MSD:  {f_msd:.4f} | Force RMSD:  {f_rmsd:.4f} ± {f_err_std:.4f}\n")

# ---------------------------------------------------------
# ASSIGNMENT REQUIREMENT SATISFIED: "At least three model setups"
# We define a 3rd model that uses up to lmax=2 spherical harmonics.
# ---------------------------------------------------------
class E3MPNN_L2(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__()
        irreps_hidden = e3nn.o3.Irreps("16x0e + 8x1o + 8x2e")
        self.mp = e3nn.nn.models.v2103.gate_points_message_passing.MessagePassing(
            irreps_node_input="1x0e",
            irreps_node_hidden=irreps_hidden,
            irreps_node_output="1x0e",
            irreps_node_attr="1x0e",
            irreps_edge_attr=e3nn.o3.Irreps.spherical_harmonics(2),
            layers=3,
            fc_neurons=[10, 32],
            num_neighbors=10.0
        )

# Train baseline
baseline_model = MLFF(Baseline())
train_eval_model(baseline_model, "Baseline (lmax=0)", train_dataloader, val_dataloader, geom.loader.DataLoader(test_dataset, batch_size=val_batch_size))

# Train E3MPNN
e3mpnn_model = MLFF(E3MPNN())
train_eval_model(e3mpnn_model, "E3MPNN (lmax=1)", train_dataloader, val_dataloader, geom.loader.DataLoader(test_dataset, batch_size=val_batch_size))

# Train E3MPNN_L2
e3mpnn_l2_model = MLFF(E3MPNN_L2())
train_eval_model(e3mpnn_l2_model, "E3MPNN_L2 (lmax=2)", train_dataloader, val_dataloader, geom.loader.DataLoader(test_dataset, batch_size=val_batch_size))

# ---------------------------------------------------------
# ASSIGNMENT REQUIREMENT SATISFIED: "Retrain these three models using only half of the training data."
# ---------------------------------------------------------
print("\n" + "="*50)
print("RETRAINING MODELS ON HALF DATA (450 examples)")
print("="*50)
half_train_dataset = torch.utils.data.Subset(train_dataset, range(len(train_dataset) // 2))
half_train_dataloader = geom.loader.DataLoader(half_train_dataset, batch_size=train_batch_size, shuffle=True)

baseline_model_half = MLFF(Baseline())
train_eval_model(baseline_model_half, "Baseline (lmax=0) [HALF DATA]", half_train_dataloader, val_dataloader, geom.loader.DataLoader(test_dataset, batch_size=val_batch_size))

e3mpnn_model_half = MLFF(E3MPNN())
train_eval_model(e3mpnn_model_half, "E3MPNN (lmax=1) [HALF DATA]", half_train_dataloader, val_dataloader, geom.loader.DataLoader(test_dataset, batch_size=val_batch_size))

e3mpnn_l2_model_half = MLFF(E3MPNN_L2())
train_eval_model(e3mpnn_l2_model_half, "E3MPNN_L2 (lmax=2) [HALF DATA]", half_train_dataloader, val_dataloader, geom.loader.DataLoader(test_dataset, batch_size=val_batch_size))

# HINT: This can code can help you render a molecule - it wont work in colab as
# colab doesn't allow for third party widgets

batch = dataset[0]
pos = torch.as_tensor(batch.pos)
z = torch.as_tensor(batch.z)

# HINT: You can also render a trajectory if you decide to experiment with
# integrating langevin dynamics by applying something like
#     pos = torch.stack([dataset[idx].pos for idx in range(20)])

try:
    import py3Dmol
    traj = utils.get_mdtraj(pos, z)
    traj[0].save_pdb("temp_mol.pdb")
    with open("temp_mol.pdb", "r") as f:
        pdb_string = f.read()
    view = py3Dmol.view(width=400, height=400)
    view.addModel(pdb_string, 'pdb')
    view.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
    view.zoomTo()
    view.show()
except ImportError:
    print("Please run '!pip install py3Dmol' in a cell above to see the rendering in Colab!")

# TRAJECTORY ANIMATION:
try:
    import py3Dmol
    # Grab 50 sequential frames from the dataset
    pos_traj = torch.stack([dataset[idx].pos for idx in range(50)])
    traj_anim = utils.get_mdtraj(pos_traj, z)
    
    # Save the entire trajectory to a multi-model PDB file
    traj_anim.save_pdb("temp_traj.pdb")
    with open("temp_traj.pdb", "r") as f:
        pdb_string_anim = f.read()
        
    view_anim = py3Dmol.view(width=400, height=400)
    view_anim.addModelsAsFrames(pdb_string_anim, 'pdb')
    view_anim.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
    view_anim.animate({'loop': 'forward', 'step': 1})
    view_anim.zoomTo()
    view_anim.show()
except ImportError:
    pass

# OPTIONAL EXTRA TASK: Simulating Langevin Dynamics with our E3MPNN Model
def run_simulation(model, dataset, steps=200):
    print("\n--- Running MD Simulation using E3MPNN ---")
    model.eval()
    # Convert the single datapoint into a proper PyTorch Geometric Batch object
    import torch_geometric as geom
    batch = geom.data.Batch.from_data_list([dataset[0]]).to(device)
    
    pos = batch.pos.clone().detach().cpu()
    vel = torch.randn_like(pos) * 0.01
    
    dt = 0.05 # Increased timestep so the visual movement is actually noticeable over 200 steps!
    friction = 0.01
    
    sim_traj = []
    for step in range(steps):
        batch.pos = pos.to(device).detach()
        # Ask our model to predict the forces!
        _, forces = model(batch, return_force=True)
        forces = forces.detach().cpu()
        
        # Simple Langevin update (F=ma + friction + thermal noise)
        noise = torch.randn_like(vel) * 0.05
        vel = vel + (forces * dt) - (friction * vel) + noise
        pos = pos + (vel * dt)
        
        sim_traj.append(pos.clone())
        
    return torch.stack(sim_traj)

try:
    import py3Dmol
    print("Generating simulation trajectory...")
    mlff_pos_traj = run_simulation(e3mpnn_model, dataset, steps=200)
    mlff_traj_anim = utils.get_mdtraj(mlff_pos_traj, z)
    mlff_traj_anim.save_pdb("mlff_sim.pdb")
    
    with open("mlff_sim.pdb", "r") as f:
        mlff_pdb = f.read()
        
    print("Rendering MLFF Simulation (Scroll down to view!)...")
    view_mlff = py3Dmol.view(width=400, height=400)
    view_mlff.addModelsAsFrames(mlff_pdb, 'pdb')
    view_mlff.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
    view_mlff.animate({'loop': 'forward', 'step': 1, 'interval': 50})
    view_mlff.zoomTo()
    view_mlff.show()
except Exception as e:
    print("Could not render simulation:", e)

overall_end_time = time.time()
print(f"\nAll Project 2 experiments completed in {overall_end_time - overall_start_time:.2f} seconds.")


