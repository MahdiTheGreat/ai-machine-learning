# setups for Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import os
import sys
import subprocess

# FIX: Clear Jupyter's import cache for our project. 
# If a previous run failed to find the module, Python permanently caches the failure (negative cache).
for key in list(sys.modules.keys()):
    if key.startswith("sml_project3"):
        del sys.modules[key]

# FIX: In Colab/Jupyter, os.chdir() persists across cell executions.
# If we re-run this script, we don't want to descend deeper into the folders.
# We explicitly reset the working directory to the folder containing this script.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def run_cmd(cmd):
    subprocess.check_call(cmd, shell=True)

installed_anything = False

if not os.path.exists("sml_project3"):
    print("Cloning sml_project3 utilities...")
    run_cmd("git clone https://github.com/olsson-group/sml_project3.git")
    print("Installing requirements...")
    run_cmd(f"{sys.executable} -m pip install -r sml_project3/requirements.txt")
    installed_anything = True

# FIX: The cloned repo's inner package is missing an __init__.py, which causes 
# Python to shadow it with the repo's outer __init__.py, breaking all imports.
# We dynamically create an empty __init__.py here to force it into a regular package.
inner_dir = os.path.join("sml_project3", "sml_project3")
if os.path.exists("sml_project3"):
    os.makedirs(inner_dir, exist_ok=True)
    init_path = os.path.join(inner_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            pass

try:
    import torch_geometric
    import mdtraj
except ImportError:
    print("Installing torch_geometric and mdtraj...")
    run_cmd(f"{sys.executable} -m pip install torch_geometric mdtraj e3nn")
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

# import python packages
import os
import numpy as np
import torch
import torch_geometric as geom
import matplotlib.pyplot as plt

# The sml_project3 utilities expect the current working directory to be the repo root
# so they can find relative paths like data/ and oracle/.weights/
os.chdir("sml_project3")

# import project files
from sml_project3.sml_project3 import data
from sml_project3.sml_project3 import mlops
from sml_project3.sml_project3 import util
from sml_project3.sml_project3 import model
from sml_project3.sml_project3 import painn
from sml_project3.sml_project3 import torsion

from oracle import util as outil

# reload
import importlib
importlib.reload(painn)
importlib.reload(util)
importlib.reload(model)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL: Basedistribution

class BaseDistribution:
    def __init__(self, device="cpu"):
        # The std of the coordinates is 0.1277
        self.std = 0.1277
        self.device = device
    
    def sample(self, n_samples):
        # Generates a tensor of dimension [n_samples * 15, 3] for a batch of pent-1-ene
        # Sample isotropic normal
        samples = torch.randn(n_samples, 15, 3, device=self.device) * self.std
        # Remove translational symmetry by moving center of mass to origin
        samples = samples - samples.mean(dim=1, keepdim=True)
        return samples.view(-1, 3)

# MODEL: BASELINE MODEL

class BaselineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input features: 3 coordinates + 1 time scalar = 4
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 3)
        )

    def forward(self, t, batch):
        # batch.pos is [n_atoms, 3]
        # t is [n_atoms]
        t_expanded = t.unsqueeze(1) if t.dim() == 1 else t
        x = torch.cat([batch.pos, t_expanded], dim=1)
        velocity = self.net(x)
        return velocity


# MODEL: Equivariant readout 

class EquivariantReadout(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # MLP mapping invariant features to weights for the equivariant vectors
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_features),
            torch.nn.SiLU(),
            torch.nn.Linear(n_features, n_features)
        )
    
    def forward(self, batch):
        # Use the variable names from painn.py
        inv = batch.invariant_node_features
        equiv = batch.equivariant_node_features
        
        # Calculate scalar weights from invariant features
        weights = self.mlp(inv) # [n_atoms, n_features]
        weights = weights.unsqueeze(-1) # [n_atoms, n_features, 1]
        
        # Linear combination of equivariant features
        v_t = (equiv * weights).sum(dim=1) # [n_atoms, 3]
        return v_t

import sys
DEBUG_MODE = '--test' in sys.argv
n_epochs = 1 if DEBUG_MODE else 1000
if DEBUG_MODE:
    print("\n[WARNING] FAST DEBUG MODE")

readout = EquivariantReadout(n_features=8).to(device)
basedistribution = BaseDistribution(device=device)

baseline_score = BaselineModel().to(device)
baseline_cfm = model.CFM(baseline_score, basedistribution).to(device)

painn_score = painn.Painn(n_features=8, readout=readout).to(device)
painn_cfm = model.CFM(painn_score, basedistribution).to(device)

dataset = data.Pentene1Dataset()
if DEBUG_MODE:
    dataset = torch.utils.data.Subset(dataset, range(10))
dataloader = geom.loader.DataLoader(dataset, batch_size=128, shuffle=True)

def train_cfm(cfm_model, name):
    print(f"\n--- Training {name} ---")
    timer = util.Timer()
    os.makedirs(f"results/{name}", exist_ok=True)
    step = 0
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            # t is a random time between 0 and 1 for each graph in the batch
            t = torch.rand(len(batch)).type_as(batch.pos).to(device)
            loss = cfm_model.get_loss(t, batch)
            cfm_model.training_step(loss)
            epoch_loss += loss.item()
            step += 1
            
            if (step + 1) % 1000 == 0 and not DEBUG_MODE:
                mlops.save(cfm_model, f"results/{name}/model_{step}.pkl")

        epoch_loss /= max(1, len(dataloader))
        cfm_model.on_epoch_end(epoch_loss)
        if epoch % max(1, (n_epochs // 10)) == 0 or epoch == n_epochs - 1:
            print(f"Epoch: {epoch}, Step: {step}, Time: {timer}, Loss: {epoch_loss:.4f}")
            
    mlops.save(cfm_model, f"results/{name}/model_latest.pkl")
    return cfm_model

baseline_cfm = train_cfm(baseline_cfm, "Baseline")
painn_cfm = train_cfm(painn_cfm, "PaiNN")

# SAMPLING & EVALUATION:
from scipy.stats import entropy

def generate_and_evaluate(cfm, name):
    print(f"\n=== Evaluating {name} ===")
    cfm.eval()
    n_samples = 10 if DEBUG_MODE else 100
    base_dataset = data.BaseDistributionDataset(n_samples, basedistribution)
    base_loader = geom.loader.DataLoader(base_dataset, batch_size=512, shuffle=False)
    
    print(f"Sampling {n_samples} molecules...")
    samples = cfm.sample(base_loader, n_steps=100)
    samples = samples.cpu().numpy()
    
    # Evaluate Torsions
    torsion_evaluator = torsion.TorsionEvaluator()
    sampled_torsions = torsion_evaluator.evaluate(samples[-1])
    
    # Compute reference dataset torsions
    ref_dataset = data.Pentene1Dataset()
    if DEBUG_MODE:
        ref_dataset = torch.utils.data.Subset(ref_dataset, range(10))
    ref_loader = geom.loader.DataLoader(ref_dataset, batch_size=512, shuffle=False)
    
    ref_pos = []
    for batch in ref_loader:
        ref_pos.append(batch.pos.view(-1, 15, 3).numpy())
    ref_pos = np.concatenate(ref_pos, axis=0)
    ref_torsions = torsion_evaluator.evaluate(ref_pos)
    
    # Evaluate energies with Oracle
    print("Evaluating energies with Oracle...")
    oracle = outil.load_oracle(device=device)
    
    # Dataset of generated samples
    sol_samples = torch.tensor(samples[-1], device=device, dtype=torch.float32)
    samples_dataset = data.Pentene1Dataset(sol_samples.cpu().numpy())
    samples_dataloader = geom.loader.DataLoader(samples_dataset, batch_size=512, shuffle=False)
    
    oracles = []
    for batch in samples_dataloader:
        batch = batch.to(device)
        oracle_energies = oracle.get_energy(batch)
        oracles.append(oracle_energies.detach().cpu().numpy())
    oracle_energies = np.concatenate(oracles, axis=0)
    
    # Reference energies
    refs = []
    for batch in ref_loader:
        batch = batch.to(device)
        ref_energy = oracle.get_energy(batch)
        refs.append(ref_energy.detach().cpu().numpy())
    ref_energies = np.concatenate(refs, axis=0)
    
    # KL Divergence Metric
    def kl_div(p, q, bins=50):
        min_val = min(p.min(), q.min())
        max_val = max(p.max(), q.max())
        p_hist, _ = np.histogram(p, bins=bins, range=(min_val, max_val), density=True)
        q_hist, _ = np.histogram(q, bins=bins, range=(min_val, max_val), density=True)
        p_hist = p_hist + 1e-8
        q_hist = q_hist + 1e-8
        return entropy(p_hist, q_hist)
        
    kl_torsion = kl_div(sampled_torsions, ref_torsions)
    kl_energy = kl_div(oracle_energies, ref_energies)
    
    print(f"KL Divergence (Torsion): {kl_torsion:.4f}")
    print(f"KL Divergence (Energy):  {kl_energy:.4f}")

generate_and_evaluate(baseline_cfm, "Baseline")
generate_and_evaluate(painn_cfm, "PaiNN")
