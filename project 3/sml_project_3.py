# setups for Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
#     !git clone https://github.com/olsson-group/sml_project3.git
#     !pip install -r sml_project3/requirements.txt

    # Installing torch_scatter and torch_cluster is cumbersome because they are
    # highly optimized implementations that depends on architecure and cuda version.
    # Building these wheels can take up to 20-30 minutes.

#     !pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__)")+cu$(python3 -c "import torch; print(torch.version.cuda.split('.')[0]+torch.version.cuda.split('.')[1])")/torch_scatter.html
#     !pip install torch-cluster -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__)")+cu$(python3 -c "import torch; print(torch.version.cuda.split('.')[0]+torch.version.cuda.split('.')[1])")/torch_cluster.html

# import python packages
import os
import numpy as np
import torch
import torch_geometric as geom
import matplotlib.pyplot as plt

# settings
if IN_COLAB:
    os.chdir(f"/content/sml_project3")

    # import project files
    from sml_project3.sml_project3 import data
    from sml_project3.sml_project3 import mlops
    from sml_project3.sml_project3 import util
    from sml_project3.sml_project3 import model
    from sml_project3.sml_project3 import painn
    from sml_project3.sml_project3 import torsion

else:
    # import project files
    from sml_project3 import data
    from sml_project3 import mlops
    from sml_project3 import util
    from sml_project3 import model
    from sml_project3 import painn
    from sml_project3 import torsion

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
    def __init__(self, ...):
        # TODO: Implement the base distribution for your model. 
        # HINT: The std of the coordinates is 0.1277
        ...
    
    def sample(self, ...):
        # TODO: Implement a sampling function for the basebasedistribtion for your model.
        
        # HINT: This function must return a tensor of dimension [n_atoms, 3]
        ...

# MODEL: BASELINE MODEL

class BaselineModel:
    def __init__(self, ...):
        # TODO: Implement a baseline model for your experiments
        ...

    def forward(self, batch):
        # TODO: Implement the forward pass of the baseline model.
        # HINT: This function must return a tensor of dimension [n_atoms, 3]
        ...


# MODEL: Equivariant readout 

class EquivariantReadout(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        # TODO: Implement the equivariant readout function for a the painn model 
        ...

    
    def forward(self, batch):
        # TODO: Implement the readout function for the painn model which embeds all nodes
        # with invariant and equivariant features
    
        # HINT: The input batch will have in and equivariant on it with shapes
        # batch.equivariant_features.shape == (n_atoms, n_features, 3)
        # batch.invariant_features.shape == (n_atoms, n_features) 

        # HINT: This function should return a tensor of shape [n_atoms, 3]
        ... 

# SETUP: set up the training script

readout = EquivariantReadout(...)
basedistribution = BaseDistribution(...)
score = painn.Painn(n_features=..., readout=readout)
cfm = model.CFM(score, basedistribution)

dataset = data.Pentene1Dataset()
dataloader = geom.data.DataLoader(dataset, batch_size=128, shuffle=True)


# Training

# TODO: train your Baseline model and compare it against your Painn model
# NOTE: you can resume the training of some model using: cfm = mlops.load("results/model/model_latest.pkl")

timer = util.Timer()
step = 0
for epoch in range(1000):
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        t = torch.rand(len(batch)).type_as(batch.pos)
        loss = cfm.get_loss(t, batch)
        cfm.training_step(loss)
        step += 1

        if (step + 1) % 1000 == 0:
            # HINT mlops.save(object, path) will save the pickled object at path, and likewise object = mlops.load(path) will load back the pickled object. 
            # Use this if you want to save your model during training

            mlops.save(cfm, f"results/model/model_{step}.pkl")
            mlops.save(cfm, f"results/model/model_latest.pkl")

        print(f"Batch: {i+1}/{len(dataloader)}, loss: {loss.item():.4f}", end="\r")

    epoch_loss /= len(dataloader)
    print(
        f"epoch: {epoch}, step: {step}, time passed: {timer}, loss: {epoch_loss:.4f}",
    )
    cfm.on_epoch_end(epoch_loss)

# HINT using my implementation of the equivariant Readout and a painn model with 8 hidden features I was able to get a loss of 0.037 in 10.000 steps running locally on my laptop. 

# SAMPLING:
# TODO: Make samples from your model: 
cfm = mlops.load("results/cfm_model/model_999.pkl")  # load a trained model 
cfm.eval()

n_samples = 100  # nr samples to generate
base_dataset = data.BaseDistributionDataset(n_samples, basedistribution)  # dataset of basedistribution samples, following the molecular graph of pentene
base_loader = geom.loader.DataLoader(base_dataset, batch_size=512, shuffle=False)

samples = cfm.sample(base_loader, n_steps=100)  # Use the Euler forward method to integrate the learned velocity and generate samples. Output shape: (n_steps, n_samples, n_atoms, 3)
samples = samples.cpu().numpy()

mlops.save(samples, 'results/samples.pkl')  # save samples

# NOTE: you can visualize numpy trajectories by running utils.nglview_pentene(samples) as the last command in a cell, samples must have shape (n_steps, 15, 3)
sample_index = 0  # index of the sample to visualize
util.nglview_pentene(samples[:, sample_index, :, :])  # NOTE: molecule might not display correctly if you use Colab so you might want to use local jupyter for this part

sol_data = mlops.load('results/samples.pkl')  # load generated samples
sol_samples = torch.tensor(sol_data, device=device, dtype=torch.float32)

samples_dataset = data.Pentene1Dataset(sol_samples[-1].cpu().numpy())  # dataset of model samples at t=1, following the molecular graph of pentene

print(f"Number of samples to evaluate: {len(samples_dataset)}")
samples_dataloader = geom.loader.DataLoader(samples_dataset, batch_size=512, shuffle=False)

# Evaluate torsion angles

torsion_evaluator = torsion.TorsionEvaluator()
sampled_torsions = torsion_evaluator.evaluate(sol_data[-1])

ref_torsions = ...  # TODO: compute torsions of reference dataset

# TODO: visualize torsions and compare through some metric of your choice (e.g. KL divergence)

# Evaluate energies with oracle model

oracle = outil.load_oracle(device=device)

oracles = []
for batch in samples_dataloader:
    oracle_energies = oracle.get_energy(batch)
    oracles.append(oracle_energies)

# TODO: compute reference energies for the reference dataset
# HINT: you can use the same oracle to compute reference energies

refs = ...

ref_energies = np.concatenate(refs, axis=0)
oracle_energies = np.concatenate(oracles, axis=0)

# TODO: visualize and compare the distributions of oracle_energies and ref_energies through some metric of your choice (e.g. KL divergence)

# !jupyter nbconvert --version
# !pdflatex --version


# !eval "$(/usr/libexec/path_helper)"

