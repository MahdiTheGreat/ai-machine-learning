# Geometric Deep Learning & Generative Modeling for Molecular Physics

A research-grade codebase investigating symmetry-preserving geometric deep learning, $E(3)$-equivariant neural networks, and continuous generative flows applied to molecular chemistry and statistical physics.

---

## Overview

This repository explores the intersection of **group theory, differential geometry, and machine learning** to solve foundational challenges in computational chemistry:
1. **Topological Representation Learning:** Benchmarking Graph Neural Networks (GCN, GAT, and DeepSets) for thermodynamic and physiological property prediction under scaffold-stratified domain shifts.
2. **Equivariant Potential Energy & Force Surfaces:** Constructing $E(3)$-equivariant message-passing neural networks (`e3nn`) to approximate coupled-cluster quantum chemistry calculations and predict conservative force vector fields.
3. **Generative Continuous Normalizing Flows:** Formulating Continuous Normalizing Flows (CNF) via **Conditional Flow Matching (CFM)** to generate independent and identically distributed (i.i.d.) molecular conformations according to the Boltzmann distribution.
4. **Classical Markov Chain Monte Carlo (MCMC):** Simulating transition pathways on complex potential energy surfaces (Müller-Brown) using Metropolis-Hastings and Gibbs sampling.

---

## Research Pillars & Architecture

### 1. Topological Graph Learning for Molecular Properties
*Directory: [`project 1/`](project%201/)*

Predicting thermodynamic and physiological observables directly from 2D molecular graphs by learning permutation-invariant representations over atomic features and bond connectivity.

```
Molecular Graph  ──>  [Atom Embedding & Normalization]
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
           [DeepSets]        [GCN]           [GAT]
           (No Graph)    (Fixed Degree)  (Attention)
               │               │               │
               └───────────────┼───────────────┘
                               ▼
               [Global Mean / Sum Pooling]
                               │
                               ▼
               [Prediction Head: ESOL / BBBP]
```

* **Target Benchmarks:**
  * **ESOL (Delaney):** Aqueous solubility regression ($\log S$ in $\text{mol/L}$), measuring thermodynamic hydration propensity.
  * **BBBP (Blood-Brain Barrier Penetration):** Binary classification for central nervous system bioavailability.
* **Stratification:** Splitting via **Bemis-Murcko scaffolds** to evaluate realistic out-of-distribution generalization rather than random partitioning.
* **Architectures Compared:**
  * **DeepSet Baseline:** Permutation-invariant set aggregation ignoring graph edges ($\sum \text{MLP}(h_v)$).
  * **Graph Convolutional Networks (GCN):** Degree-normalized neighborhood aggregation (Kipf & Welling).
  * **Graph Attention Networks (GAT):** Multi-head anisotropic attention weighing non-covalent and neighborhood importance (Dwivedi & Bresson).

#### Empirical Benchmark Results (3 Replicates, Mean $\pm$ Std)

| Architecture | ESOL Test PCC ($\uparrow$) | ESOL Test RMSE ($\downarrow$) | BBBP Test ROC-AUC ($\uparrow$) |
|:---|:---:|:---:|:---:|
| **DeepSets (Baseline)** | $0.7747 \pm 0.0165$ | $1.3416 \pm 0.0467$ | $0.6865 \pm 0.0083$ |
| **GCN** | $0.7571 \pm 0.0080$ | $1.4914 \pm 0.0789$ | $0.7077 \pm 0.0210$ |
| **GAT (Best)** | $\mathbf{0.8033 \pm 0.0165}$ | $\mathbf{1.2829 \pm 0.0381}$ | $\mathbf{0.7289 \pm 0.0191}$ |

* **Key Finding:** GAT outperforms GCN and DeepSets across both regression and classification. On ESOL, the DeepSets baseline surprisingly beats GCN, revealing that simple global atomic composition provides strong baseline solubility information, whereas isotropic GCN convolutions dilute feature variance across small molecular rings.

---

### 2. $E(3)$-Equivariant Machine-Learned Force Fields (MLFF)
*Directory: [`project 2/`](project%202/)*

Predicting potential energy surfaces $u(\mathbf{x}): \mathbb{R}^{3N} \to \mathbb{R}$ from 3D atomic coordinates while strictly preserving rotational and translational symmetries ($G = E(3)$).

```
Atomic Point Cloud (x, z) ──> Complete Relational Graph
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
   Radial Components                                    Angular Components
[Soft Cosine Distance Basis]                           [Spherical Harmonics Y_l^m]
           │                                                     │
           └──────────────────────────┬──────────────────────────┘
                                      ▼
                   [Clebsch-Gordan Tensor Product Convolutions]
                                      │
                   [Equivariant Message Passing (e3nn)]
                                      │
                   [Global Invariant Sum: Potential Energy u(x)]
                                      │
                   [torch.autograd: F = -∇_x u(x)]
                                      ▼
                      Conservative Force Vector Field
```

* **Dataset:** MD17 molecular dynamics trajectories for **Ethanol** ($C_2H_6O$) computed at the gold-standard **CCSD(T)** level of theory (Coupled-Cluster Singles, Doubles, and Perturbative Triples).
* **Geometric Formalism:**
  * Directional vectors decomposed into **Spherical Harmonics** basis functions ($Y_l^m$).
  * Interatomic scalar distances projected onto a smooth **cosine radial basis** (`soft_one_hot_linspace`).
  * Convolutions parameterized via **Clebsch-Gordan tensor products** and gated nonlinearities.
  * System comparisons: Invariant scalar network ($l_{\text{max}} = 0$) versus higher-order geometric tensor features ($l_{\text{max}} > 0$).
* **Conservative Force Derivation:** Enforcing physical energy conservation by computing atomic force vectors analytically through auto-differentiation:
  $$\mathbf{F}_i = -\nabla_{\mathbf{x}_i} u_\theta(\mathbf{x})$$
* **Validation:** Evaluated across sample efficiency splits ($N=900$ vs. $N=450$) and validated via force Root Mean Squared Distance (RMSD) and Langevin molecular dynamics trajectories.

---

### 3. Continuous Normalizing Flows for Boltzmann Sampling
*Directory: [`project 3/`](project%203/)*

Sampling non-normalized Boltzmann distributions $p(\mathbf{x}) \propto \exp(-\beta u(\mathbf{x}))$ without the correlation penalties and autocorrelation times inherent to Markov Chain Monte Carlo (MCMC).

```
Base Prior p_0(x) ──[Numerical ODE Integration: dx/dt = v_θ(t, x)]──> Target Sample ρ_θ(x)
       ▲                                                                      │
       │                                                                      ▼
  Isotropic Gaussian                                               [Oracle Energy Validation]
  (Zero Center-of-Mass)                                            [Torsion Angle KDE]
                                                                   [KL Divergence & ESS]
```

* **Methodology: Conditional Flow Matching (CFM):**
  * Learns a time-dependent vector field $v_\theta(t, \mathbf{x}_t)$ governing an ordinary differential equation (ODE):
    $$\frac{d\mathbf{x}_t}{dt} = v_\theta(t, \mathbf{x}_t), \quad \mathbf{x}_0 \sim p_0(\mathbf{x}), \quad \mathbf{x}_1 \sim \rho_\theta(\mathbf{x})$$
  * Regresses directly against the conditional velocity $v(t, \mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}) = \mathbf{x} - \mathbf{x}_0$ via the mean-squared error objective:
    $$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}, t}\left[ \| v_\theta(t, \mathbf{x}_t) - (\mathbf{x} - \mathbf{x}_0) \|^2 \right]$$
* **Target System:** Small organic hydrocarbon **pent-1-ene** ($C_5H_{10}$) governed by the SMIRNOFF force-field potential energy surface.
* **Architectures Evaluated:**
  * **Unconstrained Baseline:** Multi-layer perceptron regressing Euclidean velocity vectors directly.
  * **Equivariant PaiNN Flow:** Polarizable Atom Interaction Neural Network incorporating equivariant directional features and scalar invariant interactions to predict an $E(3)$-equivariant velocity field.

#### Distributional Fidelity & Generative Performance (100 Generated Molecules)

| Metric | Baseline MLP | PaiNN Equivariant Flow | Relative Improvement |
|:---|:---:|:---:|:---:|
| **Histogram KL (Torsion)** | $0.9765$ | $\mathbf{0.5173}$ | $\mathbf{47.0\% \text{ reduction}}$ |
| **Histogram KL (Energy)** | $14.4824$ | $\mathbf{11.4145}$ | $\mathbf{21.2\% \text{ reduction}}$ |
| **KDE KL (Torsion)** | $0.2411$ | $\mathbf{0.1077}$ | $\mathbf{55.3\% \text{ reduction}}$ |
| **KDE KL (Energy)** | $23.6749$ | $\mathbf{2.6971}$ | $\mathbf{88.6\% \text{ reduction}}$ |

* **Effective Sample Size (ESS) & Statistical Mechanics Impact:**
  * **Baseline Collapse ($n_{\text{eff}} \approx 1$ out of $N$):** Lacking geometric equivariance, standard MLPs frequently place atoms into sterically clashing configurations ($r_{ij} \to 0$), causing energy explosions ($\exp(-\beta u(\mathbf{x})) \to 0$) and causing importance weights to degenerate.
  * **PaiNN Flow ($n_{\text{eff}} \approx 50\%\text{--}80\%$ of $N$):** Preserving spatial symmetries enforces physically plausible conformers, maintaining uniform importance weights $w_i = \frac{\exp(-\beta u(\mathbf{x}_i))}{\rho_\theta(\mathbf{x}_i)}$ and enabling low-variance expectation estimates for macroscopic observables.

---

### 4. Classical Sampling on Rugged Energy Landscapes
*Directory: [`misc/`](misc/)*

Exploration of classical statistical physics sampling algorithms over the 2D **Müller-Brown potential**, a standard benchmark featuring three local minima separated by high transition energy barriers:
* **Metropolis-Hastings (Random Walk):** Simulates thermal barrier crossings and quantifies step-size vs. acceptance rate tradeoffs.
* **Gibbs Sampling:** Alternating conditional density updates demonstrating coordinate-axis alignment limitations in narrow valleys.

---

## Technical Stack

* **Deep Learning Frameworks:** PyTorch (`torch`), PyTorch Geometric (`torch_geometric`).
* **Equivariant & Geometric Neural Networks:** `e3nn` (Euclidean neural networks), PaiNN equivariant convolutions.
* **Cheminformatics & Physics:** RDKit, MoleculeNet, MD17 (CCSD(T)), SMIRNOFF Force Field.
* **Numerical & Statistical Analysis:** SciPy, NumPy, Torcheval, Matplotlib, Weights & Biases (`wandb`).

---

## Repository Structure

```text
Structured-machine-learning/
├── README.md
├── project 1/
│   └── Structured_ML_Project_Notebook.ipynb   # GCN vs GAT vs DeepSet on ESOL & BBBP
├── project 2/
│   └── sml_project_2.ipynb                    # e3nn Equivariant Force Fields on MD17
├── project 3/
│   └── sml_project_3.ipynb                    # Flow Matching & Continuous Normalizing Flows
└── misc/
    ├── Metropolis-Hastings_Gibbs_Müller-Brown_potential.ipynb
    └── reg_regression.py
```

---

## References

1. **GAT:** Dwivedi, V. P., & Bresson, X. *A generalization of transformer networks to graphs* (2020).
2. **GCN:** Kipf, T. N., & Welling, M. *Semi-supervised classification with graph convolutional networks* (2016).
3. **MoleculeNet:** Wu, Z. et al. *MoleculeNet: A benchmark for molecular machine learning*. Chemical Science (2018).
4. **e3nn:** Geiger, M., & Smidt, T. *e3nn: Euclidean neural networks* (2022).
5. **MD17:** Chmiela, S. et al. *Towards exact molecular dynamics simulations with machine-learned force fields*. Nature Communications (2018).
6. **PaiNN:** Schütt, K. T. et al. *Equivariant message passing for the prediction of tensorial properties and molecular spectra* (ICML 2021).
7. **Equivariant Normalizing Flows:** Köhler, J. et al. *Equivariant flows on spheres and manifolds* (ICML 2020).
8. **Flow Matching:** Lipman, Y. et al. *Flow matching for generative modeling* (ICLR 2023).
