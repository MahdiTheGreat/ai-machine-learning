# Project 3 Report Answers & Discussion

## Question 1: Boltzmann Distribution Invariance
**Q:** *The potential u(x) of 'pent-1-ene' is E(3) invariant. What does that imply for the corresponding Boltzmann distribution?*
**Answer:** The Boltzmann distribution is defined as $p(x) = \frac{1}{Z} \exp(-\beta u(x))$. Because the potential energy $u(x)$ is invariant to E(3) transformations (translations and rotations), applying any rotation or translation $R$ to the coordinates does not change the energy: $u(Rx) = u(x)$. Consequently, the probability density remains the same: $p(Rx) = p(x)$. This means the Boltzmann distribution itself is E(3) invariant.

## Question 2: Velocity Field Equivariance vs Target Invariance
**Q:** *Why does our model $v_\theta$ need to be equivariant when our target is invariant?*
**Answer:** The target distribution $p(x)$ is a scalar probability density (which is invariant). However, the model $v_\theta(t, x_t)$ predicts a **velocity field**, which is a collection of vectors detailing how each atom should move in 3D space to transform noise into a molecular conformation. Vectors are inherently directional. If we rotate the entire molecule, the directions the atoms need to travel must rotate by the exact same amount. Thus, the velocity field must be **equivariant** to E(3) transformations, ensuring the generative trajectory is valid regardless of the molecule's spatial orientation.

## Question 3: Importance Sampling Computation
**Q:** *When you have an exact likelihood method, what would you need to compute importance weights $w_i = \exp(-\beta u(x_i)) / p_\theta(x_i)$?*
**Answer:** To compute these weights, we would need to calculate the exact log-likelihood $\log p_\theta(x_i)$ of each generated sample under our learned model. For a continuous normalizing flow (like CFM), computing this requires evaluating the instantaneous change-of-variables formula by integrating the trace of the Jacobian of the velocity field over time. This trace computation is extremely expensive (often requiring Hutchinson's trace estimator) and is bypassed in simulation-free CFM training, which is why we don't compute likelihoods natively here.

## Question 4: Effective Sample Size
**Q:** *What ballpark range would the effective sample size $n_{eff} = (\sum w_i)^2 / \sum (w_i^2)$ be? Compare baseline vs PaiNN.*
**Answer:** The effective sample size measures how closely the generated distribution matches the true Boltzmann distribution. A perfectly matched generative model would have $n_{eff} \approx N$ (where $N$ is the number of samples). In practice, because molecular conformations are highly peaked and complex, the baseline model (which lacks equivariance and structural priors) will propose many invalid or high-energy structures, leading to a few samples dominating the weights and dropping $n_{eff}$ close to 0. The PaiNN model, by leveraging E(3) equivariance and message passing, stays much closer to the true energy manifold. Therefore, PaiNN will have a significantly higher $n_{eff}$ compared to the baseline, indicating much higher sample efficiency.
