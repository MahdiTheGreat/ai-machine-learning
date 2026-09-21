# Project 2 Report Answers & Discussion

## 1. Comparing Models: Baseline vs. E3MPNN

**Force RMSD Discussion:**
The baseline model (lmax=0) only utilizes invariant features (distances between atoms) and scalar node features. It is fundamentally incapable of passing directional vector messages between nodes. While it can fit the potential energy surface reasonably well (since energy is invariant), predicting forces (which are vectors, thus requiring equivariance) using only invariant intermediate representations is highly inefficient. 

The E3MPNN (lmax=1) utilizes equivariant features, allowing vector quantities to be passed as messages during the graph convolution process. This allows the model to naturally respect the 3D geometry and directional dependencies of the atomic forces. Consequently, the E3MPNN should achieve a significantly lower Force RMSD on the validation/test sets compared to the baseline model.

## 2. Equivariance vs. Invariance
*Equivariance* means that if the input is transformed (e.g., rotated), the output transforms in the exact same way. Forces are equivariant: if you rotate the molecule by 90 degrees, the force vectors must also rotate by 90 degrees.
*Invariance* means that if the input is transformed, the output remains completely unchanged. Energy is invariant: rotating a molecule does not change its total potential energy.

By using an equivariant model (E3MPNN), we ensure that the internal representations can track directional information faithfully, leading to much better gradients when we derive forces via autodiff.
