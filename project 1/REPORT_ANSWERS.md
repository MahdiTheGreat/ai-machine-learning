# Project 1 Report Answers & Discussion

## Model Comparison

Below is the comparison of the different models on the ESOL (Regression) and BBBP (Classification) tasks:

1. **DeepSet (Baseline)**: This model applies a per-node MLP and then pools the results globally. It does not use the graph structure (edges) at all. It serves as a baseline to see how much can be learned purely from atomic properties without structural context.
2. **GCN (Graph Convolutional Network)**: GCN aggregates features from immediate neighbors using an isotropic weighting scheme based on node degrees. It can capture local structural motifs but struggles to distinguish between neighbors of different importance.
3. **GAT (Graph Attention Network)**: GAT uses multi-head attention to learn dynamic weights for different neighbors. It can attend to specific chemical bonds or substructures that are most relevant to the target property.

**Discussion**:
GCN and GAT are expected to outperform the DeepSet baseline because molecular properties (like solubility or blood-brain barrier permeability) heavily depend on the structural arrangement of atoms (e.g., rings, functional groups), not just the sheer presence of certain atoms. Between GCN and GAT, GAT often achieves better performance because its attention mechanism allows it to focus on critical structural features rather than treating all bonds equally.
