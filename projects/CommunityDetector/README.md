# Graph Community Detection via Edge Clustering Coefficients

An algorithmic graph mining and community detection system for unweighted, undirected network topologies, partitioning graphs into dense modular subgraphs based on triangular cycles and edge clustering coefficients.

---

## Theoretical Framework

A community in a complex network is defined as a subset of vertices $V_c \subset V$ with dense internal edge connections and relatively sparse connections to external vertices.

```
       [ Community A ]               [ Community B ]
     (Dense Triangles)              (Dense Triangles)
      (•)───(•)    (•)               (•)───(•)    (•)
       │  X  │      │                 │  X  │      │
      (•)───(•)    (•)               (•)───(•)    (•)
              \                       /
               \──[ Inter-Community ]/
                     (Bridge Edge)
```

### Edge Clustering Metric (Radicchi et al.)
The algorithm identifies and iteratively prunes inter-community "bridge" edges by evaluating the edge clustering coefficient $C_{ij}$:

$$C_{ij} = \frac{T_{ij} + 1}{\min(d_i - 1, \, d_j - 1)}$$

where:
* $T_{ij}$ is the number of 3-node triangles in which edge $(i, j)$ participates:

$$T_{ij} = | \{ k \in V \mid (i, k) \in E \land (j, k) \in E \} |$$

* $d_i, d_j$ are the degrees of nodes $i$ and $j$.
* The denominator represents the theoretical maximum number of triangles that could contain edge $(i, j)$.
* Low $C_{ij}$ values indicate bridging edges connecting distinct modular clusters, whereas high values correspond to cohesive intra-community bonds.

---

## Algorithmic Workflow

1. **Initial Metric Calculation:** Computes edge weights $C_{ij}$ across all active edges in $O(1)$ amortized time per lookup using hash-mapped adjacency lists.
2. **Ascending Edge Ordering:** Sorts edges by ascending $C_{ij}$ weight using configurable sorting engines (Merge Sort, Iterative Quick Sort, Insertion Sort, Bubble Sort).
3. **Bridge Elimination:** Removes the edge with the minimal clustering score.
4. **Connectivity & Termination Check:** Executes a breadth-first search (BFS) reachability pass. If removing the edge bisects the graph into disconnected components ($A$ and $B$), the partition is finalized.
5. **Localized Delta Re-evaluation:** Rather than recomputing all network weights, updates only the subset of edges incident to the pruned edge endpoints, decrementing triangle counts by 1.
6. **Iterate:** Repeats from step 2 until the target community division is achieved.

---

## Technical Stack & Complexity

* **Language:** Python 3 (multiprocessing queues, iterative quicksort avoiding recursion limits).
* **Data Structures:** 2D adjacency hash maps (`dict`), priority queues.
* **Input / Output:** CSV edge lists $\rightarrow$ community partition membership files.
