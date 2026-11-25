# Feedforward Artificial Neural Network for Digit Recognition

A modular implementation of a multi-layer fully-connected feedforward neural network from scratch in Python, implementing mini-batch Stochastic Gradient Descent (SGD), analytical backpropagation with matrix calculus, and evaluation on the MNIST handwritten digit benchmark.

---

## Network Architecture & Mathematical Formulation

The neural network classifies grayscale handwritten digits ($28 \times 28$ pixels) into 10 target classes ($0-9$).

```
  Input Layer (784 Units) ──► Hidden Layer 1 (16 Units) ──► Hidden Layer 2 (16 Units) ──► Output Layer (10 Units)
       [28x28 Pixels]                [Sigmoid σ]                  [Sigmoid σ]                   [Softmax / Argmax]
```

### 1. Feedforward Layer Operations
For each layer $l \in \{1, 2, \dots, L\}$, the activation vector $a^{(l)}$ is computed via affine transformation followed by the sigmoid non-linearity $\sigma(z)$:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

$$a^{(l)} = \sigma(z^{(l)}) = \frac{1}{1 + e^{-z^{(l)}}}$$

where:
* $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ is the weight matrix connecting layer $l-1$ to layer $l$.
* $b^{(l)} \in \mathbb{R}^{n_l}$ is the bias vector for layer $l$.
* $a^{(0)} = x \in \mathbb{R}^{784}$ is the normalized input image vector ($x_i \in [0, 1]$).

---

## Learning Algorithm: Backpropagation & Mini-Batch SGD

The objective is to minimize the quadratic cost function over the dataset:

$$C = \frac{1}{2n} \sum_{x} \| y(x) - a^{(L)}(x) \|^2$$

### 1. Error Propagation Equations
* **Output Layer Error ($\delta^{(L)}$):**

$$\delta^{(L)} = \nabla_a C \odot \sigma'(z^{(L)}) = (a^{(L)} - y) \odot \sigma(z^{(L)}) \odot (1 - \sigma(z^{(L)}))$$

* **Hidden Layer Error ($\delta^{(l)}$):**

$$\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$

* **Gradients with Respect to Parameters:**

$$\frac{\partial C}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T, \quad \frac{\partial C}{\partial b^{(l)}} = \delta^{(l)}$$

### 2. Mini-Batch Parameter Updates
For a mini-batch $B$ of size $m$ with learning rate $\eta$:

$$W^{(l)} \leftarrow W^{(l)} - \frac{\eta}{m} \sum_{x \in B} \frac{\partial C_x}{\partial W^{(l)}}$$

$$b^{(l)} \leftarrow b^{(l)} - \frac{\eta}{m} \sum_{x \in B} \frac{\partial C_x}{\partial b^{(l)}}$$

---

## Experimental Benchmark

* **Dataset:** MNIST (60,000 training samples, 10,000 test samples).
* **Configuration:** Batch size $m = 50$, learning rate $\eta = 1.0$, 5 epochs.
* **Accuracy:** Reaches $\sim 90\%$ classification accuracy on the held-out test partition.
