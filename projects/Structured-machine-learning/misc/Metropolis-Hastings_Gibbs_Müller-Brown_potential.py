import numpy as np
import matplotlib.pyplot as plt

# Müller-Brown parameters
A = np.array([-200, -100, -170, 15])
a = np.array([-1, -1, -6.5, 0.7])
b = np.array([0, 0, 11, 0.6])
c = np.array([-10, -10, -6.5, 0.7])
x_bar = np.array([1, 0, -0.5, -1])
y_bar = np.array([0, 0.5, 1.5, 1])

def mullerm_brown_potential(x, y):
    """Compute Müller-Brown potential at (x, y)"""
    U = 0
    for Ai, ai, bi, ci, xi, yi in zip(A, a, b, c, x_bar, y_bar):
        U += Ai * np.exp(ai*(x-xi)**2 + bi*(x-xi)*(y-yi) + ci*(y-yi)**2)
    return U

def target_pdf(xy):
    x, y = xy
    return np.exp(-mullerm_brown_potential(x, y))

def proposal_sampler(current):
    """Gaussian random walk proposal"""
    return current + np.random.normal(0, 0.1, size=2)

def mh_sampler_2d(target_pdf, initial_state, proposal_sampler, n_samples):
    """
    Generic Metropolis-Hastings sampler.

    Parameters:
        target_pdf: function
            The target probability density function (unnormalized is fine).
        proposal_sampler: function
            Function that takes current state and returns a proposed new state.
        initial_state: float or array-like
            Starting point of the Markov chain.
        n_samples: int
            Number of samples to generate.

    Returns:
        samples: np.ndarray
            Array of generated samples.
    """
    samples = [np.array(initial_state)]
    current = np.array(initial_state)
    
    for _ in range(n_samples-1):
        proposal = proposal_sampler(current)
        acceptance_ratio = target_pdf(proposal)/target_pdf(current)
        if np.random.rand() < min(1, acceptance_ratio):
            current = proposal
        samples.append(current)
        
    return np.array(samples)

def autocorr_lag(x, lag):
    """Compute Pearson correlation coefficient at a given lag"""
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

# Running the sampler
np.random.seed(42)
samples = mh_sampler_2d(target_pdf=target_pdf, initial_state=[-0.5, 1.4], 
                        proposal_sampler=proposal_sampler, n_samples=10000)        
print(samples[:10])  # Show first 10 samples
x_mean, y_mean = samples.mean(axis=0)
x_std, y_std = samples.std(axis=0)
print("Sample mean:", x_mean, y_mean)
print("Sample std:", x_std, y_std)

print("disparsity between actual and sampled mean:",
      np.abs(np.array([-0.46, 1.25]) - np.array([x_mean, y_mean])))

print("disparsity between actual and sampled std:",
      np.abs(np.array([0.38, 0.46]) - np.array([x_std, y_std])))

x_samples = samples[:, 0]  # extract x-coordinate
lags = [1, 10, 100]
autocorr_values = [autocorr_lag(x_samples, lag) for lag in lags]
for lag, val in zip(lags, autocorr_values):
    print(f"Lag {lag}: autocorrelation = {val:.3f}")

# Plotting autocorrelation
plt.figure(figsize=(6,4))
plt.bar([str(l) for l in lags], autocorr_values)
plt.xlabel("Lag j")
plt.ylabel("Pearson correlation")
plt.title("Autocorrelation of x along MH trajectory")
plt.show()

# Plot autocorrelation for a range of lags
max_lag = 200
autocorrs_full = [autocorr_lag(x_samples, lag) for lag in range(1, max_lag+1)]

plt.figure(figsize=(6,4))
plt.plot(range(1, max_lag+1), autocorrs_full, marker='o', linestyle='-')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation of x_samples vs. lag")
plt.tight_layout()
plt.show()

def estimate_ess(x_samples, max_lag=100):
    """Approximate effective sample size (ESS) for a 1D chain."""
    n = len(x_samples)
    sum_rho = 0
    for lag in range(1, max_lag+1):
        ac = autocorr_lag(x_samples, lag)
        if ac <= 0:
            break
        sum_rho += ac
    ess = n / (1 + 2 * sum_rho)
    return ess

ess_x = estimate_ess(x_samples)
print(f"Approximate ESS for x: {ess_x:.1f} out of {len(x_samples)} samples")





