import torch

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'

class TargetDistribution:
    """ Gaussian Mixture Model (GMM) Target Distribution
    """
    def __init__(self, means, variances, weights):
        self.means = means.float()
        self.variances = variances.float()
        self.weights = weights.float() / torch.sum(weights).float()

    def sample(self, num_samples):
        component_indices = torch.multinomial(self.weights, num_samples, replacement=True)
        samples = []
        for idx in component_indices:
            mean = self.means[idx]
            variance = self.variances[idx]
            sample = torch.normal(mean, variance)
            samples.append(sample)
        return torch.stack(samples)

n_samples = 10_000
target_distr = TargetDistribution(
    means=torch.tensor([-4, 2]), 
    variances=torch.tensor([0.5, 2]), 
    weights=torch.tensor([1.0, 1.0])
)
target_samples = target_distr.sample(n_samples)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
bins = torch.linspace(-8, 8, 50)
ax.hist(target_samples.numpy(), bins.numpy(), density=True, alpha=0.6, color='blue')

ax.set_title('Target')

plt.tight_layout()
plt.show()

class BaseDistribution:
    """ Standard Normal Base Distribution
    """
    def sample(self, num_samples, dim):
        return torch.randn(num_samples, dim)

n_samples = 10_000
base_distr = BaseDistribution()
base_samples = base_distr.sample(n_samples, dim=1)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
bins = torch.linspace(-8, 8, 50)
ax.hist(base_samples.numpy(), bins.numpy(), density=True, alpha=0.6, color='red')

ax.set_title('Base')

plt.tight_layout()
plt.show()

class TimeDistriution:
    """ Uniform Time Distribution
    """
    def sample(self, num_samples):
        return torch.rand(num_samples, 1)

time_distr = TimeDistriution()
time_samples = time_distr.sample(n_samples)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
bins = torch.linspace(-0.5, 1.5, 50)
ax.hist(time_samples.numpy(), bins.numpy(), density=True, alpha=0.6, color='green')
ax.set_title('Time')
plt.tight_layout()
plt.show()

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class FlowMatchingModel(torch.nn.Module):
    def __init__(self, base_distr, velocity):
        super(FlowMatchingModel, self).__init__()
        self.base_distr = base_distr
        self.velocity = velocity
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, t, xt):
        in_feats = torch.cat([t, xt], dim=-1)
        return self.velocity(in_feats)
    
    def interpolate(self, x0, x1, t):
        eps = torch.randn_like(x0) * 0.01
        return (1 - t) * x0 + t * x1 + eps
    
    def get_loss(self, x1, t):
        x0 = self.base_distr.sample(x1.shape[0], dim=x1.shape[1])
        xt = self.interpolate(x0, x1, t)

        bt = self.forward(t, xt)
        target_bt = (x1 - x0)
        
        loss = self.loss_fn(bt, target_bt)
        return loss
    
    def sample_euler(self, n_samples, n_steps=10):
        dt = 1.0 / n_steps

        traj = torch.zeros(n_samples, n_steps, 1)
        xt = self.base_distr.sample(n_samples, dim=1)  # initial sample from base distribution

        for step in range(n_steps):
            t = torch.ones_like(xt) * step * dt  # get current time
            bt = self.forward(t, xt)  # query model for velocity
            xt = xt + bt * dt  # Euler update
            traj[:, step, :] = xt  # save 
        return traj

# sample some data from Gaussian Mixture Model target distribution
dataset_size = 100_000
target_data = target_distr.sample(dataset_size)
target_dataset = torch.utils.data.TensorDataset(target_data)
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=512, shuffle=True)

# define model and optimizer
velocity = MLP(input_dim=2, hidden_dim=128, output_dim=1)
model = FlowMatchingModel(base_distr, velocity)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

n_epochs = 30

for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch in target_loader:
        x1 = batch[0].unsqueeze(-1)
        t = time_distr.sample(x1.shape[0])
        
        loss = model.get_loss(x1, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x1.shape[0]
        
    epoch_loss /= dataset_size
    print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f} \r', end='')

# sample from the trained model
n_steps = 100
n_samples = 10_000
trajectory = model.sample_euler(n_samples=n_samples, n_steps=n_steps).detach()

print(f"Trajecotry shape: {trajectory.shape}")

bins = torch.linspace(-8, 8, 50)
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
ax[0].hist(trajectory[:, -1, :].numpy(), bins.numpy(), density=True, alpha=0.6, color='red')
ax[0].set_title('Generated')
ax[1].hist(target_samples.numpy(), bins.numpy(), density=True, alpha=0.6, color='blue')
ax[1].set_title('Target')
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(9, 4), sharey=True, width_ratios=[1, 3, 1])

times = torch.linspace(0, 1, 100)
ax[0].hist(trajectory[:, 0, :].numpy(), bins.numpy(), density=True, alpha=0.6, color='r', orientation='horizontal')
ax[0].set_xlabel(r"$\rho_0(x)$")
ax[0].set_ylabel(r"$x$")

for i in range(1000):
    ax[1].plot(times, trajectory[i, :, 0], color='r', alpha=0.05)

ax[1].set_xlabel(r"$t$")

ax[2].hist(trajectory[:, -1, :].numpy(), bins.numpy(), density=True, alpha=0.6, color='r', orientation='horizontal')
ax[2].set_xlabel(r"$\rho_1(x)$")

plt.tight_layout()
plt.show()
