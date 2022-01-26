import torch
import numpy as np
from torch import nn

# 第三回講義資料に概ね則っている


class DynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim, units=(32, 32)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, units[0]),
            nn.ReLU(),
            nn.Linear(units[0], units[1]),
            nn.ReLU(),
            nn.Linear(units[1], output_dim)
        )
        self._loss_fn = nn.MSELoss(reduction="mean")
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, inputs):
        return self.model(inputs)

    def fit(self, inputs, labels):
        predicts = self.predict(inputs)
        loss = self._loss_fn(predicts, labels)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.data.numpy()


class RandomPolicy:
    def __init__(self, max_action, act_dim):
        self._max_action = max_action  # action の最大値
        self._act_dim = act_dim  # action の次元数

    def get_actions(self, batch_size):
        return np.random.uniform(
            low=-self._max_action,
            high=self._max_action,
            size=(batch_size, self._act_dim))


class GaussianActor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return compute_log_probs(self.net(states), self.log_stds, actions)


class Critic(nn.Module):
    def __init__(self, state_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, states):
        return self.net(states)


class PPO:
    def __init__(self,
                 state_shape,
                 action_shape,
                 max_action=1.,
                 device=torch.device('cpu'),
                 seed=0,
                 batch_size=64,
                 lr=3e-4,
                 discount=0.9,
                 horizon=2048,
                 n_epoch=10,
                 clip_eps=0.2,
                 lam=0.95,
                 coef_ent=0.,
                 max_grad_norm=10.):
        fix_seed(seed)

        self.actor = GaussianActor(state_shape, action_shape).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_shape).to(device)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.device = device
        self.batch_size = batch_size
        self.discount = discount
        self.horizon = horizon
        self.n_epoch = n_epoch
        self.clip_eps = clip_eps
        self.lam = lam
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def get_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            if test:
                action = self.actor(state)
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0] * self.max_action

    def get_action_and_val(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, logp = self.actor.sample(state)
            value = self.critic(state)
        return action * self.max_action, logp, value

    def train(self, states, actions, advantages, logp_olds, returns):
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions / self.max_action).float()
        advantages = torch.from_numpy(advantages).float()
        logp_olds = torch.from_numpy(logp_olds).float()
        returns = torch.from_numpy(returns).float()
        self.update_actor(states, actions, logp_olds, advantages)
        self.update_critic(states, returns)

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, logp_olds, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - logp_olds).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def reparameterize(means, log_stds):
    noises = torch.randn_like(means)

    actions = means + noises * log_stds.exp()
    actions = torch.tanh(actions)

    log_pis = calculate_log_pi(log_stds, noises, actions)
    return actions, log_pis


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def compute_log_probs(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


def calculate_log_pi(log_stds, noises, actions):
    return ((-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 *
            np.log(2 * np.pi) * log_stds.size(-1) - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True))
