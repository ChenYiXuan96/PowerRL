import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution_with_obs(obs, pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class GridActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -2 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation=nn.Sigmoid)

    def _distribution(self, obs):
        if len(obs.shape) == 1:
            t_Q_C = torch.as_tensor(obs[-int((obs.shape[0])/2):])  # DANGEROUS HERE
        elif len(obs.shape) == 2:
            t_Q_C = torch.as_tensor(obs[:, -int((obs.shape[1])/2):])
        else:
            raise ValueError()

        mu = self.mu_net(obs)
        # debug mode...
        # mu = mu * (t_Q_C != 0)
        std = torch.exp(self.log_std)

        if len(obs.shape) == 2:
            std = torch.repeat_interleave(std[None, :], obs.shape[0], dim=0)

        # debug mode...
        # std = std * (t_Q_C != 0) + 1e-6

        # print('Action_mu: ', mu)
        # print('Action_std: ', std)
        return Normal(mu, std)

    def _log_prob_from_distribution_with_obs(self, obs, pi, act):
        if len(obs.shape) == 1:
            t_Q_C = torch.as_tensor(obs[-int((obs.shape[0])/2):])  # DANGEROUS HERE
        elif len(obs.shape) == 2:
            t_Q_C = torch.as_tensor(obs[:, -int((obs.shape[1])/2):])
        else:
            raise ValueError()
        # print('pi: ', pi)
        # print('act: ', act.shape)
        # print('log_prob', pi.log_prob(act).shape)
        # # print('in log_prob: ')
        # print('t_Q_C: ', t_Q_C.shape)
        # # print(t_Q_C == 0)
        # print(pi.log_prob(act))
        # print(t_Q_C != 0)

        # debug mode...
      # return (pi.log_prob(act) * (t_Q_C != 0)).sum(axis=-1)
        return (pi.log_prob(act)).sum(axis=-1)
        # Last axis sum needed for Torch Normal distribution


# class GridCritic(nn.Module):
#
#     def __init__(self, obs_dim, hidden_sizes, activation):
#         super().__init__()
#         self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
#
#     def forward(self, obs):
#         # print('Forward: ', obs)
#         return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


# Make a new critic network that utilizes OPF results...
class GridCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # print('Forward: ', obs)
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class GridActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        self.pi = GridActor(obs_dim, action_space.shape[0], hidden_sizes, activation)

        # build value function
        self.v = GridCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            # print('-----------------------')
            pi = self.pi._distribution(obs)
            a = pi.sample()
            # print(obs)
            # print(pi)
            # print(a)
            logp_a = self.pi._log_prob_from_distribution_with_obs(obs, pi, a)
            # print(obs)
            v = self.v(obs)
            # print(a)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]