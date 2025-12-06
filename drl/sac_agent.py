import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from typing import Tuple, Dict

from .utils import ReplayBuffer

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

class SACAgent:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        gamma=0.99, 
        tau=0.005, 
        alpha=0.2, 
        lr=3e-4, 
        hidden_dim=256, 
        batch_size=256,
        device='cpu'
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.critic1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.q_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)
        
        self.training_stats = {
            'critic_loss': [],
            'actor_loss': [],
            'alpha_loss': [],
            'alpha': []
        }

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def update(self, update_steps=1):
        if self.replay_buffer.size < self.batch_size:
            return {}

        stats = {'critic_loss': 0, 'actor_loss': 0, 'alpha_loss': 0, 'alpha': 0}
        
        for _ in range(update_steps):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state)
                target_Q1 = self.critic1_target(next_state, next_action)
                target_Q2 = self.critic2_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
                target_Q = reward + (1 - done) * self.gamma * target_Q

            # Critic update
            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self.q_optimizer.step()

            # Actor update
            new_action, log_prob, _ = self.actor.sample(state)
            q1_new_policy = self.critic1(state, new_action)
            q2_new_policy = self.critic2(state, new_action)
            q_new_policy = torch.min(q1_new_policy, q2_new_policy)
            actor_loss = (self.alpha * log_prob - q_new_policy).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Alpha update
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            # Soft update target networks
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            stats['critic_loss'] += critic_loss.item()
            stats['actor_loss'] += actor_loss.item()
            stats['alpha_loss'] += alpha_loss.item()
            stats['alpha'] += self.alpha.item()

        # Average stats
        for key in stats:
            stats[key] /= update_steps
            self.training_stats[key].append(stats[key])
            
        return stats

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1'])
        self.critic2_target.load_state_dict(checkpoint['critic2'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp()
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
