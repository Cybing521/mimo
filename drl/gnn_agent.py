import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class GraphAttentionLayer(nn.Module):
    """
    Simple Graph Attention Layer (Multi-Head)
    """
    def __init__(self, in_features, out_features, num_heads=4, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        self.head_dim = out_features // num_heads
        assert self.head_dim * num_heads == out_features, "out_features must be divisible by num_heads"
        
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        
    def forward(self, x, mask=None):
        # x: [batch_size, num_nodes, in_features]
        batch_size, num_nodes, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch_size, num_heads, num_nodes, head_dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        # Attention scores: [batch_size, num_heads, num_nodes, num_nodes]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        
        # Context: [batch_size, num_heads, num_nodes, head_dim]
        context = torch.matmul(attn, V)
        
        # Concatenate heads: [batch_size, num_nodes, out_features]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, self.out_features)
        
        if self.concat:
            return F.elu(context)
        else:
            return context

class GNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, N, M, hidden_dim=64):
        super(GNNActor, self).__init__()
        self.N = N
        self.M = M
        self.num_nodes = N + M
        
        # Global feature dimension (eigenvalues + channel stats + history)
        # 4 + 3 + 5 = 12
        self.global_dim = 12
        
        # Node feature dimension: [x, y, type_tx, type_rx] + global_features
        self.node_in_dim = 2 + 2 + self.global_dim
        
        # GNN Layers
        self.gat1 = GraphAttentionLayer(self.node_in_dim, hidden_dim, num_heads=4)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=4)
        
        # Output layers (predict mean and std for dx, dy)
        self.mean_head = nn.Linear(hidden_dim, 2)
        self.log_std_head = nn.Linear(hidden_dim, 2)
        
    def parse_state(self, state):
        # state: [batch_size, state_dim]
        batch_size = state.size(0)
        
        # Extract global features
        # Structure: [eig(4), feat(3), tx_pos(2N), rx_pos(2M), history(5)]
        global_features = torch.cat([state[:, :7], state[:, -5:]], dim=1) # [batch, 12]
        
        # Extract positions
        pos_start = 7
        tx_pos = state[:, pos_start : pos_start + 2*self.N].view(batch_size, self.N, 2)
        rx_pos = state[:, pos_start + 2*self.N : pos_start + 2*(self.N+self.M)].view(batch_size, self.M, 2)
        
        # Construct node features
        # Tx nodes: [x, y, 1, 0]
        tx_type = torch.tensor([1.0, 0.0], device=state.device).expand(batch_size, self.N, 2)
        tx_nodes = torch.cat([tx_pos, tx_type], dim=2)
        
        # Rx nodes: [x, y, 0, 1]
        rx_type = torch.tensor([0.0, 1.0], device=state.device).expand(batch_size, self.M, 2)
        rx_nodes = torch.cat([rx_pos, rx_type], dim=2)
        
        # All nodes: [batch, N+M, 4]
        nodes = torch.cat([tx_nodes, rx_nodes], dim=1)
        
        # Append global features to each node
        global_expanded = global_features.unsqueeze(1).expand(batch_size, self.num_nodes, self.global_dim)
        nodes = torch.cat([nodes, global_expanded], dim=2)
        
        return nodes

    def forward(self, state):
        x = self.parse_state(state)
        
        x = self.gat1(x)
        x = self.gat2(x)
        
        # Output per node
        mean = self.mean_head(x) # [batch, num_nodes, 2]
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        # Flatten to match action space [tx_x, tx_y, ..., rx_x, rx_y]
        # Current order: node1(tx), node2(tx)... nodeN+1(rx)...
        # We need to flatten carefully to match env expectation
        # Env expects: [tx1_x, tx1_y, ..., rx1_x, rx1_y]
        # Our output is [batch, N+M, 2] -> flatten last two dims -> [batch, 2(N+M)]
        # This matches!
        
        mean = mean.view(state.size(0), -1)
        log_std = log_std.view(state.size(0), -1)
        
        return mean, log_std

class GNNCritic(nn.Module):
    def __init__(self, state_dim, N, M, hidden_dim=64):
        super(GNNCritic, self).__init__()
        self.N = N
        self.M = M
        self.num_nodes = N + M
        self.global_dim = 12
        self.node_in_dim = 2 + 2 + self.global_dim
        
        self.gat1 = GraphAttentionLayer(self.node_in_dim, hidden_dim, num_heads=4)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=4)
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def parse_state(self, state):
        # Same as Actor
        batch_size = state.size(0)
        global_features = torch.cat([state[:, :7], state[:, -5:]], dim=1)
        pos_start = 7
        tx_pos = state[:, pos_start : pos_start + 2*self.N].view(batch_size, self.N, 2)
        rx_pos = state[:, pos_start + 2*self.N : pos_start + 2*(self.N+self.M)].view(batch_size, self.M, 2)
        tx_type = torch.tensor([1.0, 0.0], device=state.device).expand(batch_size, self.N, 2)
        rx_type = torch.tensor([0.0, 1.0], device=state.device).expand(batch_size, self.M, 2)
        nodes = torch.cat([torch.cat([tx_pos, tx_type], dim=2), torch.cat([rx_pos, rx_type], dim=2)], dim=1)
        global_expanded = global_features.unsqueeze(1).expand(batch_size, self.num_nodes, self.global_dim)
        return torch.cat([nodes, global_expanded], dim=2)

    def forward(self, state):
        x = self.parse_state(state)
        x = self.gat1(x)
        x = self.gat2(x)
        
        # Global Pooling (Mean)
        x = torch.mean(x, dim=1) # [batch, hidden_dim]
        
        value = self.value_head(x)
        return value

class GNNPPOAgent:
    def __init__(self, state_dim, action_dim, N, M, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, device='cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.actor = GNNActor(state_dim, action_dim, N, M).to(self.device)
        self.critic = GNNCritic(state_dim, N, M).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        
        self.mse_loss = nn.MSELoss()
        
        # Memory
        self.memory_states = []
        self.memory_actions = []
        self.memory_logprobs = []
        self.memory_rewards = []
        self.memory_is_terminals = []
        
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, log_std = self.actor(state)
            std = log_std.exp()
            
            if deterministic:
                action = torch.tanh(mean)
                return action.cpu().numpy().flatten()
            
            dist = Normal(mean, std)
            action = dist.sample()
            action_tanh = torch.tanh(action)
            
            # Calculate log prob for tanh transform
            log_prob = dist.log_prob(action)
            log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            self.memory_states.append(state)
            self.memory_actions.append(action_tanh)
            self.memory_logprobs.append(log_prob)
            
            return action_tanh.cpu().numpy().flatten()
            
    def store_transition(self, reward, done):
        self.memory_rewards.append(reward)
        self.memory_is_terminals.append(done)
        
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory_rewards), reversed(self.memory_is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.cat(self.memory_states).detach()
        old_actions = torch.cat(self.memory_actions).detach()
        old_logprobs = torch.cat(self.memory_logprobs).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            mean, log_std = self.actor(old_states)
            std = log_std.exp()
            dist = Normal(mean, std)
            
            # We need to reverse tanh to get raw action for log_prob
            # But we stored tanh(action). 
            # Approximation: just use the stored action and treat it as if it came from Normal (ignoring tanh for now or re-sampling)
            # Correct PPO with Tanh is tricky. 
            # Let's use the standard PPO trick: re-calculate log_prob of the *stored* action
            # But stored action is tanh(x). 
            # Let's assume the distribution is over the tanh-ed space directly? No, that's Beta.
            # Let's invert tanh? x = atanh(action).
            # Numerical instability.
            
            # Simplification: For update, we just want log_prob(old_action).
            # Let's use the standard Gaussian PPO and clip actions to [-1, 1] manually in env, 
            # but here we used tanh in select_action.
            # Let's stick to the implementation in `select_action` and try to match it.
            
            # Inverse tanh for log_prob calculation
            # x = 0.5 * log((1+y)/(1-y))
            y = old_actions
            x = 0.5 * torch.log((1+y).clamp(min=1e-6) / (1-y).clamp(min=1e-6))
            
            new_logprobs = dist.log_prob(x)
            new_logprobs -= torch.log(1 - y.pow(2) + 1e-6)
            new_logprobs = new_logprobs.sum(1, keepdim=True)
            
            state_values = self.critic(old_states)
            dist_entropy = dist.entropy().sum(1, keepdim=True)
            
            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            # Surrogate Loss
            advantages = rewards.unsqueeze(1) - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards.unsqueeze(1)) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Clear memory
        self.memory_states = []
        self.memory_actions = []
        self.memory_logprobs = []
        self.memory_rewards = []
        self.memory_is_terminals = []
