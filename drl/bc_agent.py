import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class BCAgent:
    """
    Behavior Cloning Agent
    Pre-trains an Actor network using expert demonstrations.
    """
    def __init__(self, actor_network, device='cpu', lr=1e-3):
        self.actor = actor_network
        self.device = device
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def train(self, dataset, batch_size=64, epochs=100):
        """
        Train the actor using the dataset.
        
        Args:
            dataset: List of (state, action) tuples
            batch_size: Batch size
            epochs: Number of epochs
        """
        states, actions = zip(*dataset)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        
        train_data = TensorDataset(states, actions)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        print(f"Starting Behavior Cloning for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_actions in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                # Actor outputs action directly (deterministic)
                # Note: This assumes the actor has a method to output deterministic action
                # For GaussianPolicy (SAC), we usually take the mean.
                # For PPO, we take the mean.
                # We need to check how the actor is implemented.
                # Assuming actor(state) returns action or distribution.
                
                # Let's inspect how SAC/PPO actors are called.
                # SAC: action, _, _ = policy.sample(state) -> returns (action, log_prob, mean)
                # We want to match the MEAN to the expert action.
                
                if hasattr(self.actor, 'sample'):
                    _, _, pred_action = self.actor.sample(batch_states)
                elif hasattr(self.actor, 'forward'):
                    # PPO Actor returns dist. We want dist.mean
                    dist = self.actor(batch_states)
                    if hasattr(dist, 'mean'):
                        pred_action = dist.mean
                    else:
                        # Deterministic actor (TD3/DDPG)
                        pred_action = dist
                else:
                    pred_action = self.actor(batch_states)
                
                loss = self.criterion(pred_action, batch_actions)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")
                
        print("Behavior Cloning Complete.")
