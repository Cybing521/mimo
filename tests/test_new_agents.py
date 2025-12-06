import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.sac_agent import SACAgent
from drl.td3_agent import TD3Agent
from drl.ddpg_agent import DDPGAgent

class TestNewAgents(unittest.TestCase):
    def setUp(self):
        self.state_dim = 10
        self.action_dim = 4
        self.batch_size = 32
        self.device = 'cpu'

    def test_sac_agent(self):
        print("\nTesting SAC Agent...")
        agent = SACAgent(self.state_dim, self.action_dim, batch_size=self.batch_size, device=self.device)
        
        # Test action selection
        state = np.random.randn(self.state_dim)
        action = agent.select_action(state)
        self.assertEqual(action.shape, (self.action_dim,))
        
        # Test update (need to fill buffer first)
        for _ in range(self.batch_size + 10):
            s = np.random.randn(self.state_dim)
            a = np.random.randn(self.action_dim)
            r = np.random.randn()
            ns = np.random.randn(self.state_dim)
            d = False
            agent.replay_buffer.add(s, a, r, ns, d)
            
        stats = agent.update()
        self.assertIn('critic_loss', stats)
        self.assertIn('actor_loss', stats)
        print("SAC Agent passed.")

    def test_td3_agent(self):
        print("\nTesting TD3 Agent...")
        agent = TD3Agent(self.state_dim, self.action_dim, batch_size=self.batch_size, device=self.device)
        
        state = np.random.randn(self.state_dim)
        action = agent.select_action(state)
        self.assertEqual(action.shape, (self.action_dim,))
        
        for _ in range(self.batch_size + 10):
            s = np.random.randn(self.state_dim)
            a = np.random.randn(self.action_dim)
            r = np.random.randn()
            ns = np.random.randn(self.state_dim)
            d = False
            agent.replay_buffer.add(s, a, r, ns, d)
            
        stats = agent.update()
        self.assertIn('critic_loss', stats)
        # actor_loss might be 0 if not updated due to delay, but key should exist if we handle it right
        # My implementation returns 0 or previous loss if skipped, so it should be there.
        self.assertIn('actor_loss', stats)
        print("TD3 Agent passed.")

    def test_ddpg_agent(self):
        print("\nTesting DDPG Agent...")
        agent = DDPGAgent(self.state_dim, self.action_dim, batch_size=self.batch_size, device=self.device)
        
        state = np.random.randn(self.state_dim)
        action = agent.select_action(state)
        self.assertEqual(action.shape, (self.action_dim,))
        
        for _ in range(self.batch_size + 10):
            s = np.random.randn(self.state_dim)
            a = np.random.randn(self.action_dim)
            r = np.random.randn()
            ns = np.random.randn(self.state_dim)
            d = False
            agent.replay_buffer.add(s, a, r, ns, d)
            
        stats = agent.update()
        self.assertIn('critic_loss', stats)
        self.assertIn('actor_loss', stats)
        print("DDPG Agent passed.")

if __name__ == '__main__':
    unittest.main()
