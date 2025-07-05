from collections import deque
import random
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 2)

    def forward(self, x):        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x        
    

class DQNCartPoleSolver:
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v1', render_mode="human")
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.dqn = DQN()
        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.dqn.parameters(), lr=0.01)

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return torch.tensor(np.reshape(state, [1, 4]), dtype=torch.float32) 
    
    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            return self.env.action_space.sample() 
        else:
            with torch.no_grad():
                return torch.argmax(self.dqn(state)).item()

    def remember(self, state, action, reward, next_state, next_action, done):
        reward = torch.tensor(reward)
        self.memory.append((state, action, reward, next_state, next_action, done))
    
    def replay(self, minibatch):
        states = torch.cat([s for s, _, _, _, _, _ in minibatch])
        actions = torch.tensor([a for _, a, _, _, _, _ in minibatch], dtype=torch.long)
        rewards = torch.stack([r for _, _, r, _, _, _ in minibatch])
        next_states = torch.cat([s1 for _, _, _, s1, _, _ in minibatch])
        next_actions = torch.tensor([a1 for _, _, _, _, a1, _ in minibatch], dtype=torch.long)
        dones = torch.tensor([d for _, _, _, _, _, d in minibatch], dtype=torch.bool)

        # Current Q-values
        q_values = self.dqn(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Next Q-values (SARSA-style)
        next_q_values = self.dqn(next_states)
        next_q_selected = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
        next_q_selected[dones] = 0.0  # zero out where episode ended

        # Compute target
        q_targets = rewards + self.gamma * next_q_selected

        # Optimize
        self.opt.zero_grad()
        loss = self.criterion(q_selected, q_targets)
        loss.backward()
        self.opt.step()

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            obs, _ = self.env.reset()
            state = self.preprocess_state(obs)
            done = False
            i = 0
            while not done:
                if e % 100 == 0 and not self.quiet:
                    self.env.render()
                action = self.choose_action(state, self.get_epsilon(e))
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocess_state(next_obs)
                next_action = self.choose_action(next_state, self.get_epsilon(e))
    
                self.remember(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            if len(self.memory) >= self.batch_size:
                minibatch = random.sample(self.memory, self.batch_size)
                self.replay(minibatch)
        
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
    agent.env.close()
