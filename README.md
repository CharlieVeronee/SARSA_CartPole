# CartPole Reinforcement Learning Solutions

- 2 reinforcement learning agents to solve CartPole problem

  1. Q Learning (off-policy)
     Q[s, a] ← Q[s, a] + α _ (r + γ _ max(Q[s', a']) - Q[s, a])
  2. SARSA (on-policy) to solve the classic CartPole problem
     Q[s, a] ← Q[s, a] + α _ (r + γ _ Q[s', a'] - Q[s, a])

- Implemented in PyTorch and trained on OpenAI's Gymnasium environments.
  https://gymnasium.farama.org
