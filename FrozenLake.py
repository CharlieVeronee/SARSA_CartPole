import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

#initialise the environment
env = gym.make("FrozenLake-v1", is_slippery = False, render_mode = 'human')

#states = 16, one for each space in grid
#env.observation_space
#action space = l, r, u, d
#env.action_space

#table with all zeroes to house rewards: observation space size x action space size
Q = np.zeros([env.observation_space.n, env.action_space.n])

#learning parameters
alpha = 0.1 #learning rate for the Q function
gamma = 0.95 #discount rate for future rewards
epsilon = 0.1  #exploration rate, can also decay over time
num_episodes = 2000 #number of episodes agent will learn from

#array of reward for each episode
rs = np.zeros([num_episodes])

for i in range(num_episodes):
    #set total reward and time to zero, done to False
    r_sum_i = 0
    t = 0
    done = False
    
    #reset environment and get first new observation
    s, _ = env.reset()
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            #explore: choose a random action
            a = env.action_space.sample()
        else:
            #exploit: choose the action with the highest Q-value
            a = np.argmax(Q[s, :])
        
        #get new state and reward from environment
        s1, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        
        #update Q-Table with new knowledge
        Q[s,a] = (1 - alpha)*Q[s,a] + alpha*(r + gamma*np.max(Q[s1,:])) #Q learning policy
        
        #add reward to episode total
        r_sum_i += r*gamma**t
        
        #update state and time
        s = s1
        t += 1
    rs[i] = r_sum_i


#plot reward vs episodes
r_cumsum = np.cumsum(np.insert(rs, 0, 0)) 
r_cumsum = (r_cumsum[50:] - r_cumsum[:-50]) / 50

#plot
plt.plot(r_cumsum)
plt.show()