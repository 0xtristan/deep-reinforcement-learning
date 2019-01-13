from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt
import dill

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

with open('taxi-agent.pkl', 'wb') as outfile:
    dill.dump(agent.Q, outfile, -1)

# plt.figure(figsize=(20, 10))
# plt.plot(avg_rewards)
# plt.show()
