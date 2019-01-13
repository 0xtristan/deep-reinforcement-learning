import gym
import numpy as np
import dill

with open('taxi-agent.pkl', 'rb') as infile:
    Q = dill.load(infile)

env = gym.make('Taxi-v2')
state = env.reset()
score = 0
for t in range(200):
    action = np.argmax(Q[state])
    env.render()
    next_state, reward, done, _ = env.step(action)
    #agent.step(state, action, reward, next_state, done)
    score += reward
    state = next_state
    if done:
        break
print('Final score:', score)
env.close()
