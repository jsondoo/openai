import gym
import random
import numpy as np

from HillClimbingAgent import HillClimbingAgent
from RandomAgent import RandomAgent

env = gym.make('CartPole-v0')
# https://github.com/openai/gym/wiki/CartPole-v0
# print(env.action_space)

num_episodes = 100
num_steps = 100
agent = RandomAgent(env)

for ep in range(num_episodes):
  observation = env.reset()
  total_reward = 0
  
  while True:
    action = agent.get_action(observation)
    observation, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

    if done: 
      break
      
  agent.update_model(total_reward)
  print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))

env.close()