import numpy as np
from Agent import Agent

class HillClimbingAgent(Agent):
  def __init__(self, env):
    self.ob_dim = env.observation_space.shape
    self.action_size = env.action_space.n
    self.build_model()
      
  def build_model(self):
    self.weights = 1e-4*np.random.rand(*self.ob_dim, self.action_size)
    self.best_reward = -np.Inf
    self.best_weights = np.copy(self.weights)
    self.noise_scale = 1e-2

  def get_action(self, observation):
    # take observation (1x4) multiply w/ weights (4x2) to get an output (1x2)
    p = np.dot(observation, self.weights)
    action = np.argmax(p)
    return action

  # called at the end of every episode
  def update_model(self, reward):
    if reward >= self.best_reward:
      self.best_reward = reward
      self.best_weights = np.copy(self.weights)
      self.noise_scale = max(self.noise_scale/2, 1e-3)  # play around with these values
    else:
      self.noise_scale = min(self.noise_scale*2, 1) 

    self.weights = self.best_weights + self.noise_scale * np.random.rand(*self.ob_dim, self.action_size)
