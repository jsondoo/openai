from Agent import Agent

class RandomAgent(Agent):
  def __init__(self, env):
    self.action_space = env.action_space
  
  def get_action(self, observation):
    # RNGesus
    return self.action_space.sample()