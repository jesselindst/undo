#######################################
# Environment
#######################################

import gymnasium as gym

class Environment:
  
  def __init__(self, render_mode="human"):
    self.env = gym.make("LunarLander-v2", render_mode=render_mode)
    if render_mode =="human":
      self.env.render()
    self.observation, self.info = self.env.reset()
    self.terminated = False
    self.truncated = False
    self.total_reward = 0
    self.step = 0
  
  def step(self, action):
    self.observation, reward, self.terminated, self.truncated, self.info = self.env.step(action)
    self.total_reward += reward
  
    
  
#######################################
# Model
#######################################
import torch
import torch.nn as nn

class PPO(nn.Module):
  
  def __init__():
    self.
   
  
  
  
