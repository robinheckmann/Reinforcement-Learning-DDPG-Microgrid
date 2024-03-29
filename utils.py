import torch
from vars import *
from collections import namedtuple
import random
from environment import EMS
import numpy as np
import copy
from collections import deque


# Taken from
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
import os
import pickle as pkl

class Normalizer():
    """
    Normalizes the input data by computing an online variance and mean
    """
    def __init__(self, num_inputs):
      
        self.n = torch.zeros(num_inputs).to(device)
        self.mean = torch.zeros(num_inputs).to(device)
        self.mean_diff = torch.zeros(num_inputs).to(device)
        self.var = torch.zeros(num_inputs).to(device)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()  
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var).to(device)
        return (inputs - self.mean)/obs_std

'''
'''

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class ReplayMemory(object):
    """
    This class serves as storage capability for the replay memory. It stores the Transition tuple
    (state, action, next_state, reward) that can later be used by a DQN agent for learning based on experience replay.

    :param capacity: The size of the replay memory
    :type capacity: Integer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition. (the transition tuple)"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        #print(len(self.memory))

    def sample(self, batch_size):
        """
        Randomly selects batch_size elements from the memory.

        :param batch_size: The wanted batch size
        :type batch_size: Integer
        :return:
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


    
class OrnsteinUhlenbeckActionNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.1, sigma=.5, sigma_min = 0.05, sigma_decay=.99):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
