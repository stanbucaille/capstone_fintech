# -*- coding: utf-8 -*-
"""
Created on Wed May 12 03:01:38 2021

@author: CycloneMAO
@contact: 877245759@qq.com
"""
import sys

import gym
import uuid
import logging
import numpy as np

from gym.spaces import Discrete, Space, Box
from typing import Union, Tuple, List, Dict



from tensortrade.environments import TradingEnvironment

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'continuous-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
        
class continuous_v0(TradingEnvironment):
    def __init__(self, 
                 stochastic_reset: bool = True,
                 max_episode_timesteps: int = None,
                 is_action_continuous = True,
                 **kwargs):
        
        self._stochastic_reset = stochastic_reset
        
        self._is_action_continuous = is_action_continuous
        super().__init__(**kwargs)
        if max_episode_timesteps is not None:
            assert max_episode_timesteps <= self.feed.max_len, "max_episode_timesteps cannot exceed the length of data feed!! --JianminMao"
        self._max_episode_timesteps = max_episode_timesteps
        self.action_space = self.action_space[-1][-1]
    
        
    @property
    def is_done(self):
        return (self.portfolio.profit_loss < 0.80) # or not self.feed.has_next()  # meet the end of the feed is not a kind of terminal, especially for trading activities which have no sense of ending unless exiting subjectively