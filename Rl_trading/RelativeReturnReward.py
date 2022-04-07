# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:30:01 2020

@author: CycloneMAO
@contact: 877245759@qq.com
"""


from tensortrade.rewards import RewardScheme

#culmulated wealth within a window; not applicable if the total asset value is changing over time(keep losing or gaining profit)
class RelativeReturnReward(RewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases in net worth."""

    def __init__(self, window_size: int = 1, done_reward: int = -1):
        self._window_size = window_size
        self._step = 0
        self._done_reward = done_reward
        
    @property
    def step(self):
        return self._step
    
    @property
    def done_reward(self):
        return self._done_reward
    
    @done_reward.setter
    def done_reward(self, done_reward):
        self._done_reward = done_reward
    
    @property
    def window_size(self):
        return self._window_size
    
    @window_size.setter
    def window_size(self, window_size):
        self._window_size = window_size

    def get_reward(self, portfolio: 'Portfolio', done: bool) -> float:
        """Rewards the agent for relative return rate(%) in net worth over a sliding window.

        Args:
            portfolio: The portfolio being used by the environment.

        Returns:
            The relative return rate(%) in net worth over the previous `window_size` timesteps. in percentage
        """
        
        self._step += 1
        if self.step % self.window_size == 0:
            start_net_wealth = portfolio.performance['net_worth'].iloc[-self.window_size-1]
            end_net_wealth = portfolio.performance['net_worth'].iloc[-1]
            reward = end_net_wealth / start_net_wealth - 1 if not done else self.done_reward
            self._step -= self.window_size
        else:
            """
            0 is just a placeholder which is not recommended! 
            cannot specify reward value for intermediate steps before collecting a window_size of steps --> need to specify a back-propagation scheme
            eg: in the AlphaGO-Zero paper, they match the final value(1 or -1, win or lose) to the value estimation for each intermediate step to compute the loss(mse),
                Generally, for a value network, assume the window_size is equal to the episode length, 
                specify discount factor to 0 and set the reward for intermediate steps to the final reward(value)
                if the window_size isn't equal to the episode length, need to figure out another propagation scheme for the intermediate rewards
            """
            reward = 0  
        return reward * 100
    
    def reset(self):
        self._step = 0