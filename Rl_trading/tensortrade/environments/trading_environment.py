import gym
import uuid
import logging
import numpy as np

from gym.spaces import Discrete, Space, Box
from typing import Union, Tuple, List, Dict

import tensortrade.actions as actions
import tensortrade.rewards as rewards
import tensortrade.wallets as wallets

from tensortrade.base import TimeIndexed, Clock
from tensortrade.actions import ActionScheme
from tensortrade.rewards import RewardScheme
from tensortrade.data import DataFeed, Select
from tensortrade.data.internal import create_internal_feed
from tensortrade.orders import Broker, Order
from tensortrade.wallets import Portfolio
from tensortrade.environments import ObservationHistory


class TradingEnvironment(gym.Env, TimeIndexed):
    """A trading environments made for use with Gym-compatible reinforcement learning algorithms."""

    agent_id: str = None
    episode_id: str = None

    def __init__(self,
                 portfolio: Union[Portfolio, str],
                 action_scheme: Union[ActionScheme, str],
                 reward_scheme: Union[RewardScheme, str],
                 max_episode_timesteps: int = None,
                 external_feed: DataFeed = None,
                 window_size: int = 1,
                 observe_internal_feed =True,
                 observable_keys: list = [],
                 on_execute_verbose = True,
                 stochastic_reset: bool = True,
                 **kwargs):
        """
        Arguments:
            portfolio: The `Portfolio` of wallets used to submit and execute orders from.
            action_scheme:  The component for transforming an action into an `Order` at each timestep.
            reward_scheme: The component for determining the reward at each timestep.
            external_feed (optional): The pipeline of features to pass the observations through.
            kwargs (optional): Additional arguments for tuning the environments, logging, etc.
        """
        super().__init__()

        self._portfolio = portfolio
        self._action_scheme = action_scheme
        self._reward_scheme = reward_scheme
        self._max_episode_timesteps = max_episode_timesteps
        self._external_feed = external_feed
        self._external_keys = None
        self._window_size = window_size
        self._observe_internal_feed = observe_internal_feed
        self._observable_keys = observable_keys
        self._stochastic_reset = stochastic_reset
        
        self.history = ObservationHistory(window_size=window_size)
        self._on_execute_verbose = on_execute_verbose
        self._broker = Broker(exchanges=self.portfolio.exchanges, on_execute_verbose = self._on_execute_verbose)

        self.clock = Clock()
        self.action_space = None
        self.observation_space = None
        self.viewer = None
        
        self._enable_logger = kwargs.get('enable_logger', False)
        self._observation_dtype = kwargs.get('dtype', np.float32)
        self._observation_lows = kwargs.get('observation_lows', -np.inf)
        self._observation_highs = kwargs.get('observation_highs', np.inf)

        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

        self.compile()
        
    @property
    def max_episode_timesteps(self):
        return self._max_episode_timesteps
    
    @max_episode_timesteps.setter
    def max_episode_timesteps(self, max_episode_timesteps):
        self._max_episode_timesteps = max_episode_timesteps

    @property
    def on_execute_verbose(self):
        return self._on_execute_verbose
    
    @on_execute_verbose.setter
    def on_execute_verbose(self, on_execute_verbose):
        self._on_execute_verbose = on_execute_verbose
    
    @property
    def observe_internal_feed(self):
        return self._observe_internal_feed
    
    @property
    def observable_keys(self):
        return self._observable_keys
    
    @observable_keys.setter
    def observable_keys(self, observable_keys):
        self._observable_keys = observable_keys
        
    @property
    def internal_keys(self):
        return self._internal_keys
        
    @property
    def external_keys(self):
        return self._external_keys

    @property
    def external_feed(self):
        return self._external_feed
    
    @property
    def window_size(self):
        return self._window_size
    
    # non-static --> need to refine its functionality, eg, move next() outside
    def extract_obs(self, steps: int = 1, return_obs: bool = True):

        all_obs = self.feed.next(steps)
        obs = {key: all_obs[key] for key in self.observable_keys}
        if return_obs:
            return obs
    
    def compile(self):
        """
        Sets the observation space and the action space of the environment.
        Creates the internal feed and sets initialization for different components.
        """
        
        self.exchanges = self.portfolio.exchanges
            
        components = [self._broker, self.portfolio, self.action_scheme,
                      self.reward_scheme] + self.exchanges

        for component in components:
            component.clock = self.clock

        self.action_scheme.set_pairs(exchange_pairs=self.portfolio.exchange_pairs)  #will add the dimension of action space
        self.action_space = self.action_scheme.actions

        self._internal_feed = create_internal_feed(self.portfolio)
        self.external_feed.reset()
        self._internal_keys = list(self._internal_feed.next().keys())  
        
        if not self._external_feed is None:
            self.external_feed.reset()
            self._external_keys = list(self.external_feed.next().keys())
            self.feed = self._internal_feed + self.external_feed
        else:
            self.feed = self._internal_feed
        
        if self.max_episode_timesteps is not None:
            assert self.max_episode_timesteps <= self.feed.max_len, "max_episode_timesteps cannot exceed the length of data feed!! --JianminMao"
        
        all_keys = (self.internal_keys if self.observe_internal_feed else []) + (self.external_keys or [])
        if len(self.observable_keys) == 0:
            self._observable_keys = all_keys
        else:
            self._observable_keys = [key for key in all_keys if key in self.observable_keys]
            
        sample_obs = self.extract_obs()
        n_features = len(sample_obs)

        self.observation_space = Box(
            low=self._observation_lows,
            high=self._observation_highs,
            shape=(self.window_size, n_features),
            dtype=self._observation_dtype
        )
        
        self.feed.reset()
    
    @property
    def portfolio(self) -> Portfolio:
        """The portfolio of instruments currently held on this exchange."""
        return self._portfolio

    @portfolio.setter  # cannot exist without the property's definition
    def portfolio(self, portfolio: Union[Portfolio, str]):
        self._portfolio = wallets.get(portfolio) if isinstance(portfolio, str) else portfolio

    @property
    def broker(self) -> Broker:
        """The broker used to execute orders within the environment."""
        return self._broker

    @property
    def episode_trades(self) -> Dict[str, 'Trade']:
        """A dictionary of trades made this episode, organized by order id."""
        return self._broker.trades

    @property
    def action_scheme(self) -> ActionScheme:
        """The component for transforming an action into an `Order` at each time step."""
        return self._action_scheme

    @action_scheme.setter
    def action_scheme(self, action_scheme: Union[ActionScheme, str]):
        self._action_scheme = actions.get(action_scheme) if isinstance(
            action_scheme, str) else action_scheme

    @property
    def reward_scheme(self) -> RewardScheme:
        """The component for determining the reward at each time step."""
        return self._reward_scheme

    @reward_scheme.setter
    def reward_scheme(self, reward_scheme: Union[RewardScheme, str]):
        self._reward_scheme = rewards.get(reward_scheme) if isinstance(
            reward_scheme, str) else reward_scheme
    
    @property
    def stochastic_reset(self):
        return self._stochastic_reset

    @stochastic_reset.setter
    def stochastic_reset(self, x: bool):
        self._stochastic_reset = x
        
    def initialize_history(self):
        for _ in range(self.window_size):
            obs_row = self.extract_obs()
            self.history.push(obs_row)

    def is_done(self):
        return False # (self.portfolio.profit_loss < 0.95) # or not self.feed.has_next()  # meet the end of the feed is not a kind of terminal, especially for trading activities which have no sense of ending unless exiting subjectively
    
    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        """Run one timestep within the environments based on the specified action.

        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environments's exchange, often OHLCV or tick trade history data points.
            reward (float): An size corresponding to the benefit earned by the action taken this timestep.
            done (bool): If `True`, the environments is complete and should be restarted.
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """
        orders = self.action_scheme.get_order(action, self.portfolio)
        for order in orders:
            if order is not None:
                self._broker.submit(order)
        self._broker.update()  #execution of the order
            
        obs_row = self.extract_obs()
        self.history.push(obs_row)
        obs = self.history.observe()

        done = self.is_done()
        
        reward = self.reward_scheme.get_reward(self.portfolio, done)
        
        if (reward is None) or np.bitwise_not(np.isfinite(reward)).any():
            raise ValueError('Reward returned by the reward scheme must by a finite float.')

        info = {}
        if self._enable_logger:
            self.logger.debug('Order:       {}'.format(order))
            self.logger.debug('Observation: {}'.format(obs))
            self.logger.debug('P/L:         {}'.format(self._portfolio.profit_loss))
            self.logger.debug('Reward ({}): {}'.format(self.clock.step, reward))
            self.logger.debug('Performance: {}'.format(self._portfolio.performance.tail(1)))

        self.clock.increment()

        return obs, reward, done, info

    def reset(self, stochastic_reset: bool = None) -> np.array:
        """Resets the state of the environments and returns an initial observation.

        Returns:
            The episode's initial observation.
        """
        self.episode_id = uuid.uuid4()

        self.clock.reset()  #重置计步器
        self.feed.reset()   #重置数据流，默认的internal_feed里面包含了exchange的Stream，所以这一步会重置exchange里面的Stream
        self.action_scheme.reset()  #重置action_scheme
        self.reward_scheme.reset()  #重置reward_shceme
        self.portfolio.reset()   #重置投资组合（钱包），但这一步不会重置钱包绑定的交易所
        self.history.reset()  #重置数据历史
        self._broker.reset()  #重置broker（交易记录，未完成订单，已完成订单等）

        if stochastic_reset is None:
            stochastic_reset = self.stochastic_reset
        if stochastic_reset:
            random_start = np.random.randint(0, self.feed.max_len-self.window_size-self.max_episode_timesteps)
            if random_start > 0:
                self.extract_obs(steps=random_start, return_obs=False)
        
        self.portfolio.reset()
        
        self.initialize_history()
        obs = self.history.observe()
        
        self.clock.increment()
        
        return obs

    def render(self, mode='none'):
        """Renders the environment via matplotlib."""
        if mode == 'log':
            self.logger.info('Performance: ' + str(self._portfolio.performance))
        elif mode == 'chart':
            if self.viewer is None:
                raise NotImplementedError()

            self.viewer.render(self.clock.step - 1,
                               self._portfolio.performance,
                               self._broker.trades)

    def close(self):
        """Utility method to clean environment before closing."""
        if self.viewer is not None:
            self.viewer.close()
