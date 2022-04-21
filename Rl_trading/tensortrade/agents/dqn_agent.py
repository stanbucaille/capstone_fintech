"""
References:
    - https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
    - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
"""

import uuid
import random
import numpy as np
import tensorflow as tf
import os
import time
from typing import Callable, Tuple
from collections import namedtuple

from tensortrade.agents import Agent, ReplayMemory
from tqdm import tqdm

import matplotlib.pyplot as plt

DQNTransition = namedtuple('DQNTransition', ['state', 'action', 'reward', 'next_state', 'done'])

def max_dropdown(series):
    _acc = series.cumsum().tolist()
    _max = -np.inf
    max_dd = 0

    for _val in _acc:
        _max = max(_max, _val)
        max_dd = max(max_dd, _max - _val)
    return max_dd

class DQNAgent(Agent):

    def __init__(self,
                 env: 'TradingEnvironment',
                 policy_network: None):
        self.env = env
        self.n_actions = len(env.action_space)
        self.observation_shape = env.observation_space.shape

        self.policy_network = policy_network or self._build_policy_network()

        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

        self.id = str(uuid.uuid4())
        self.episode_id = None

    def _build_policy_network(self):
        network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.observation_shape),
            tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=6, kernel_size=3, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_actions, activation="sigmoid"),
            tf.keras.layers.Dense(self.n_actions, activation="softmax")
        ])

        return network

    def restore(self, path: str, **kwargs):
        self.policy_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)
        if not os.path.exists(path):
            os.mkdir(path)

        filename = "__".join(["policy_network", self.id, str(episode)]) + ".h5"

        self.policy_network.save("/".join([path, filename]))

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        threshold: float = kwargs.get('threshold', 0)

        rand = random.random()

        if rand < threshold:
            return np.random.choice(self.n_actions, size=(1,))
        else:
            policy_input = np.expand_dims(state, 0).astype('float32')
            return np.argmax(self.policy_network.predict(policy_input)).reshape(
                -1)  ##获取action时，注意只有state作为输入，但action本身会影响这个网络，MRP转化成MDP

    def _apply_gradient_descent(self, memory: ReplayMemory,
                                batch_size: int,
                                learning_rate: float,
                                discount_factor: float,
                                optimizer,
                                loss):

        transitions = memory.sample(batch_size)
        batch = DQNTransition(*zip(*transitions))

        state_batch = tf.convert_to_tensor(np.array(batch.state))
        action_batch = tf.convert_to_tensor(np.array(batch.action))
        reward_batch = tf.convert_to_tensor(np.array(batch.reward), dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(np.array(batch.next_state))
        done_batch = tf.convert_to_tensor(np.array(batch.done))

        with tf.GradientTape() as tape:
            state_action_values = tf.math.reduce_sum(
                self.policy_network(state_batch) * tf.one_hot(action_batch, self.n_actions),
                axis=1
            )  # use the policy network to compute Q(St, At)

            # the expected_state_action_value can be treated as the true value.
            # But its calculation for terminal state is different from the original paper(DeepMind, Deep Q-learning with Experience Replay)
            next_state_values = tf.where(
                done_batch,
                tf.zeros(batch_size),
                # by default, the next_state_value of done(terminal) state is 0(i.e., the future return at terminal is 0 since there's no further steps).
                tf.math.reduce_max(self.target_network(next_state_batch), axis=1)
                # greedy policy(the paradigm is off-policy since the action we assume it will take at the next step is from any policy other than the the epsilon-policy the agent uses at the current step)
            )  # use the target network to compute Q(St+1, At+1)

            expected_state_action_values = reward_batch + (discount_factor * next_state_values)
            loss_value = loss(expected_state_action_values, state_action_values)  # also called the advantage

        variables = self.policy_network.trainable_variables
        gradients = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    def test(self,
             start_step: int = 0,
             n_steps: int = None):
        state = self.env.reset(stochastic_reset=False)
        self.env.extract_obs(steps=start_step, return_obs=False)
        for i in tqdm(range(n_steps)):
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            if done:
                print("Encounter terminal(done) state!!")
                break
            else:
                state = next_state

        net_worth = self.env.portfolio.performance.net_worth.iloc[start_step:]
        plt.title("Test Result")
        net_worth.plot()
        plt.show()

        ret = - net_worth.diff(-1) / net_worth
        sharpe = np.sqrt(252) * ret.mean() / ret.std()
        mdd = max_dropdown(ret)
        cum_ret = ret.sum()

        print("Annual Sharpe: {}".format(sharpe))
        print("Maximum drop down: {}".format(mdd))
        print("Cumulative return(single-rate): {}".format(cum_ret))
        print("Cumulative return(compound-rate) : {}".format(net_worth.iloc[-1]/net_worth.iloc[0] - 1))

    def train(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              step_verbose: bool = False,
              episode_verbose: bool = True,
              evaluate_every_n_episode: int = None,
              save_episodic_performance: bool = False,
              **kwargs):
        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.98)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 1)
        eps_end: float = kwargs.get('eps_end', 0.1)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 100)
        update_target_every: int = kwargs.get('update_target_every', 1)
        memory_capacity: int = kwargs.get('memory_capacity', 1000)
        ini_steps: int = kwargs.get('ini_steps', memory_capacity)
        max_episode_timesteps: int = kwargs.get('max_episode_timesteps', self.env.max_episode_timesteps)

        assert ini_steps >= batch_size, "initialization steps cannot be less than batch size!!  --Jianmin Mao"
        if max_episode_timesteps is not None and self.env.max_episode_timesteps is not None:
            assert max_episode_timesteps <= self.env.max_episode_timesteps, "max_episode_timesteps cannot exceed the max_episode_timesteps for the environment!! --JianminMao"

        self.memory: ReplayMemory() = kwargs.get('memory', ReplayMemory(memory_capacity, transition_type=DQNTransition))

        if batch_size > self.memory.capacity:
            raise NotImplementedError("batch size is larger than memory's capacity!! --Jianmin Mao")

        episode = 0
        total_steps_done = 0

        # add a dictionary containing performances of each episode
        self.episode_performance = dict()
        self.episode_avg_reward = dict()

        state = self.env.reset()
        steps_done = 0
        print("Collecting memories...")
        for i in tqdm(range(ini_steps)):
            action = self.get_action(state, threshold=1)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.push(state, action[0], reward, next_state, done)
            steps_done += 1
            total_steps_done += 1
            if done or (False if max_episode_timesteps is None else steps_done >= max_episode_timesteps):
                state = self.env.reset()
                steps_done = 0
                continue
            state = next_state

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = tf.keras.losses.MSE
        print("Start training...")
        for i in tqdm(range(n_episodes)):
            self.episode_id = str(uuid.uuid4())
            state = self.env.reset()
            steps_done = 0
            if episode_verbose:
                print('====      EPISODE ID: {}      ===='.format(self.env.episode_id))

            reward_sum = 0
            done = False
            while not done and (True if max_episode_timesteps is None else steps_done < max_episode_timesteps):
                threshold = eps_end + (eps_start - eps_end) * np.exp(-total_steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                if step_verbose:
                    print("Net worth: ", self.env.portfolio.net_worth, "Reward: ", reward)

                self.memory.push(state, action[0], reward, next_state, done)

                state = next_state
                steps_done += 1
                total_steps_done += 1

                self._apply_gradient_descent(self.memory, batch_size, learning_rate, discount_factor, self.optimizer, self.loss)

                if total_steps_done % update_target_every == 0:
                    self.target_network = tf.keras.models.clone_model(self.policy_network)
                    self.target_network.trainable = False

            avg_reward = reward_sum / steps_done
            self.episode_avg_reward[episode] = avg_reward

            if episode_verbose:
                print('episode {} done ({} steps) with average reward: {}'.format(episode, steps_done, avg_reward))

            if save_episodic_performance:
                self.episode_performance[episode] = self.env.portfolio.performance

            episode += 1

            if evaluate_every_n_episode is not None and episode % evaluate_every_n_episode == 0:
                reward_sum = 0
                steps_done = 0
                threshold = 1
                state = self.env.reset(stochastic_reset=False)
                while self.env.feed.has_next():
                    action = self.get_action(state, threshold=threshold)
                    next_state, reward, done, _ = self.env.step(action)
                    reward_sum += reward
                    steps_done += 1
                    state = next_state
                print('Evaluation after {} episodes ({} steps) with average reward: {}'.format(episode, steps_done,
                                                                                               reward_sum / steps_done))
                plt.title("Evaluation result ({} steps) after {} episodes".format(steps_done, episode))
                self.env.portfolio.performance.net_worth.plot()
                plt.show()

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes):
                self.save(save_path, episode=episode)
