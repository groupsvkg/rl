from replay_memory import ReplayMemory
from actor_network import Actor
from critic_network import Critic
from copy import deepcopy
import gym
import numpy as np
import torch
from torch.optim import Adam
from gym import wrappers
from time import time


class MountainCar:
    # def __init__(self, env, test_env, episodes=400000, max_episode_length=1000,
    #              replay_memory_capacity=int(1e6), replay_minibatch_size=64,
    #              policy_learning_rate=1e-4, q_learning_rate=1e-3, actor_noise=0.1,
    #              discount_factor=0.99, soft_target_update_factor=0.001,
    #              update_start=1000, update_interval=1, test_start=4000, test_interval=2000,
    #              test_episodes=1, action_start=10000):
    def __init__(self, env, test_env, episodes=400000, max_episode_length=1000,
                 replay_memory_capacity=int(1e6), replay_minibatch_size=64,
                 policy_learning_rate=0.001, q_learning_rate=0.001, actor_noise=0.2,
                 discount_factor=0.99, soft_target_update_factor=0.001,
                 update_start=1, update_interval=1, test_start=2000, test_interval=4000,
                 test_episodes=5, action_start=500):
        self.env = env
        self.test_env = test_env
        self.episodes = episodes
        self.max_episode_length = max_episode_length
        self.replay_memory_capacity = replay_memory_capacity
        self.replay_minibatch_size = replay_minibatch_size
        self.actor_learning_rate = policy_learning_rate
        self.critic_learning_rate = q_learning_rate
        self.actor_noise = actor_noise
        self.discount_factor = discount_factor
        self.soft_target_update_factor = soft_target_update_factor
        self.update_start = update_start
        self.update_interval = update_interval

        self.test_start = test_start
        self.test_interval = test_interval
        self.test_episodes = test_episodes

        self.action_start = action_start

        # Actor/Critic
        self.actor = Actor(self.env)
        self.critic = Critic(self.env)

        # Target Actor/Critic
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        # Action Range
        self.max_action_value = int(self.env.action_space.high[0])
        self.min_action_value = int(self.env.action_space.low[0])

    def apply_ddgp(self):
        torch.manual_seed(0)
        np.random.seed(0)

        # Initialise replay memory D to capacity N
        replay_memory = ReplayMemory(env=self.env, replay_memory_capacity=self.replay_memory_capacity,
                                     replay_minibatch_size=self.replay_minibatch_size)

        # Freeze target network
        self.target_actor.freeze()
        self.target_critic.freeze()

        # Optimizers
        self.policy_optimizer = Adam(self.actor.policy.parameters(),
                                     lr=self.actor_learning_rate)
        self.q_optimizer = Adam(self.critic.q.parameters(),
                                lr=self.critic_learning_rate)

        # Interact with Environment
        current_state = self.env.reset()
        episode_return = 0
        episode_length = 0

        for step in range(self.episodes):
            if step > self.action_start:
                action = self.actor.action(torch.as_tensor(
                    current_state, dtype=torch.float32)) + \
                    self.actor_noise * \
                    np.random.uniform()
                # np.random.randn(self.max_action_value)
                action = np.clip(action, self.min_action_value,
                                 self.max_action_value)
            else:
                action = self.env.action_space.sample()

            # Execute action $a_t$ and observe reward $r_t$ , next state $s_{t+1}$
            next_state, reward, done, info = self.env.step(action)
            episode_return += reward
            episode_length += 1

            # Continue if episode length reached maximum allowed length.
            done = False if episode_length == self.max_episode_length else done

            # self.env.render()

            # Store transition $(s_t , a_t , r_t , s_{t+1} )$ in Ds
            replay_memory.store(current_state, action,
                                reward, next_state, done)

            # Update current state
            current_state = next_state

            # Reset on terminal state or episode length reaches maximum allowed length
            if done or (episode_length == self.max_episode_length):
                # print(step, next_state, action, reward, done)
                current_state = self.env.reset()
                episode_return = 0
                episode_length = 0

            # Sample random minibatch of transitions $(s_j , a_j , r_j , s_{j+1} )$ from D
            if step >= self.update_start and step % self.update_interval == 0:
                for _ in range(self.update_interval):
                    transactions = replay_memory.sample()
                    txn_current_states = transactions["current_states"]
                    txn_actions = transactions["actions"]
                    txn_rewards = transactions["rewards"]
                    txn_next_states = transactions["next_states"]
                    txn_dones = transactions["dones"]

                    ########################################
                    # Gradient decent on Critic
                    self.q_optimizer.zero_grad()
                    critic_q_value = self.critic(
                        txn_current_states, txn_actions)
                    with torch.no_grad():
                        y = txn_rewards + self.discount_factor * \
                            (1-txn_dones) * self.target_critic(txn_next_states,
                                                               self.target_actor(txn_next_states))
                    critic_q_loss = ((critic_q_value - y)**2).mean()
                    critic_q_loss.backward()
                    self.q_optimizer.step()

                    # Freeze critic network
                    self.critic.freeze()
                    #######################################

                    # Gradient decent on Actor
                    self.policy_optimizer.zero_grad()
                    actor_policy_loss = - \
                        self.critic(txn_current_states, self.actor(
                            txn_current_states)).mean()
                    actor_policy_loss.backward()
                    self.policy_optimizer.step()
                    self.critic.unfreeze()
                    #######################################

                    # Update target network
                    with torch.no_grad():
                        for actor_p, critic_p, target_actor_p, \
                            target_critic_p in zip(self.actor.parameters(),
                                                   self.critic.parameters(),
                                                   self.target_actor.parameters(),
                                                   self.target_critic.parameters()):
                            target_actor_p.data.mul_(
                                self.soft_target_update_factor)
                            target_actor_p.data.add_(
                                (1-self.soft_target_update_factor)*actor_p.data)

                            target_critic_p.data.mul_(
                                self.soft_target_update_factor)
                            target_critic_p.data.add_(
                                (1-self.soft_target_update_factor)*critic_p.data)
            if step > self.test_start and step % self.test_interval == 0:
                for j in range(self.test_episodes):
                    test_current_state, test_done, test_episode_return, test_episode_length = self.test_env.reset(), False, 0, 0
                    while not(test_done or (test_episode_length == self.max_episode_length)):
                        test_action = self.actor.action(torch.as_tensor(
                            test_current_state, dtype=torch.float32))
                        test_action = np.clip(test_action, self.min_action_value,
                                              self.max_action_value)
                        test_current_state, test_reward, test_done, _ = self.test_env.step(
                            test_action)
                        test_episode_return += test_reward
                        test_episode_length += 1
                        self.test_env.render(mode='rgb_array')
                    print(step, ", ", j, "return: ", test_episode_return,
                          ", ep len: ", test_episode_length)


env1 = gym.make('MountainCarContinuous-v0')
env2 = gym.make('MountainCarContinuous-v0')

env2 = wrappers.Monitor(env2, './videos/' + str(time()) + '/')

mountainCar = MountainCar(
    env=env1, test_env=env2)

# mountainCar = MountainCar(
#     env=gym.make('MountainCarContinuous-v0'), test_env=gym.make('MountainCarContinuous-v0'))
mountainCar.apply_ddgp()
