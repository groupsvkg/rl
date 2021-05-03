from replay_memory import ReplayMemory
from actor_network import Actor
from critic_network import Critic
from copy import deepcopy
import gym
import numpy as np
import torch
from torch.optim import Adam


class MountainCar:
    def __init__(self, env, episodes=10000, max_episode_length=1000,
                 replay_memory_capacity=int(1e6), replay_minibatch_size=64,
                 policy_learning_rate=1e-4, q_learning_rate=1e-3, actor_noise=0.1,
                 discount_factor=0.99, soft_target_update_factor=0.001):
        self.env = env
        self.episodes = episodes
        self.max_episode_length = max_episode_length
        self.replay_memory_capacity = replay_memory_capacity
        self.replay_minibatch_size = replay_minibatch_size
        self.actor_learning_rate = policy_learning_rate
        self.critic_learning_rate = q_learning_rate
        self.actor_noise = actor_noise
        self.discount_factor = discount_factor
        self.soft_target_update_factor = soft_target_update_factor

        # Actor/Critic
        self.actor = Actor(self.env)
        self.critic = Critic(self.env)

        # Target Actor/Critic
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        # Optimizers
        self.policy_optimizer = Adam(self.actor.policy.parameters(),
                                     lr=self.actor_learning_rate)
        self.q_optimizer = Adam(self.critic.q.parameters(),
                                lr=self.critic_learning_rate)

        # Action Range
        self.max_action_value = int(self.env.action_space.high[0])
        self.min_action_value = int(self.env.action_space.low[0])

    def apply_ddgp(self):
        # Initialise replay memory D to capacity N
        replay_memory = ReplayMemory(env=self.env, replay_memory_capacity=self.replay_memory_capacity,
                                     replay_minibatch_size=self.replay_minibatch_size)

        # Freeze target network
        self.target_actor.freeze()
        self.target_critic.freeze()

        # Interact with Environment
        current_state = self.env.reset()
        episode_return = 0
        episode_length = 0

        for step in range(self.episodes):
            action = self.actor.action(torch.as_tensor(
                current_state, dtype=torch.float32)) + \
                self.actor_noise * \
                np.random.randn(self.max_action_value)
            action = np.clip(action, self.min_action_value,
                             self.max_action_value)

            # Execute action $a_t$ and observe reward $r_t$ , next state $s_{t+1}$
            next_state, reward, done, info = self.env.step(action)
            done = False if episode_length == self.max_episode_length else done
            episode_return += reward
            episode_length += 1

            self.env.render()
            print(step, next_state, reward, done)

            # Store transition $(s_t , a_t , r_t , s_{t+1} )$ in Ds
            replay_memory.store(current_state, action,
                                reward, next_state, done)

            # Update current state
            current_state = next_state

            # Reset on terminal state or episode length reaches maximum allowed length
            if done or (episode_length == self.max_episode_length):
                current_state = self.env.reset()
                episode_return = 0
                episode_length = 0

            # Sample random minibatch of transitions $(s_j , a_j , r_j , s_{j+1} )$ from D
            transactions = replay_memory.sample()


mountainCar = MountainCar(gym.make('MountainCarContinuous-v0'))
mountainCar.apply_ddgp()

#  import gym
# env = gym.make('MountainCarContinuous-v0')
# print("State Space:  ", env.observation_space)
# print("Action Space: ", env.action_space.high)
# for i_episode in range(2):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(i_episode, t, observation, action, reward, done, info)

#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
