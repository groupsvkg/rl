import numpy as np
import torch


class ReplayMemory:
    def __init__(self, env, replay_memory_capacity, replay_minibatch_size):
        self.env = env
        self.replay_memory_capacity = replay_memory_capacity
        self.replay_minibatch_size = replay_minibatch_size

        self.current_states = np.zeros(
            (replay_memory_capacity, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros(
            (replay_memory_capacity, env.action_space.shape[0]), dtype=np.float32)
        self.rewards = np.zeros(replay_memory_capacity, dtype=np.float32)
        self.next_states = np.zeros(
            (replay_memory_capacity, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(replay_memory_capacity, dtype=np.float32)
        self.rollback_pointer = 0
        self.sample_interval = 0

    def store(self, current_state, action, reward, next_state, done):
        self.current_states[self.rollback_pointer] = current_state
        self.actions[self.rollback_pointer] = action
        self.rewards[self.rollback_pointer] = reward
        self.next_states[self.rollback_pointer] = next_state
        self.dones[self.rollback_pointer] = done
        self.rollback_pointer = (
            self.rollback_pointer + 1) % self.replay_memory_capacity
        self.sample_interval = min(
            self.sample_interval+1, self.replay_memory_capacity)

    def sample(self):
        idxs = np.random.randint(
            0, self.sample_interval, size=self.replay_minibatch_size)
        batch = dict(current_states=self.current_states[idxs],
                     actions=self.actions[idxs],
                     rewards=self.rewards[idxs],
                     next_states=self.next_states[idxs],
                     dones=self.dones[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
