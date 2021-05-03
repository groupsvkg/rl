import gym


class MountainCar:
    def __init__(self, env, episodes=100, max_episode_length=1000,
                 replay_memory_capacity=int(1e6), replay_minibatch_size=32,
                 actor_learning_rate=1e-4, critic_learning_rate=1e-3, actor_noise=0.1,
                 discount_factor=0.99, soft_target_update_factor=0.001):
        self.env = env
        self.episodes = episodes
        self.max_episode = max_episode_length
        self.replay_memory_capacity = replay_memory_capacity
        self.replay_minibatch_size = replay_minibatch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_noise = actor_noise
        self.discount_factor = discount_factor
        self.soft_target_update_factor = soft_target_update_factor

    def apply_ddgp(self):

        # import gym
        # env = gym.make('MountainCarContinuous-v0')
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
