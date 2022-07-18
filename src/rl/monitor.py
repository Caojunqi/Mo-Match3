import gym


class Match3MonitorWrapper(gym.Wrapper):
    def __init__(self, env):
        super(Match3MonitorWrapper, self).__init__(env)
        self.rewards = []
        self.total_episode = 0

    def reset(self):
        obs = self.env.reset()
        self.rewards = []
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward != 0:
            print("reward:" + str(reward))
        self.rewards.append(reward)
        if done:
            self.total_episode += 1
            total_reward = sum(self.rewards)
            if total_reward != 0:
                print("total reward:" + str(total_reward))
            if self.total_episode % 100 == 0:
                print("total episode:" + str(self.total_episode))
        return obs, reward, done, info
