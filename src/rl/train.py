from collections import deque
from src.rl.environment import Match3Env
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy, MlpPolicy
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from src.rl.policy import Match3FeaturesExtractor, Match3Policy
from src.rl.monitor import Match3MonitorWrapper

if __name__ == '__main__':
    env = Match3Env()
    monitor_env = Match3MonitorWrapper(env)
    policy_kwargs = dict(
        features_extractor_class=Match3FeaturesExtractor,
    )
    model = PPO(Match3Policy, monitor_env, policy_kwargs=policy_kwargs, verbose=0)
    print("start learn=============")
    model.learn(total_timesteps=1000000)
    print("end learn=============")
    mean_reward, std_reward = evaluate_policy(model, monitor_env, n_eval_episodes=3)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
