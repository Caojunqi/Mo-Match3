from src.envs.match3_env import Match3Env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from src.rl.policy import Match3FeaturesExtractor, Match3Policy, FeatureExtractor
import src.rl.config as config
import gym
import numpy as np

if __name__ == '__main__':
    env = Match3Env()
    env = Monitor(env)
    policy_kwargs = dict(
        # features_extractor_class=FeatureExtractor,
    )
    model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=0, batch_size=512)
    # model = PPO('MlpPolicy', monitor_env, verbose=0, batch_size=512)

    eval_callback = EvalCallback(eval_env=env, best_model_save_path=config.MODEL_DIR)
    print("start learn=============")
    model.learn(total_timesteps=10000000, callback=[eval_callback])
    print("end learn=============")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
