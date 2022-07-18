from src.rl.environment import Match3Env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.rl.policy import Match3FeaturesExtractor, Match3Policy
import src.rl.config as config

if __name__ == '__main__':
    env = Match3Env()
    monitor_env = Monitor(env)
    policy_kwargs = dict(
        features_extractor_class=Match3FeaturesExtractor,
    )
    model = PPO(Match3Policy, monitor_env, policy_kwargs=policy_kwargs, verbose=0)

    eval_callback = EvalCallback(eval_env=monitor_env, best_model_save_path=config.MODEL_DIR)
    print("start learn=============")
    model.learn(total_timesteps=1000000, callback=[eval_callback])
    print("end learn=============")
    mean_reward, std_reward = evaluate_policy(model, monitor_env, n_eval_episodes=3)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
