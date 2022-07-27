from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src import definitions
from src.envs.levels import Level, Match3Levels
from src.envs.match3_env import Match3Env
from src.rl.policy import FeatureExtractor

TRAIN_LEVELS = [
    Level(9, 9, 6, [
        [-1, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, -1],
    ])
]

if __name__ == '__main__':
    env_kwargs = dict(
        levels=Match3Levels(TRAIN_LEVELS),
        random_state=0
    )

    vec_env = make_vec_env(Match3Env, n_envs=10, wrapper_class=Monitor, env_kwargs=env_kwargs,
                           vec_env_cls=SubprocVecEnv)

    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
    )
    model = MaskablePPO(MaskableActorCriticCnnPolicy, vec_env, policy_kwargs=policy_kwargs, verbose=0, n_steps=200,
                        batch_size=512)
    # model = PPO('MlpPolicy', monitor_env, verbose=0, batch_size=512)

    eval_callback = MaskableEvalCallback(eval_env=vec_env, eval_freq=10, best_model_save_path=definitions.MODEL_DIR)
    print("start learn=============")
    model.learn(total_timesteps=10000000, callback=[eval_callback])
    print("end learn=============")

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=5)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
