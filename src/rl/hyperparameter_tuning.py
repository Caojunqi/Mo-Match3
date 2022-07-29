import optuna
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

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


def suggest_params(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': trial.suggest_int('n_steps', 16, 2048, log=True),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'n_epochs': trial.suggest_int('n_epochs', 1, 48, log=True),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 1.)
    }


def optimize_agent(trial):
    """ Train the model and optimize
            Optuna maximises the negative log likelihood, so we
            need to negate the reward here
        """
    env_kwargs = dict(
        levels=Match3Levels(TRAIN_LEVELS),
        random_state=None
    )
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
    )
    agent_params = suggest_params(trial)
    vec_env = make_vec_env(Match3Env, n_envs=10, wrapper_class=Monitor, env_kwargs=env_kwargs)

    model = MaskablePPO(MaskableActorCriticCnnPolicy, vec_env, policy_kwargs=policy_kwargs, verbose=0, **agent_params)
    model.learn(10000)
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)

    return mean_reward


if __name__ == '__main__':
    study = optuna.create_study(study_name="optimize-ppo-hyperparameter", direction="maximize")
    study.optimize(optimize_agent, n_trials=100)
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     for _ in range(5):
    #         executor.submit(study.optimize, optimize_agent, 100)
