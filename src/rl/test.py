import time

from sb3_contrib.ppo_mask import MaskablePPO

import src.rl.config as config
from src.envs.match3_env import Match3Env

if __name__ == '__main__':
    loaded_model = MaskablePPO.load(config.MODEL_DIR + "/best_model.zip")
    env = Match3Env()
    obs = env.reset()
    # env.render()
    print(obs)
    done = False
    while not done:
        time.sleep(5)
        action, state = loaded_model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        match3_action = env.get_action(action)
        print(match3_action)
        obs, reward, done, info = env.step(action)
        # env.render()
        print("action " + str(action) + "  reward " + str(reward))
        print(obs)
