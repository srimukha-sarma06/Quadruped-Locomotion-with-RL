import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_env_new import Quadruped_Env
import time
import numpy as np

env = DummyVecEnv([lambda: Quadruped_Env(render_mode='human',
                                         xml_path='flat_scene.xml')])

env = VecNormalize.load('git_repo_3_retrain_2.pkl', env)

env.training = False
env.norm_reward = False

env.set_attr("enable_curriculum", False)

model = PPO.load('checkpoints/git_repo_3_retrain_2/git_repo_3_50000000_steps.zip', env=env, device="cpu")

obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    env.render()
    time.sleep(0.03) 

    if done:
        obs = env.reset()



