import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC
from custom_env_SAC import Quadruped_Env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env():
    return Quadruped_Env()

env = DummyVecEnv([make_env])

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10
)

checkpoint = CheckpointCallback(save_path='./checkpoints_SAC_2',
                                save_freq=100000,
                                name_prefix="quad")

model = SAC(
    policy="MlpPolicy",
    env=env,
    device="cpu",
    learning_rate=3e-4,
    gamma=0.99,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    verbose=1,
)


model.learn(total_timesteps=3000000,
            callback=checkpoint)

model.save('Quadruped_SAC_2')
env.save('quadruped_vecnorm_sac_2.pkl')
