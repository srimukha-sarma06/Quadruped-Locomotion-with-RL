import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from new_env_refined import Quadruped_Env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn

def make_env():
    return Quadruped_Env()

env = DummyVecEnv([lambda: Quadruped_Env()])

env = VecNormalize.load('vecnorm_pd_lstm_ppo.pkl', env)

policy_kwargs = dict(
    activation_fn = nn.Tanh,
    net_arch = dict(pi=[256, 256], vf=[256, 256]),

    log_std_init = -2.0
)

checkpoint = CheckpointCallback(
    save_freq=1_000_000,
    save_path="./checkpoints/lstm_ppo3",
    name_prefix="quad_pd_lstm_ppo2"
)

model = RecurrentPPO.load(
    "checkpoints/lstm_ppo2/quad_pd_lstm_ppo2_1500000_steps.zip",
    env,
    learning_rate=3e-5, #number of times to optimize the surrogate loss per rollout
    gamma=0.99,
    gae_lambda=0.95,        # Factor for trade-off of bias vs variance
    clip_range=0.2,         # Clipping parameter (crucial for PPO)
    ent_coef=0.0,           # Entropy coefficient (float, not "auto")
    device="cpu",
    vf_coef=1,
    n_epochs=5,
    n_steps=4096,
    batch_size=512,
    policy_kwargs=policy_kwargs,
    verbose=1,
)
try:
    model.learn(
        total_timesteps=3_300_000,
        callback=checkpoint
    )
except:
    print("Training error.")
finally:
    model.save("quad_pd_lstm_ppo2")
    env.save("vecnorm_pd_lstm_ppo2.pkl")

