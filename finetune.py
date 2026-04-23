import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from new_env_refined_3 import Quadruped_Env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

class VelocityLogger(BaseCallback):
    def _on_step(self):
        vel = self.training_env.get_attr("last_forward_vel")[0]
        self.logger.record("robot/forward_velocity", vel)
        return True

def make_env():
    def _init():
        env = Quadruped_Env()
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':

    num_envs = 8

    #env = DummyVecEnv([make_env])

    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99 
    )

    eval_env = DummyVecEnv([make_env()])

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99
    )

    policy_kwargs = dict(
        activation_fn = nn.LeakyReLU,
        net_arch = dict(pi=[256, 128, 32], vf=[256, 128]),
        log_std_init = -1.5
    )

    checkpoint = CheckpointCallback(
        save_freq=5_000_000 // num_envs,
        save_path="./checkpoints/ppo_finetune_8_envs_retrain",
        name_prefix="quad_pd_ppo_finetune_8_envs_retrain",
        save_vecnormalize=True
    )

    best_model_ckpt = EvalCallback(
        eval_env,
        best_model_save_path='./best_model',
        log_path="./logs_2/",
        eval_freq = 10000,
        deterministic=True,
        render=False

    )

    model = PPO.load(
        "checkpoints/ppo_finetune_8_envs_2/quad_pd_ppo_finetune_8_envs_2_10000000_steps.zip",
        env,
        learning_rate=1e-4, 
        gamma=0.99,
        gae_lambda=0.97,        # Factor for trade-off of bias vs variance
        clip_range=0.2,         # Clipping parameter (crucial for PPO)
        ent_coef=0.005,           # Entropy coefficient (float, not "auto")
        device="cpu",
        vf_coef=0.5,
        n_epochs=10,
        n_steps=512,
        batch_size=512,
        policy_kwargs=policy_kwargs,
        target_kl=0.01,
        max_grad_norm = 0.5,
        verbose=1,
        tensorboard_log = './ppo_logs_retrain'
    )
    try:
        model.learn(
            total_timesteps=50_000_000,
            callback=[checkpoint, VelocityLogger(), best_model_ckpt],
        )
    except Exception as e:
        print("Training error.", e)
        raise
    finally:
        model.save("quad_pd_ppo_finetune_8_retrain")
        env.save("vecnorm_pd_ppo_finetune_8_retrain.pkl")

