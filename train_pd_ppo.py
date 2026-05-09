from stable_baselines3 import PPO
from custom_env_new import Quadruped_Env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

class VelocityLogger(BaseCallback):
    def _on_step(self):
        vel = self.training_env.get_attr("last_forward_vel")[0]
        self.logger.record("robot/forward_velocity", vel)
        return True
    
if __name__ == '__main__':

    num_envs = 8

    def make_env():
        def _init():
            env = Quadruped_Env(render_mode='human')
            env.enable_curriculum = True
            env = Monitor(env)
            return env
        return _init

    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99 
    )

    policy_kwargs = dict(
        activation_fn = nn.LeakyReLU,
        net_arch = dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        log_std_init = -2.0
    )

    checkpoint = CheckpointCallback(
        save_freq=10_000_000 // num_envs,
        save_path="./checkpoints/git_repo_3_retrain_2",
        name_prefix="git_repo_3"
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq = 100000,
        deterministic=True,
        render=False
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4, 
        gamma=0.99,
        gae_lambda=0.95,        # Factor for trade-off of bia vs variance(increased from 0.95 for smoothness)
        clip_range=0.2,         
        ent_coef=0.002,         
        device="cpu",
        vf_coef=0.5,
        n_epochs=6,
        n_steps=2048,
        batch_size=512,
        policy_kwargs=policy_kwargs,
        max_grad_norm = 0.5,
        verbose=1,
        tensorboard_log = './ppo_logs_3'
    )
    try:
        model.learn(
            total_timesteps=30_000_000,
            callback=[checkpoint, VelocityLogger(), eval_callback],
        )
    except Exception as e:
        print("Training error.", e)
        raise

    finally:
        model.save("git_repo_3_retrain_2")
        env.save("git_repo_3_retrain_2.pkl")

