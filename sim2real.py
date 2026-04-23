import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from new_env_refined_3 import Quadruped_Env
import time
import numpy as np
import serial

ser = serial.Serial(port='', timeout=1, baudrate=9600) 

env = DummyVecEnv([lambda: Quadruped_Env(render_mode='human',
                                         xml_path='/media/srimukha-sarma/Windows-SSD/xtr_lair-main/src/robots/m2_metal_description/mujoco/flat_scene.xml')])

env = VecNormalize.load('vecnorm_pd_ppo_finetune_8_retrain.pkl', env)

mj_model = mujoco.MjModel.from_xml_path('/media/srimukha-sarma/Windows-SSD/xtr_lair-main/src/robots/m2_metal_description/mujoco/flat_scene.xml')

model = PPO.load('best_model/best_model.zip', env=env, device="cpu")

env.training = False
env.norm_reward = False

obs = env.reset()

#pd controller parameters
q_default = np.array([
            0.0, 0.9, -1.8, #FL
            0.0, 0.9, -1.8, #FR
            0.0, 0.9, -1.8, #RL
            0.0, 0.9, -1.8  #RR
        ])

q_range_leg = np.array([0.15, 0.4, 0.65])
q_range = np.tile(q_range_leg, 4)

torque_max = mj_model.actuator_ctrlrange[:, 1]

#Function to read the joint positions and velocities
def get_joint_positions():
    pass

def get_joint_velocites():
    pass

def pwm_to_motors(pwm):
    pass

Kp = 25
Kd = 2

while True:
    q = get_joint_positions()
    qd = get_joint_velocites()
    action, _ = model.predict(obs, deterministic=True)

    #pd controller components
    action = np.clip(action, -1, 1)
    qdes = q_default + action * q_range
    torques = Kp * (qdes - q) - Kd * qd
    torques = np.clip(torques, -torque_max, torque_max)
    u = torques / torque_max
    pwm = (u * 255).astype(int)
    pwm_to_motors(pwm)

    #NOT NEEDED FOR DEPLOYMENT, use sensor data instead 
    obs, reward, done, info = env.step(action)

    env.render()
    time.sleep(0.03) 

    if done:
        obs = env.reset()



