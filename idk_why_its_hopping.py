import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class Quadruped_Env(gym.Env):
    metadata = {'render_modes' : ['human'], "render_fps": 60}
    def __init__(self, render_mode=None, xml_path=None):
        super(Quadruped_Env, self).__init__()

        if xml_path is None:
            xml_path = '/media/srimukha-sarma/Windows-SSD/xtr_lair-main/src/robots/m2_metal_description/mujoco/scene.xml'

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.sim = None
        self.step_count = 0
        self.max_steps=1000
        self.dt = 0.02

        #Foot contact ids
        self.fid_fr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FR_foot")
        self.fid_fl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FL_foot")
        self.fid_rr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "RR_foot")
        self.fid_rl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "RL_foot")

        # In __init__
        # We use the BODY names for collision checking
        self.calves = ["FL_shank_link", "FR_shank_link", "RL_shank_link", "RR_shank_link"]
        self.thighs = ["FL_thigh_link", "FR_thigh_link", "RL_thigh_link", "RR_thigh_link"]

        # We get the BODY IDs
        self.calf_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.calves]
        self.thigh_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.thighs]

        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base"
        )

        self.q_default = np.array([
            0.0, 0.8, -1.6, #FL
            0.0, 0.8, -1.6, #FR
            0.0, 0.8, -1.6, #RL
            0.0, 0.8, -1.6  #RR
        ])

        self.q_range = np.array([0.8]*12)
        self.Kp = np.array([50.0]*12)
        self.Kd = np.array([1.0]*12)

        assert self.model.nu == 12, "Expected 12 torque actuators"

        # ---- Action space (normalized torques) ----
        self.torque_limit = self.model.actuator_ctrlrange[:, 1]

        self.render_mode = render_mode
        self.viewer=None
        

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32) #.Box for continuos and .Discrete for discrete
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(32,), dtype=np.float32)
        
        self.reset()
    
    def _get_rpy(self):
        # Mujoco Quat is [w, x, y, z]
        q = self.data.qpos[3:7]
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2) # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation) - WE CALCULATE IT BUT DON'T RETURN IT
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch])
        
    def _get_obs(self):
        qpos = self.data.qpos[7:19]
        qvel = self.data.qvel[6:18]
        base_quat = self.data.qpos[3:7]

        base_ang_vel = self.data.qvel[3:6]
        global_lin_vel = self.data.qvel[0:3]

        gravity = np.array([0, 0, -1])

        rot_mat = self.data.xmat[self.base_body_id].reshape(3, 3)

        gravity_body = rot_mat.T @ gravity

        base_lin_vel = rot_mat.T @ global_lin_vel

        orientation = self._get_rpy()  

        obs = np.concatenate([
            qpos,
            qvel,
            base_ang_vel,
            base_lin_vel,
            orientation,
        ])

        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0) 
        q_des = self.q_default + action * self.q_range

        for _ in range(10):
            q = self.data.qpos[7:19]
            qd = self.data.qvel[6:18]
            
            torques = self.Kp * (q_des - q) - self.Kd * qd
            torques = np.clip(torques, -self.torque_limit, self.torque_limit)
            
            self.data.ctrl[:] = torques
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        
        obs = self._get_obs()
        reward = self._compute_reward(torques)

        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward -= 1.0

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 1. Reset to Default Standing Pose first
        self.data.qpos[7:19] = self.q_default
        
        # 2. Add Noise
        self.data.qpos[7:19] += np.random.uniform(-0.05, 0.05, 12) #
        self.data.qvel[:] = np.random.uniform(-0.1, 0.1, 18)     

        # 3. CRITICAL: Lift the base up (Adjust 0.4 to your robot's leg length)
        self.data.qpos[2] = 0.4  
        
        # 4. Correct Quaternion (Upright)
        self.data.qpos[3:7] = [1, 0, 0, 0] 

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}
    
    def _compute_reward(self, torques):
        global_vel = self.data.qvel[0:3]
        rot_mat = self.data.xmat[self.base_body_id].reshape(3, 3)
        body_vel = rot_mat.T @ global_vel 
        
        forward_vel = body_vel[0]
        #FORWARD VELOCITY REWARD
        r_forward = 2.0 * np.clip(forward_vel, -1.0, 10.0)

        #CONTACT PENALTY and REWARD
        c_fl = np.linalg.norm(self.data.cfrc_ext[self.fid_fl][:3]) > 0.1 #TRUE if in contact
        c_fr = np.linalg.norm(self.data.cfrc_ext[self.fid_fr][:3]) > 0.1 #TRUE if in contact
        c_rl = np.linalg.norm(self.data.cfrc_ext[self.fid_rl][:3]) > 0.1 #TRUE if in contact
        c_rr = np.linalg.norm(self.data.cfrc_ext[self.fid_rr][:3]) > 0.1 #TRUE if in contact

        contact_penalty = 0.0

        if forward_vel > 0.3:
            if (c_fl and c_fr) or (c_rl and c_rr):
                contact_penalty -= 1.0
            if (c_fl and c_rl) or (c_fr and c_rr):
                contact_penalty -= 1.0
            if (c_fl and c_rr) or (c_fr and c_rl):
                contact_penalty += 1.0
        #IF all legs in contact at once
        if c_fl and c_fr and c_rl and c_rr:
            contact_penalty -= 1.0

        #HEIGHT PENALTY
        z_height = self.data.qpos[2]
        if z_height < 0.25:
            height_penalty = -1.0
        else:
            height_penalty = 0.0
            
        r_alive = 1.0

        r_energy = -0.0005 * np.sum(np.square(torques))

        # Stabilization
        roll_rate = self.data.qvel[3]
        pitch_rate = self.data.qvel[4]

        r_roll = -0.10 * np.square(roll_rate)

        r_pitch = -0.02 * np.square(pitch_rate)

        collision_penalty = 0.0

        for body_id in self.calf_ids + self.thigh_ids:
            if np.linalg.norm(self.data.cfrc_ext[body_id][:3]) > 0.1:
                collision_penalty -= 1.0

        #idle penalty
        if forward_vel <  0.2:
            r_idle = -0.5
        else:
            r_idle = 0.0

        total = 1.5*r_forward + 3.0*contact_penalty + r_alive + r_energy + r_idle + height_penalty + 2*r_roll + 2*r_pitch + collision_penalty 
        #return np.tanh(total)
        return total

    def _is_fallen(self):
        z_height = self.data.qpos[2]
        return z_height < 0.12

    
    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    