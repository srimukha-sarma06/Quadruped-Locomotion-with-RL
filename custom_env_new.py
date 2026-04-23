import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class Quadruped_Env(gym.Env):
    metadata = {'render_modes' : ['human'], "render_fps": 60}
    def __init__(self, render_mode=None, xml_path=None):
        super(Quadruped_Env, self).__init__()

        if xml_path is None:
            xml_path = '/media/srimukha-sarma/Windows-SSD/xtr_lair-main/src/robots/m2_metal_description/mujoco/flat_scene.xml'

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.sim = None
        self.step_count = 0
        self.max_steps=1000
        self.dt = 0.02

        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base"
        )

        #Foot contact ids
        self.fid_fr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FR_shank_link")
        self.fid_fl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FL_shank_link")
        self.fid_rr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "RR_shank_link")
        self.fid_rl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "RL_shank_link")

        #Domain Randomization
        self.default_friction = self.model.geom_friction[:, 0].copy()
        self.default_body_mass = self.model.body_mass[self.base_body_id]

        # We use the BODY names for collision checking
        self.calves = ["FL_shank_link", "FR_shank_link", "RL_shank_link", "RR_shank_link"]
        self.thighs = ["FL_thigh_link", "FR_thigh_link", "RL_thigh_link", "RR_thigh_link"]

        # We get the BODY IDs
        self.calf_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.calves]
        self.thigh_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.thighs]

        self.q_default = np.array([ #changed for stability
            0.0, 0.9, -1.8, #FL
            0.0, 0.9, -1.8, #FR
            0.0, 0.9, -1.8, #RL
            0.0, 0.9, -1.8  #RR
        ])

        self.total_timesteps = 0

        self.last_forward_vel = 0.0

        self.prev = 0.0
        self.prev_prev = np.zeros(12)
        self.prev_action = np.zeros(12)

        #highest velo per step
        self.velo_sum = 0.0
        self.velo_count = 0

        #target height for foot clearance
        self.target_clearance_height = 0.1
        
        #target angular velocity 
        self.target_ang_velocity = np.array([0.0, 0.0, 0.0])

        #Air time reward componenets
        self.air_time = np.array([0.0, 0.0, 0.0, 0.0])
        self.last_contact_state = np.array([0.0, 0.0, 0.0, 0.0])

        #Initial values
        self.Kp_leg = 25
        self.Kp_hip = 30
        self.Kd_leg = 0.5
        self.Kd_hip = 0.8

        self.q_range_leg = np.array([0.25, 0.4, 0.65])
        self.q_range = np.tile(self.q_range_leg, 4)
        
        self.kp_leg = np.array([self.Kp_leg, self.Kp_leg, self.Kp_hip])
        self.Kp = np.tile(self.kp_leg, 4)

        self.kd_leg = np.array([self.Kd_leg, self.Kd_leg, self.Kd_hip])
        self.Kd = np.tile(self.kd_leg, 4)

        assert self.model.nu == 12, "Expected 12 torque actuators"

        #default command velocity
        self.command = np.array([0.6, 0.0, 0.0])

        self.site_ids = {
            "FL": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "FL_foot"),
            "FR": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "FR_foot"),
            "RL": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "RL_foot"),
            "RR": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "RR_foot"),
        }

        # Initialize last_action
        self.last_action = np.zeros(12)
        self.prev_contacts = [False, False, False, False]

        self.torque_limit = self.model.actuator_ctrlrange[:, 1]

        self.render_mode = render_mode
        self.viewer=None
        

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32) #.Box for continuos and .Discrete for discrete
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(48,), dtype=np.float32)
        
        self.reset()

    def _site_velocity(self, site_name) -> list:
        site_id = self.site_ids[site_name]
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            vel,
            0
        )
        return vel[:3]
    
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
    
    def _contact_state(self, fid, idx, update):
        force = self.data.cfrc_ext[fid][5] # (Torque_X, Torque_Y, Torque_Z, Force_X, Force_Y, Force_z)

        if self.prev_contacts[idx]:
            contact = force > 2.0   # stay contact
        else:
            contact = force > 8.0   # enter contact

        if update:
            self.prev_contacts[idx] = contact
        return contact

    def _foot_height(self, site_name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id][2]
    
    def _get_curriculum(self):
        t = self.total_timesteps
        if t <= 25e+6:
            k = 1 - np.cos(2 * np.pi * 1e-8 * t)
        else:
            k = 1
        return k
        
    def _get_obs(self):
        qpos = self.data.qpos[7:19] 
        qvel = self.data.qvel[6:18]
        rot_mat = self.data.xmat[self.base_body_id].reshape(3, 3)

        base_lin_vel = rot_mat.T @ self.data.qvel[0:3]
        base_ang_vel = rot_mat.T @ self.data.qvel[3:6]

        '''
        c_fl = self._contact_state(self.fid_fl, 0, False)
        c_fr = self._contact_state(self.fid_fr, 1, False)
        c_rl = self._contact_state(self.fid_rl, 2, False)
        c_rr = self._contact_state(self.fid_rr, 3, False)

        contacts = np.array([c_fl, c_fr, c_rl, c_rr]).astype(np.float32)
        '''

        commands = self.command

        projected_gravity = rot_mat[:, 2]

        last_action = self.last_action

        obs = np.concatenate([
            qpos,
            qvel,
            base_ang_vel,
            base_lin_vel,
            projected_gravity,
            last_action,
            commands
        ])

        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0) 
        q_des = self.q_default + action * self.q_range

        for _ in range(20):
            q = self.data.qpos[7:19]
            qd = self.data.qvel[6:18]
            torques = self.Kp * (q_des - q) - self.Kd * qd
            torques = np.clip(torques, -self.torque_limit, self.torque_limit)
            self.data.ctrl[:] = torques
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        self.total_timesteps += 1

        self.prev_prev = self.prev_action.copy()

        self.prev_action = self.last_action.copy()

        # FIX 1: Update action BEFORE observation
        self.last_action = action.copy()
        
        obs = self._get_obs()
        
        # Reward
        reward, r_forward, r_ang_vel, r_clear, r_smooth, r_pose, r_slip, r_energy, r_orient, body_vel, ang_vel = self._compute_reward(torques, action, self.prev_action, self.prev_prev)
        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_steps

        self.velo_sum += body_vel[0]
        self.velo_count += 1

        if terminated or truncated:
            avg_velo = self.velo_sum / self.velo_count
            print(f"Forward Reward: {r_forward}")
            print(f"Energy Penalty: {r_energy}")
            print(f"Orientation Penalty: {r_orient}")
            print(f"Pose Penalty: {r_pose}")
            print(f"Slip Penalty: {r_slip}")
            print(f"Smoothness Penalty: {r_smooth}")
            print(f"Clearance Penalty: {r_clear}")
            print(f"Ang Vel Reward: {r_ang_vel}")
            print(f"body vel: {avg_velo}")
            print("------TOTAL REWARD------")
            print(reward)
            print(f"Angular Velocity: {ang_vel[0]}x, {ang_vel[1]}y, {ang_vel[2]}z")
            print(f"Total_timesteps: {self.total_timesteps}")
            print("------------------------")
            self.velo_sum = 0.0
            self.velo_count = 0
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        #Random values for friction
        friction_mult = np.random.uniform(0.8, 1.2)
        self.model.geom_friction[:, 0] = self.default_friction * friction_mult

        #Random values for mass
        body_mass_mult = np.random.uniform(0.9, 1.1)
        self.model.body_mass[self.base_body_id] = self.default_body_mass * body_mass_mult 

        #Random values for Kp in +- 5 for dynamic randomization(only for post training)
        #self.Kp_leg  = np.random.uniform(20, 30)
        #self.Kp_hip = np.random.uniform(25, 35)

        #Random values for Kd in += 0.1 for dyanamic randomization (onlt for post training)
        
        #self.Kd_leg = np.random.uniform(0.5, 0.7)
        #self.Kd_hip = np.random.uniform(0.7, 0.8)

        self.kp_leg = np.array([self.Kp_leg, self.Kp_leg, self.Kp_hip])
        self.Kp = np.tile(self.kp_leg, 4)
        
        self.kd_leg = np.array([self.Kd_leg, self.Kd_leg, self.Kd_hip])
        self.Kd = np.tile(self.kd_leg, 4)

        # 1. Reset Internal State
        self.step_count = 0
        self.last_action = np.zeros(12)  # Clear history
        self.prev_action = np.zeros(12)
        self.prev_prev = np.zeros(12)
        self.prev = 0.0
        self.prev_contacts = [False, False, False, False]

        # 2. Randomize Command (Target Velocity)
        if self.total_timesteps <= 10e+6:
            target_vx = np.random.uniform(0.3, 0.5)
        else:
            target_vx = np.random.uniform(0.3, 1.0)
        self.command = np.array([target_vx, 0.0, 0.0]) 

        # 3. Reset Pose (Standing)
        self.data.qpos[7:19] = self.q_default
        self.data.qpos[2] = 0.4  # Lift base
        self.data.qpos[3:7] = [1, 0, 0, 0] # Upright quaternion

        # 4. Add Noise (Domain Randomization)
        self.data.qpos[7:19] += np.random.uniform(-0.05, 0.05, 12)
        self.data.qvel[6:18] = np.random.uniform(-0.5, 0.5, 12) #changed to only the [6:18] and increased the noise from 0.1

        # 5. Stabilize
        mujoco.mj_forward(self.model, self.data)
        
        # 6. Get first observation
        obs = self._get_obs()
        return obs, {}
    
    def _compute_reward(self, torques, action, prev_action, prev_prev):
        rot_mat = self.data.xmat[self.base_body_id].reshape(3, 3)
        body_vel = self.data.qvel[0:3]  # major fix(dont take the velocity from rot_matrix)
        body_vel_local = rot_mat.T @ body_vel


        qd = self.data.qvel[6:18]
        q = self.data.qpos[7:19]
        
        forward_vel = body_vel_local[0]
        self.last_forward_vel = forward_vel

        #FORWARD VELOCITY REWARD
        error = (body_vel_local[0] - self.command[0])**2 + (body_vel_local[1] - self.command[1])**2
        r_forward = 6.0 * np.exp(- 5* error)

        #Angular Velocity Tracking 
        ang_vel = self.data.qvel[3:6]
        r_ang_vel = 1.5 * np.exp(- 5 * (ang_vel[2] - self.target_ang_velocity[2])**2)

        #Action Smoothness Penalty
        r_smooth = -0.5 * np.sum(np.square(action - prev_action)) -0.2 * (np.sum(np.square(action - 2*prev_action + prev_prev)))

        #Power loss penlaty
        tau_u = 0.0477
        b = 0.000135
        K = 4.81
        c_E = 0.01

        tau_f = tau_u * np.sign(qd) + b * qd

        P_f = np.abs(tau_f * qd)

        P_J = (1.0 / K) * np.sum(np.square((torques + tau_f)))

        P_total = P_f + P_J

        r_energy = - c_E * np.mean(P_total)
        #Foot slip penalty

        c_fl = self._contact_state(self.fid_fl, 0, True)
        c_fr = self._contact_state(self.fid_fr, 1, True)
        c_rl = self._contact_state(self.fid_rl, 2, True)
        c_rr = self._contact_state(self.fid_rr, 3, True)

        v_fl = self._site_velocity("FL")
        v_fr = self._site_velocity("FR")
        v_rr = self._site_velocity("RR")
        v_rl = self._site_velocity("RL")

        r_slip = 0.0

        for contact, vel in ((c_fl, v_fl), (c_fr, v_fr), (c_rl, v_rl), (c_rr, v_rr)):
            if contact:
                slip_speed = vel[0]**2 + vel[1]**2
                r_slip -= 0.07 * slip_speed

        #Foot Clearance penalty

        clearance_error = 0
        contacts = [
            (c_fl, "FL"),
            (c_fr, "FR"),
            (c_rl, "RL"),   
            (c_rr, "RR")
        ]
        
        for contact, name in contacts:
            height = self._foot_height(name)
            velocity = self._site_velocity(name)
            v_xy = (velocity[0]**2 + velocity[1]**2)**0.5
            clearance_error += (height - self.target_clearance_height)**2 * (v_xy**0.5)

        r_clear = -20.0 * clearance_error 

        # Stabilization
        roll  = ang_vel[0]
        pitch = ang_vel[1]

        r_orient = -3.0 * (roll**2 + pitch**2) -1.2 * (body_vel[2]**2)

        #Joint Pose penalty
        r_pose = -0.5 * np.sum((q - self.q_default)**2)

        #Curriculum term 
        k = self._get_curriculum()

        penalties = r_clear + r_smooth + r_pose + r_slip + r_energy + r_orient 

        total = r_forward + r_ang_vel + k * (penalties)

        return total, r_forward, r_ang_vel, r_clear, r_smooth, r_pose, r_slip, r_energy, r_orient, body_vel, ang_vel

    def _is_fallen(self):
        z_height = self.data.qpos[2]
        roll, pitch = self._get_rpy()
        terminated = False
        if abs(roll) > np.deg2rad(29) or abs(pitch) > np.deg2rad(34) or z_height < 0.18:
            terminated = True
        return terminated

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()