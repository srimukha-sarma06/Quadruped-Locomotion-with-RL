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

        self.total_timesteps = 0

        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base"
        )

        #Foot contact ids
        self.fid_fr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FR_foot")
        self.fid_fl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FL_foot")
        self.fid_rr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "RR_foot")
        self.fid_rl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "RL_foot")

        #Domain Randomization(inital values)
        self.default_friction = self.model.geom_friction[:, 0].copy()
        self.default_body_mass = self.model.body_mass[self.base_body_id]

        # We use the BODY names for collision checking(not in the reward rn)
        self.calves = ["FL_shank_link", "FR_shank_link", "RL_shank_link", "RR_shank_link"]
        self.thighs = ["FL_thigh_link", "FR_thigh_link", "RL_thigh_link", "RR_thigh_link"]

        # We get the BODY IDs
        self.calf_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.calves]
        self.thigh_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.thighs]

        self.q_default = np.array([
            0.0, 0.9, -1.8, #FL
            0.0, 0.9, -1.8, #FR
            0.0, 0.9, -1.8, #RL
            0.0, 0.9, -1.8  #RR
        ])

        self.prev = 0.0
        self.prev_prev = np.zeros(12)
        self.prev_action = np.zeros(12)

        #Velocity logging 
        self.last_forward_vel = 0

        #Target Clearance for r_clear penalty
        self.target_clearance_height = 0.1

        #Initial values
        #Kp changed from 20 to 30 , Kd changed from 0.5 to 0.75
        self.Kp_leg = 25
        self.Kp_hip = 30
        self.Kd_leg = 0.6
        self.Kd_hip = 0.8 

        self.q_range_leg = np.array([0.15, 0.4, 0.65])
        self.q_range = np.tile(self.q_range_leg, 4)
        
        self.kp_leg = np.array([self.Kp_leg, self.Kp_leg, self.Kp_hip])
        self.Kp = np.tile(self.kp_leg, 4)

        self.kd_leg = np.array([self.Kd_leg, self.Kd_leg, self.Kd_hip])
        self.Kd = np.tile(self.kd_leg, 4)

        #initializing phase reward
        self.phase = 0.0

        assert self.model.nu == 12, "Expected 12 torque actuators"

        #Target linear and angular velocites
        self.command = np.array([0.6, 0.0, 0.0]) #changed from 0.5(according to froudes number)
        self.target_ang_vel = np.array([0.0, 0.0, 0.0]) #for straight line gait

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
                                            shape=(52,), dtype=np.float32)
        
        self.reset()

    def _site_velocity(self, site_name) -> list: #returns a list of x,y,z velocities
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
    
    def _contact_state(self, fid, idx, update) -> bool: #return true if in contact or false
        force = max(self.data.cfrc_ext[fid][5], 0) #(Torque_X, Torque_Y, Torque_Z, Force_X, Force_Y, Force_Z)

        if self.prev_contacts[idx]:
            contact = force > 2.0   # stay contact
        else:
            contact = force > 8.0   # enter contact
        if update:
            self.prev_contacts[idx] = contact
        return contact

    def _get_rpy(self) -> list: #return a list containing roll and pitch 
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

    def _foot_height(self, site_name) -> list:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id][2]
    
    def _get_curriculum_param(self) -> float: # Calculating the curriculum parameter for penalties, k -> [0, 1]
        t = self.total_timesteps
        k = 1 - np.cos(np.pi*t / 12500000)
        return k

    def _contact_prob(self, fid, idx, update):
        force = max(self.data.cfrc_ext[fid][5], 0)
        prob = 1 / (1 + np.exp(-10*(force - 5.0)))  # soft sigmoid around 5 N
        if update:
            self.prev_contacts[idx] = force > 2.0
        return prob

    def _pd_controller(self):
        return self.data.qpos[7:19] , self.data.qvel[6:18]

    def _get_obs(self) -> list[np.float32]:

        qpos = self.data.qpos[7:19]
        qvel = self.data.qvel[6:18]

        base_ang_vel = self.data.qvel[3:6]

        base_lin_vel = self.data.qvel[0:3]

        c_fl = self._contact_state(self.fid_fl, 0, False)
        c_fr = self._contact_state(self.fid_fr, 1, False)
        c_rl = self._contact_state(self.fid_rl, 2, False)
        c_rr = self._contact_state(self.fid_rr, 3, False)
        contacts = np.array([c_fl, c_fr, c_rl, c_rr])

        rot_mat = self.data.xmat[self.base_body_id].reshape(3, 3)

        projected_gravity = rot_mat[:, 2]

        commands = self.command

        last_act = self.last_action

        obs = np.concatenate([
            qpos,
            qvel,
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            commands,
            last_act,
            contacts
        ])

        return obs.astype(np.float32)

    def step(self, action) -> list:
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

        self.prev_prev = self.prev_action.copy()

        self.prev_action = self.last_action.copy()

        self.total_timesteps += 1

        self.last_action = action.copy()
        
        obs = self._get_obs()
        
        # Reward
        reward = self._compute_reward(torques, action, self.prev_action, self.prev_prev)
        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        #Random values for friction
        friction_mult = np.random.uniform(0.6, 1.4)
        self.model.geom_friction[:, 0] = self.default_friction * friction_mult

        #Random values for mass
        body_mass_mult = np.random.uniform(0.8, 1.2)
        self.model.body_mass[self.base_body_id] = self.default_body_mass * body_mass_mult 

        #Random values for Kp in +- 5 for dynamic randomization
        self.Kp_leg  = np.random.uniform(20, 30)
        self.Kp_hip = np.random.uniform(25, 35)

        #Random values for Kd in += 0.1 for dyanamic randomization
        
        self.Kd_leg = np.random.uniform(0.5, 0.8)
        self.Kd_hip = np.random.uniform(0.7, 0.9)

        self.kp_leg = np.array([self.Kp_leg, self.Kp_leg, self.Kp_hip])
        self.Kp = np.tile(self.kp_leg, 4)
        
        self.kd_leg = np.array([self.Kd_leg, self.Kd_leg, self.Kd_hip])
        self.Kd = np.tile(self.kd_leg, 4)
        
        #Resetting the phase value
        self.phase = 0.0

        # 1. Reset Internal State
        self.step_count = 0
        self.last_action = np.zeros(12)  # Clear history
        self.prev_action = np.zeros(12)
        self.prev_prev = np.zeros(12)
        self.prev = 0.0
        self.prev_contacts = [False, False, False, False]

        # 2. Randomize Command (Target Velocity)
        target_vx = np.random.uniform(0.5, 1.3) 
        self.command = np.array([target_vx, 0.0, 0.0]) 

        # 3. Reset Pose (Standing)
        self.data.qpos[7:19] = self.q_default
        self.data.qpos[2] = 0.4  # Lift base
        self.data.qpos[3:7] = [1, 0, 0, 0] # Upright quaternion

        # 4. Add Noise (Domain Randomization)
        self.data.qpos[7:19] += np.random.uniform(-0.05, 0.05, 12)
        self.data.qvel[:] = np.random.uniform(-0.1, 0.1, 18)

        # 5. Stabilize
        mujoco.mj_forward(self.model, self.data)
        
        # 6. Get first observation
        obs = self._get_obs()
        return obs, {}
    
    def _compute_reward(self, torques, action, prev_action, prev_prev) -> float:
        global_vel = self.data.qvel[0:3]
        rot_mat = self.data.xmat[self.base_body_id].reshape(3, 3)
        body_vel = rot_mat.T @ global_vel 

        body_ang_vel = self.data.qvel[3:6]
        
        forward_vel = body_vel[0]
        self.last_forward_vel = forward_vel
        #FORWARD VELOCITY REWARD
        target_velo = self.command[0]
        r_forward = 4.0 * np.exp(-5.0*(forward_vel - target_velo)**2) #changed from 2.5 , -5.0 -> 5.0, -10.0 to make the guassian distribution sharper(increased the weight too)

        #Angular velocity tracking
        error = (body_ang_vel[0] - self.target_ang_vel[0])**2 + (body_ang_vel[1] - self.target_ang_vel[1])**2 + (body_ang_vel[2] - self.target_ang_vel[2])**2
        r_ang_vel = 1.5 * np.exp(-5.0 * error)

        #Moving reward
        r_smooth = -0.1 * np.sum(np.square(action - prev_action)) -0.1 * (np.sum(np.square(action - 2*prev_action + prev_prev)))
        #HEIGHT PENALTY
        z_height = self.data.qpos[2]
        r_height = -10 * np.square(z_height - 0.35) #changed from -50 -> -30 for smoother learning at the beginning
        
        #Energy Penalty
        r_energy = - 0.0015 * np.sum(np.square(torques)) #weight changed from 0.0015 -> 0.0002 -> 0.0003 for reduced torque penalty

        #Foot slip penalty  

        c_fl = self._contact_prob(self.fid_fl, 0, True)
        c_fr = self._contact_prob(self.fid_fr, 1, True)
        c_rl = self._contact_prob(self.fid_rl, 2, True)
        c_rr = self._contact_prob(self.fid_rr, 3, True)

        v_fl = self._site_velocity("FL")
        v_fr = self._site_velocity("FR")
        v_rr = self._site_velocity("RR")
        v_rl = self._site_velocity("RL")

        r_slip = 0.0

        for contact, vel in ((c_fl, v_fl), (c_fr, v_fr), (c_rl, v_rl), (c_rr, v_rr)):
            if contact >= 0.5:
                slip_speed = vel[0]**2 + vel[1]**2
                r_slip -= 0.3 * slip_speed / max(abs(forward_vel), 0.1) #changed from 0.1, to increase the max
        
        #Foot Clearance penalty
        #foot heights and contact states

        clearance_error = 0
        contacts = [
            (c_fl, "FL"),
            (c_fr, "FR"),
            (c_rl, "RL"),   
            (c_rr, "RR")
        ]
        
        for contact, name in contacts:
            if contact < 0.5:
                height = self._foot_height(name)
                velocity = self._site_velocity(name)
                v_xy = (velocity[0]**2 + velocity[1]**2)**0.5
                clearance_error += (height - self.target_clearance_height)**2 * (v_xy**0.5)

        r_clear = - 1.0 * clearance_error 


        # Stabilization
        projected_gravity = rot_mat[:, 2]
        r_orient = - 1.0 * (np.sum(projected_gravity[0]**2 + projected_gravity[1]**2))

        # Z velocity penalty
        z_body_vel = body_vel[2]
        r_z_vel = - 0.5 * np.square(z_body_vel)

        #Joint Pose penalty
        q = self.data.qpos[7:19]
        r_pose = - 0.2 * np.sum((q - self.q_default)**2) #changed from 0.5 -> 0.3 -> 0.1

        #LATERAL DRIFT penalty(to stop it from walking sideways)
        r_lat = - 0.3 * np.square(body_vel[1]) 

        #Penalizing yaw rate(to prevent the bot from moving in circles)
        #yaw = self.data.qvel[5]
        #r_yaw = - 0.3 * np.square(yaw)

        # Alive Reward
        r_alive = 0.1

        #getting the curriculum factor
        k = self._get_curriculum_param()

        #Phase reward
        dt_eff = self.dt * 10
        freq = 0.3
        self.phase  = (self.phase + 2 * np.pi * dt_eff * freq)  % (2 * np.pi)

        p_fl = 0.5 * (1 + np.sin(self.phase))
        p_fr = 0.5 * (1 - np.sin(self.phase))
        p_rl = 0.5 * (1 - np.sin(self.phase))
        p_rr = 0.5 * (1 + np.sin(self.phase))

        r_phase = 0.0
        actual_contacts = c_fl, c_fr, c_rl, c_rr
        r_phase = -0.8 * (
            (c_fl - p_fl)**2 +
            (c_fr - p_fr)**2 +
            (c_rl - p_rl)**2 +
            (c_rr - p_rr)**2
        )

        total = r_forward + r_ang_vel + r_alive + k*(r_energy + r_height + r_orient  + r_z_vel + r_pose + r_slip + r_smooth + r_clear + r_lat)

        return total

    def _is_fallen(self) -> bool:
        #terminating if roll, pitch and height cross thresholds and if the velocity is too low after half the iteration
        z_height = self.data.qpos[2]
        terminated = False
        roll, pitch = self._get_rpy()
        if abs(roll) > np.deg2rad(29) or abs(pitch) > np.deg2rad(34) or z_height < 0.12:
            terminated = True
        return terminated

    
    def render(self, mode='human'):
        #Live rendering on training
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    
