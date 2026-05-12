import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class Quadruped_Env(gym.Env):
    metadata = {'render_modes' : ['human'], "render_fps": 60}
    def __init__(self, render_mode=None, xml_path=None):
        super(Quadruped_Env, self).__init__()

        if xml_path is None:
            xml_path = 'flat_scene.xml'

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.enable_curriculum = True

        self.render_mode = render_mode
        self.sim = None
        self.step_count = 0
        self.max_steps=1000
        self.dt = 0.02
        self.num_envs = 8

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

        #Domain Randomization(body mass, friction initial values)
        self.default_friction = self.model.geom_friction[:, 0].copy()
        self.default_body_mass = self.model.body_mass[self.base_body_id]

        # We use the BODY names for collision checking
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

        self.total_timesteps = 0

        self.last_forward_vel = 0.0

        self.prev = 0.0
        self.prev_prev = np.zeros(12)
        self.prev_action = np.zeros(12)

        #target height for foot clearance
        self.target_clearance_height = 0.08
        
        #target angular velocity 
        self.target_ang_velocity = np.array([0.0, 0.0, 0.0])

        #target orientation
        self.target_orientation = np.array([0.0, 0.0, 1.0])

        #Joint ids
        self.adr_lin_vel    = self.model.sensor_adr[self.model.sensor("base_lin_vel").id]
        self.adr_ang_vel    = self.model.sensor_adr[self.model.sensor("base_ang_vel").id]
        self.adr_imu_ori    = self.model.sensor_adr[self.model.sensor("imu_orientation").id]
        self.adr_cf_fl      = self.model.sensor_adr[self.model.sensor("contact_force_FL").id]
        self.adr_cf_fr      = self.model.sensor_adr[self.model.sensor("contact_force_FR").id]
        self.adr_cf_rl      = self.model.sensor_adr[self.model.sensor("contact_force_RL").id]
        self.adr_cf_rr      = self.model.sensor_adr[self.model.sensor("contact_force_RR").id]

        self.adr_force_calf_fl = self.model.sensor_adr[self.model.sensor("FL_calf_force_sensor").id]
        self.adr_force_calf_fr = self.model.sensor_adr[self.model.sensor("FR_calf_force_sensor").id]
        self.adr_force_calf_rl = self.model.sensor_adr[self.model.sensor("RL_calf_force_sensor").id]
        self.adr_force_calf_rr = self.model.sensor_adr[self.model.sensor("RR_calf_force_sensor").id]


        self.joint_names = ["FL_hip_position_sensor", "FL_thigh_position_sensor", "FL_calf_position_sensor",
                            "FR_hip_position_sensor", "FR_thigh_position_sensor", "FR_calf_position_sensor",
                            "RL_hip_position_sensor", "RL_thigh_position_sensor", "RL_calf_position_sensor",
                            "RR_hip_position_sensor", "RR_thigh_position_sensor", "RR_calf_position_sensor"]
        
        self.joint_vels = ["FL_hip_velocity_sensor", "FL_thigh_velocity_sensor", "FL_calf_velocity_sensor",
                           "FR_hip_velocity_sensor", "FR_thigh_velocity_sensor", "FR_calf_velocity_sensor",
                           "RL_hip_velocity_sensor", "RL_thigh_velocity_sensor", "RL_calf_velocity_sensor",
                           "RR_hip_velocity_sensor", "RR_thigh_velocity_sensor", "RR_calf_velocity_sensor"]
        
        self.joint_ids = []
        for name in self.joint_names:
            self.joint_ids.append(self.model.sensor_adr[self.model.sensor(name).id])

        self.joint_vel_ids = []
        for name in self.joint_vels:
            self.joint_vel_ids.append(self.model.sensor_adr[self.model.sensor(name).id])

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

        #torque monitoring 
        self.torques = np.zeros(12)
        self.prev_torques = np.zeros(12)

        assert self.model.nu == 12, "Expected 12 torque actuators"

        #Air time for feet
        self.air_time = np.array([0.0, 0.0, 0.0, 0.0]) #(FL, FR, RL, RR)
        self.last_contact_state = np.array([0.0, 0.0, 0.0, 0.0])

        #default command velocity
        self.command = np.array([0.6, 0.0, 0.0])

        #phase reward
        self.gait_freq = 1.5
        self.phase = 0.0
        self.phase_offsets = np.array([0.0, np.pi, np.pi, 0.0]) 

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

    def _site_address(self, name):
        site_id = self.model.sensor(name).id
        return self.model.sensor_adr[site_id]

    def _site_velocity(self, site_name):
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

        return np.array([roll, pitch])
    
    def _contact_state(self, fid, idx, update):
        force = self.data.sensordata[fid : fid + 3]
        z_force = force[2]
        if self.prev_contacts[idx]:
            contact = z_force > 2.0   # stay contact
        else:
            contact = z_force > 8.0   # enter contact

        if update:
            self.prev_contacts[idx] = contact
        return contact

    def _foot_height(self, site_name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id][2]
    
    def _get_curriculum(self):
        if self.enable_curriculum:
            t = self.total_timesteps * self.num_envs
            k = 1 - np.cos(np.pi * 1e-8 * t)
        else:
            k = 1
        return k
    
    def _get_obs(self):
        qpos = [self.data.sensordata[self.joint_ids[i]] for i in range(len(self.joint_ids))]
        qvel = [self.data.sensordata[self.joint_vel_ids[i]] for i in range(len(self.joint_vel_ids))]

        base_lin_vel = self.data.sensordata[self.adr_lin_vel : self.adr_lin_vel + 3]
        base_ang_vel = self.data.sensordata[self.adr_ang_vel : self.adr_ang_vel + 3]

        #Contact addresses 
        c_fl = self._contact_state(self.adr_cf_fl, 0, False)
        c_fr = self._contact_state(self.adr_cf_fr, 1, False)
        c_rl = self._contact_state(self.adr_cf_rl, 2, False)
        c_rr = self._contact_state(self.adr_cf_rr, 3, False)

        contacts = np.array([c_fl, c_fr, c_rl, c_rr]).astype(np.float32)

        commands = self.command
    
        projected_gravity = self.data.sensordata[self.adr_imu_ori : self.adr_imu_ori + 4]
        norm_term = 0.0
        for i in range(len(projected_gravity)):
            norm_term += projected_gravity[i]**2
        
        norm_term = norm_term**0.5
        projected_gravity = projected_gravity / norm_term

        w, x, y, z = projected_gravity
        g_x = 2*(x*z - w*y)
        g_y = 2*(y*z + w*x)
        g_z = w**2 - x**2 - y**2 + z**2
        gravity = np.array([g_x, g_y, g_z])

        last_action = self.last_action

        obs = np.concatenate([
            qpos,
            qvel,
            base_ang_vel,
            base_lin_vel,
            gravity,
            last_action,
            contacts,
            commands
        ])

        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0) 
        q_des = self.q_default + action * self.q_range

        self.prev_torques = self.torques.copy()

        for _ in range(20):
            q = [self.data.sensordata[self.joint_ids[i]] for i in range(len(self.joint_ids))]
            qd = [self.data.sensordata[self.joint_vel_ids[i]] for i in range(len(self.joint_vel_ids))]
            self.torques = self.Kp * (q_des - q) - self.Kd * qd
            self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)
            self.data.ctrl[:] = self.torques
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        self.total_timesteps += 1

        self.prev_prev = self.prev_action.copy()

        self.prev_action = self.last_action.copy()

        #torque difference
        torque_diff = np.sqrt(np.mean((self.torques - self.prev_torques)**2))

        # FIX 1: Update action BEFORE observation
        self.last_action = action.copy()
        
        obs = self._get_obs()
        
        # Reward
        reward, r_forward, r_pose, r_smooth, r_ang_vel,r_omega_xy,r_z_vel, r_height, r_clear, r_slip, body_vel, ang_vel = self._compute_reward(action, self.prev_action, self.prev_prev)
        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            print(f"Forward Reward: {r_forward}")
            print(f"Pose Penalty: {r_pose}")
            print(f"Angular(XY) Penalty: {r_omega_xy}")
            print(f"Clearance Penalty: {r_clear}")
            print(f"Slip Penalty: {r_slip}")
            print(f"Height Penalty: {r_height}")
            print(f"Smoothness Penalty: {r_smooth}")
            print(f"Ang Vel Reward: {r_ang_vel}")
            print(f"Z Velocity penalty : {r_z_vel}")
            print("------TOTAL REWARD & Torques------")
            print(reward)
            print(f"Torque Difference: {torque_diff}")
            print("------ANGULAR VELOCITY--")
            print(f"Angular Velocity : {ang_vel[0]}x, {ang_vel[1]}y, {ang_vel[2]}z")
            print("-----BODY VEL AND COMMAND VEL----")
            print(f"body vel: {body_vel[0]}")
            print(f"Command Velocity : {self.command[0]}")
            print("------------------------")
            
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
        self.Kp_leg  = np.random.uniform(23, 27)
        self.Kp_hip = np.random.uniform(28, 32)

        #Random values for Kd in += 0.1 for dyanamic randomization (onlt for post training)
        
        self.Kd_leg = np.random.uniform(0.5, 0.7)
        self.Kd_hip = np.random.uniform(0.7, 0.8)

        self.kp_leg = np.array([self.Kp_leg, self.Kp_leg, self.Kp_hip])
        self.Kp = np.tile(self.kp_leg, 4)
        
        self.kd_leg = np.array([self.Kd_leg, self.Kd_leg, self.Kd_hip])
        self.Kd = np.tile(self.kd_leg, 4)

        #reset phase 
        self.phase = 0.0

        # 1. Reset Internal State
        self.step_count = 0
        self.last_action = np.zeros(12)  # Clear history
        self.prev_action = np.zeros(12)
        self.prev_prev = np.zeros(12)
        self.prev = 0.0
        self.prev_contacts = [False, False, False, False]

        # 2. Randomize Command (Target Velocity)
        if self.total_timesteps * self.num_envs <= 10e6:
            target_vx = np.random.uniform(0.3, 0.5)
        else:
            target_vx = np.random.uniform(0.4, 1.0)
        self.command = np.array([target_vx, 0.0, 0.0]) 

        # 3. Reset Pose (Standing)
        self.data.qpos[7:19] = self.q_default
        self.data.qpos[2] = 0.4  
        self.data.qpos[3:7] = [1, 0, 0, 0] 

        # 4. Add Noise (Domain Randomization)
        self.data.qpos[7:19] += np.random.uniform(-0.05, 0.05, 12)
        self.data.qvel[6:18] = np.random.uniform(-0.5, 0.5, 12) #changed to only the [6:18] and increased the noise from 0.1

        # 5. Stabilize
        mujoco.mj_forward(self.model, self.data)
        
        # 6. Get first observation
        obs = self._get_obs()
        return obs, {}
    
    def _compute_reward(self, action, prev_action, prev_prev):

        q = [self.data.sensordata[self.joint_ids[i]] for i in range(len(self.joint_ids))]
        qd = [self.data.sensordata[self.joint_vel_ids[i]] for i in range(len(self.joint_vel_ids))]

        body_vel = self.data.sensordata[self.adr_lin_vel : self.adr_lin_vel + 3]
        ang_vel = self.data.sensordata[self.adr_ang_vel : self.adr_ang_vel + 3]
        
        forward_vel = body_vel[0]
        self.last_forward_vel = forward_vel

        #FORWARD VELOCITY REWARD
        error = (body_vel[0] - self.command[0])**2 + (body_vel[1] - self.command[1])**2
        r_forward = 5.0 * np.exp(- 5*error)

        #Angular Velocity Tracking 
        error_ang_vel = (ang_vel[2] - self.target_ang_velocity[2])**2
        r_ang_vel = 1.0 * np.exp(- 5*error_ang_vel)

        #Height penalty
        z_length = self.data.qpos[2]
        r_height = - 20* (z_length - 0.28)**2

        #Z velocity Penalty
        z_vel = body_vel[2]
        r_z_vel = -5 * (z_vel - self.command[2])**2

        #Action Smoothness Penalty
        r_smooth = -1.0 * np.sum(np.square(action - prev_action)) 

        #Roll and pitch penalties 
        r_omega_xy = - 0.5 * (ang_vel[0]**2 + ang_vel[1]**2)

        #Joint Pose penalty
        r_pose = -0.5 * np.sum((q - self.q_default)**2)

        #Foot slip penalty
        c_fl = self._contact_state(self.adr_cf_fl, 0, True)
        c_fr = self._contact_state(self.adr_cf_fr, 1, True)
        c_rl = self._contact_state(self.adr_cf_rl, 2, True)
        c_rr = self._contact_state(self.adr_cf_rr, 3, True)

        v_fl = self._site_velocity("FL")
        v_fr = self._site_velocity("FR")
        v_rr = self._site_velocity("RR")
        v_rl = self._site_velocity("RL")

        r_slip = 0.0

        for contact, vel in ((c_fl, v_fl), (c_fr, v_fr), (c_rl, v_rl), (c_rr, v_rr)):
            if contact:
                slip_speed = vel[0]**2 + vel[1]**2
                r_slip -= 0.07 * slip_speed

        #Clearance Penalty
        clearance_error = 0
        contacts = [
            (c_fl, "FL"),
            (c_fr, "FR"),
            (c_rl, "RL"),   
            (c_rr, "RR")
        ]
        
        for contact, name in contacts:
            if not contact:
                height = self._foot_height(name)
                velocity = self._site_velocity(name)
                v_xy = (velocity[0]**2 + velocity[1]**2)**0.5
                clearance_error += (height - self.target_clearance_height)**2 * (v_xy**0.5)

        r_clear = -20.0 * clearance_error 

        
        #Orientation error
        projected_gravity = self.data.sensordata[self.adr_imu_ori : self.adr_imu_ori + 4]
        norm_term = 0.0
        for i in range(len(projected_gravity)):
            norm_term += projected_gravity[i]**2
        
        norm_term = norm_term**0.5
        projected_gravity = projected_gravity / norm_term

        w, x, y, z = projected_gravity
        g_x = 2*(x*z - w*y)
        g_y = 2*(y*z + w*x)
        g_z = w**2 - x**2 - y**2 + z**2
        gravity = np.array([g_x, g_y, g_z])
        
        r_orient = - 0.5 * np.sum((self.target_orientation - gravity)**2)
        
        #energy penalty 
        r_energy = -0.0015 * np.sum(np.abs(self.torques*qd))
        
        penalties = r_smooth + r_pose  + r_omega_xy + r_clear + r_slip + r_energy + r_orient

        total = r_forward + r_ang_vel + r_z_vel + r_height + penalties

        return total, r_forward, r_pose, r_smooth, r_ang_vel,r_omega_xy, r_z_vel, r_height, r_clear, r_slip, body_vel, ang_vel

    def _is_fallen(self):
        z_height = self.data.qpos[2]
        roll, pitch = self._get_rpy()
        terminated = False
        #Terminated if roll > 29 deg or pitch > 34 deg or base height < 0.18m
        if abs(roll) > np.deg2rad(29) or abs(pitch) > np.deg2rad(34) or z_height < 0.18:
            terminated = True
        return terminated

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()
