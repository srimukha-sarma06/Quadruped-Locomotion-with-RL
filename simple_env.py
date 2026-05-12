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
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])

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
                                            shape=(53,), dtype=np.float32)
        
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
    
    def _get_obs(self):
        qpos = [self.data.sensordata[self.joint_ids[i]] for i in range(len(self.joint_ids))]
        qvel = [self.data.sensordata[self.joint_vel_ids[i]] for i in range(len(self.joint_vel_ids))]

        base_lin_vel = self.data.sensordata[self.adr_lin_vel : self.adr_lin_vel + 3]
        base_ang_vel = self.data.sensordata[self.adr_ang_vel : self.adr_ang_vel + 3]

        z_height = qpos[2]

        #Contact addresses 
        c_fl = self._contact_state(self.adr_cf_fl, 0, False)
        c_fr = self._contact_state(self.adr_cf_fr, 1, False)
        c_rl = self._contact_state(self.adr_cf_rl, 2, False)
        c_rr = self._contact_state(self.adr_cf_rr, 3, False)

        contacts = np.array([c_fl, c_fr, c_rl, c_rr]).astype(np.float32)

        commands = self.command
    
        projected_gravity = self.data.sensordata[self.adr_imu_ori : self.adr_imu_ori + 4]

        last_action = self.last_action

        obs = np.concatenate([
            qpos,
            qvel,
            z_height,
            base_ang_vel,
            base_lin_vel,
            projected_gravity,
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
        reward, r_forward, r_height,r_energy, body_vel, ang_vel = self._compute_reward(action, self.prev_action, self.prev_prev)
        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            print(f"Forward Reward: {r_forward}")
            print(f"Height Penalty: {r_height}")
            print(f"Energy Penalty: {r_energy}")
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

        #torque tracking

        # 1. Reset Internal State
        self.step_count = 0
        self.last_action = np.zeros(12)  # Clear history
        self.prev_action = np.zeros(12)
        self.prev_prev = np.zeros(12)
        self.prev = 0.0
        self.prev_contacts = [False, False, False, False]

        # 2. Randomize Command (Target Velocity)
        target_vx = np.random.uniform(1.0, 1.1)
        self.command[0] = target_vx

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
        r_forward = -0.55* np.abs(body_vel[0] - self.command[0])

        #Orientation error
        projected_gravity = self.data.sensordata[self.adr_imu_ori : self.adr_imu_ori + 4]

        r_orient = - 1.2 * np.sqrt(np.sum((projected_gravity - self.target_orientation)**2))

        #Alive reward
        r_alive = 0.55* self.command[0]

        #Energy penalty
        r_energy =  - 55e-6 *np.sum(np.square(self.torques * qd))

        #Height penalty
        z_length = self.data.qpos[2]
        r_height = - 0.3* (z_length - 0.28)
        
        penalties =  r_orient + r_energy

        total = r_forward + r_height + r_alive + penalties

        return total, r_forward, r_height,r_energy, body_vel, ang_vel

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
