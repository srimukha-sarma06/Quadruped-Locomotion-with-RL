import mujoco

model = mujoco.MjModel.from_xml_path('/media/srimukha-sarma/Windows-SSD/xtr_lair-main/src/robots/m2_metal_description/mujoco/flat_scene.xml')
data = mujoco.MjData(model)

joint_vel_start = model.sensor("FL_hip_velocity_sensor").id
joint_vel_end = model.sensor("RR_calf_velocity_sensor").id

joint_vel = data.sensordata[joint_vel_start : joint_vel_end]

print(joint_vel)