import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('/media/srimukha-sarma/Windows-SSD/xtr_lair-main/src/robots/m2_metal_description/mujoco/flat_scene.xml')
data = mujoco.MjData(model)

joint_vel_start = model.sensor("FL_hip_velocity_sensor").id
joint_vel_end = model.sensor("RR_calf_velocity_sensor").id

for i in range(model.ngeom):
    print(i, model.geom(i).name)

for i in range(data.ncon):
    # Note that the contact array has more than `ncon` entries,
    # so be careful to only read the valid entries.
    contact = data.contact[i]
    print('contact', i)
    print('dist', contact.dist)
    print('geom1', contact.geom1, model.geom_id2name(contact.geom1))
    print('geom2', contact.geom2, model.geom_id2name(contact.geom2))
    # There's more stuff in the data structure
    # See the mujoco documentation for more info!
    geom2_body = model.geom_bodyid[data.contact[i].geom2]
    print(' Contact force on geom2 body', data.cfrc_ext[geom2_body])
    print('norm', np.sqrt(np.sum(np.square(data.cfrc_ext[geom2_body]))))
    # Use internal functions to read out mj_contactForce
    c_array = np.zeros(6, dtype=np.float64)
    print('c_array', c_array)
    mujoco.functions.mj_contactForce(model, data, i, c_array)
    print('c_array', c_array)