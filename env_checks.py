import mujoco

xml_path = '/media/srimukha-sarma/Windows-SSD/xtr_lair-main/src/robots/m2_metal_description/mujoco/flat_scene.xml'

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(data.contact.friction[:, :])