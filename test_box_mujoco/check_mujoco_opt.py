
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco></mujoco>')
print(dir(m.opt))
