import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="1" density="1"/></worldbody></mujoco>')
print(m.geom_solref[0])
