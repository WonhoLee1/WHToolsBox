import mujoco
import numpy as np
xml = """
<mujoco>
    <option integrator="implicitfast" timestep="0.001"/>
    <worldbody>
        <light pos="0 0 2" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.01" solref="-200000 -200"/>
        <body name="obj" pos="0 0 0.2">
            <freejoint/> <geom type="sphere" size="0.01" density="7850" solref="-200000 -0.0001" solimp="0.8 0.95 0.0001"/>
        </body>
    </worldbody>
</mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
max_h = 0
contact = False
for i in range(1200):
    mujoco.mj_step(model, data)
    h = data.qpos[2] - 0.01
    if h < 0.001: contact = True
    if contact and h > max_h: max_h = h
print("Max COR:", np.sqrt(max_h / 0.19))
