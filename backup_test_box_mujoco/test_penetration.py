import mujoco
import numpy as np

xml = """
<mujoco>
    <option integrator="implicitfast" timestep="0.001"/>
    <worldbody>
        <light pos="0 0 2" dir="0 0 -1"/>
        <geom type="plane" size="5 5 0.01" solref="1000000 0.0001"/>
        <body name="ball" pos="0 0 0.2">
            <freejoint/>
            <geom name="ball_geom" type="sphere" size="0.01" density="7850" solref="439560439.6 0.0001" solimp="1 1 0.0001"/>
        </body>
    </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

for _ in range(200):
    mujoco.mj_step(m, d)
    if d.qpos[2] < 0.011:
        print(f"Step: {_}, Z: {d.qpos[2]}, Contact force: {d.contact.dist if d.ncon > 0 else 'N/A'}")
