import mujoco
import numpy as np

xml = """
<mujoco>
    <option integrator="implicitfast" timestep="0.001"/>
    <worldbody>
        <geom name="floor" type="plane" size="0 0 .01" solref="-200000 -0.0001"/>
        <body name="ball" pos="0 0 0.2">
            <freejoint/>
            <geom name="ball_geom" type="sphere" size="0.01" density="7850" solref="-200000 -10" solimp="0.8 0.95 0.0001"/>
        </body>
    </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

# Test different damping values for same K/m
dampings = [1, 10, 100, 500, 1000]
for c in dampings:
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.2
    # m.geom_solref[1, 1] is the damping parameter
    m.geom_solref[1, 1] = -c
    
    max_h = 0
    contact = False
    for _ in range(2000):
        mujoco.mj_step(m, d)
        h = d.qpos[2] - 0.01
        if h < 0.001: contact = True
        if contact and h > max_h: max_h = h
    cor = np.sqrt(max_h / 0.19)
    print(f"Damping: {c}, COR: {cor:.4f}")
