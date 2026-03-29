import mujoco
import numpy as np

with open("stiffness_BOpenCell_TWIST.xml", "r", encoding="utf-8") as f:
    xml = f.read()

xml = xml.replace('kp="200000000" kv="100000"', 'kp="2000000" kv="1000"')

with open("temp_twist.xml", "w", encoding="utf-8") as f:
    f.write(xml)

m = mujoco.MjModel.from_xml_path("temp_twist.xml")
d = mujoco.MjData(m)
act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ram")
mujoco.mj_forward(m, d)

hist = []
for step in range(3000):
    t = d.time
    # scale target
    d.ctrl[act_id] = np.radians(30.0 * 0.5 * (1.0 - np.cos(np.pi * (t / 3.0))))
    mujoco.mj_step(m, d)
    if step % 100 == 0:
        val = d.actuator_force[act_id]
        if isinstance(val, np.ndarray): val = val[0]
        hist.append(float(val))

for i in range(10):
    print(f"step {i*100}, force={hist[i]:.4f}")
