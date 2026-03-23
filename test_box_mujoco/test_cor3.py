import mujoco
import numpy as np

def run_sim(p1, p2, dt):
    xml = f"""
    <mujoco>
        <option integrator="implicitfast" timestep="{dt}"/>
        <worldbody>
            <light pos="0 0 2" dir="0 0 -1"/>
            <geom type="plane" size="5 5 0.01" solref="0.001 0"/>
            <body name="test_obj" pos="0 0 0.5">
                <freejoint/>
                <geom type="sphere" size="0.01" density="7850" solref="{p1} {p2}" solimp="0.8 0.95 0.001" solmix="1000"/>
            </body>
        </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    max_h = 0.0
    contact = False
    rebounded = False
    for i in range(int(1.0 / dt)):
        mujoco.mj_step(model, data)
        h = data.qpos[2] - 0.01
        if not contact and h < 0.002: contact = True
        if contact:
            if h > max_h: max_h = h
            if data.qvel[2] > 0.05: rebounded = True
            if rebounded and data.qvel[2] < -0.01: break
    return np.sqrt(max(0, max_h) / 0.49)

print("0.000003 D=0 K=4.39e8:", run_sim(-4.39e8, 0, 0.000003))
print("0.000003 D=38000 K=4.39e8:", run_sim(-4.39e8, -38000, 0.000003))
print("0.001 D=0 K=4e4:", run_sim(-4.39e4, 0, 0.001))
print("0.001 D=400 K=4e4:", run_sim(-4.39e4, -400, 0.001))
print("0.001 D=0 K=4e8 (FAIL):", run_sim(-4.39e8, 0, 0.001))
