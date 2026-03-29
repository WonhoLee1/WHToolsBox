import mujoco
import numpy as np

def run_sim(p1, p2, mass, drop_height, dim, dt):
    # 양수 방식 높게 설정
    solref_plane = '0.001 1'
    xml = f"""
    <mujoco>
        <option integrator="implicitfast" timestep="{dt}"/>
        <worldbody>
            <light pos="0 0 2" dir="0 0 -1"/>
            <geom type="plane" size="5 5 0.01" solref="{solref_plane}"/>
            <body name="test_obj" pos="0 0 {drop_height}">
                <freejoint/>
                <geom type="sphere" size="{dim}" density="{mass / ((4/3)*np.pi*(dim**3))}" solref="{p1} {p2}" solimp="0.8 0.95 0.001"/>
            </body>
        </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    max_h = 0.0
    contact = False
    for i in range(int(1.5 / dt)):
        mujoco.mj_step(model, data)
        # print max h logic
        if data.qpos[2] < (dim + 0.005): contact = True
        if contact:
            h = data.qpos[2] - dim
            if h > max_h: max_h = h
    cor = np.sqrt(max(0, max_h) / (drop_height - dim))
    return cor

print("Timestep 0.0001, D 0:", run_sim(-4.39e8, 0, 0.032882, 0.2, 0.01, 0.0001))
print("Timestep 0.0001, D 100:", run_sim(-4.39e8, -100, 0.032882, 0.2, 0.01, 0.0001))
print("Timestep 0.0001, D 7500:", run_sim(-4.39e8, -7500, 0.032882, 0.2, 0.01, 0.0001))
print("Timestep 0.00001, D 0:", run_sim(-4.39e8, 0, 0.032882, 0.2, 0.01, 0.00001))
print("Timestep 0.00001, D 100:", run_sim(-4.39e8, -100, 0.032882, 0.2, 0.01, 0.00001))
