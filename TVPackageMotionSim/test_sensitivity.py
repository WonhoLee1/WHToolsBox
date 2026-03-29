import mujoco
import numpy as np

def test_damping_ratio_sensitivity(p1_val):
    xml = f"""
    <mujoco>
        <option integrator="implicitfast" timestep="0.001"/>
        <worldbody>
            <light pos="0 0 2" dir="0 0 -1"/>
            <geom name="floor" type="plane" size="1 1 .01" solref="-200000 -0.0001"/>
            <body name="ball" pos="0 0 0.2">
                <freejoint/>
                <geom name="ball_geom" type="sphere" size="0.01" density="7850" solref="{p1_val} -1.0" solimp="1 1 0.0001"/>
            </body>
        </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    
    results = []
    # Test different damping ratios for the fixed p1
    ratios = [0.0001, 1.0, 5.0, 10.0]
    for r in ratios:
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 0.2
        m.geom_solref[1, 1] = -r
        
        max_h = 0
        contact = False
        for _ in range(3000):
            mujoco.mj_step(m, d)
            h = d.qpos[2] - 0.01
            if h < 0.001: contact = True
            if contact and h > max_h: max_h = h
        cor = np.sqrt(max_h / 0.19)
        results.append(cor)
    return results

# Case 1: T=0.01 (p1=-10) - fast contact
print("T=0.01 (p1=-10):", test_damping_ratio_sensitivity(-10))
# Case 2: T=0.05 (p1=-50) - slow contact
print("T=0.05 (p1=-50):", test_damping_ratio_sensitivity(-50))
