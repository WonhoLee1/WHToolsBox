import mujoco
import numpy as np

def test_damping_effect_solref_neg_v4():
    # Fixed p1=-10, vary p2 from -1 to -1000
    p2_vals = [-0.1, -1, -5, -10, -50, -100, -500]
    
    results = []
    for p2 in p2_vals:
        xml = f"""
        <mujoco>
            <option integrator="implicitfast" timestep="0.001"/>
            <worldbody>
                <light pos="0 0 2" dir="0 0 -1"/>
                <geom type="plane" size="5 5 0.01" solref="-200000 -0.0001"/>
                <body name="ball" pos="0 0 0.2">
                    <freejoint/>
                    <geom name="ball_geom" type="sphere" size="0.01" density="7850" solref="-10 {p2}" solimp="0.8 0.95 0.001"/>
                </body>
            </worldbody>
        </mujoco>
        """
        try:
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            max_h = 0
            contact = False
            for _ in range(3000):
                mujoco.mj_step(m, d)
                h = d.qpos[2] - 0.01
                if h < 0.002: contact = True
                if contact and h > max_h: max_h = h
            cor = np.sqrt(max_h / 0.19)
            results.append((p2, cor))
        except Exception as e:
            results.append((p2, str(e)))
            
    return results

print("Damping Effect on COR with solref neg (-10 p2):")
for p2, cor in test_damping_effect_solref_neg_v4():
    print(f"p2: {p2}, COR: {cor:.4f}")
