import mujoco
import numpy as np

def test_damping_effect_solref_neg_v3():
    # Test different T values (time constants)
    T_vals = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    results = []
    for T in T_vals:
        p1 = -T / 0.001
        # Use p2=-1 (ratio mode, critical damping)
        xml = f"""
        <mujoco>
            <option integrator="implicitfast" timestep="0.001"/>
            <worldbody>
                <light pos="0 0 2" dir="0 0 -1"/>
                <geom type="plane" size="5 5 0.01" solref="-200000 -0.0001"/>
                <body name="ball" pos="0 0 0.2">
                    <freejoint/>
                    <geom name="ball_geom" type="sphere" size="0.01" density="7850" solref="{p1} -1" solimp="0.8 0.95 0.001"/>
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
            results.append((T, cor))
        except Exception as e:
            results.append((T, str(e)))
            
    return results

print("Damping Effect on COR with solref neg (-T/dt -1):")
for T, cor in test_damping_effect_solref_neg_v3():
    print(f"T: {T}, COR: {cor:.4f}")
