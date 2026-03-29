import mujoco
import numpy as np

def test_damping_effect_solref_neg_v2():
    # Use neg solref as intended by MuJoCo
    k_mujoco = 1000000 
    rho = 7850
    size = 0.01
    
    results = []
    # Test damping ratio
    ratios = [0.0001, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    for r in ratios:
        xml = f"""
        <mujoco>
            <option integrator="implicitfast" timestep="0.001"/>
            <worldbody>
                <light pos="0 0 2" dir="0 0 -1"/>
                <geom type="plane" size="5 5 0.01" solref="-200000 -0.0001"/>
                <body name="ball" pos="0 0 0.2">
                    <freejoint/>
                    <geom name="ball_geom" type="sphere" size="{size}" density="{rho}" solref="-20 -{r}" solimp="1 1 0.0001"/>
                </body>
            </worldbody>
        </mujoco>
        """
        try:
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            
            max_h = 0
            contact = False
            for _ in range(2000):
                mujoco.mj_step(m, d)
                h = d.qpos[2] - 0.01
                if h < 0.002: contact = True
                if contact and h > max_h: max_h = h
            cor = np.sqrt(max_h / 0.19)
            results.append((r, cor))
        except Exception as e:
            results.append((r, str(e)))
            
    return results

print("Damping Effect on COR with solref neg (-20 -ratio):")
for r, cor in test_damping_effect_solref_neg_v2():
    print(f"Ratio: {r}, COR: {cor}")
