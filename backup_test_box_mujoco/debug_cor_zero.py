import mujoco
import numpy as np

def test_damping_effect():
    # Steel sphere parameters from the failed run
    k = 4.395604e+08
    mass = 0.032882
    rho = 7850
    size = 0.01
    
    results = []
    # Test damping from 0 to something significant
    # Critical damping C = 2 * sqrt(K*m) approx 2 * sqrt(4.4e8 * 0.03) approx 2 * 3633 approx 7266
    dampings = [0.0, 100, 1000, 5000, 10000, 50000]
    
    for c in dampings:
        xml = f"""
        <mujoco>
            <option integrator="implicitfast" timestep="0.001"/>
            <worldbody>
                <light pos="0 0 2" dir="0 0 -1"/>
                <geom type="plane" size="5 5 0.01" solref="1000000 0.0001"/>
                <body name="ball" pos="0 0 0.2">
                    <freejoint/>
                    <geom name="ball_geom" type="sphere" size="{size}" density="{rho}" solref="{k} {max(c, 0.0001)}" solimp="1 1 0.0001"/>
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
            results.append((c, cor))
        except Exception as e:
            results.append((c, str(e)))
            
    return results

print("Damping Effect on COR for Steel Sphere (K=4.4e8):")
for c, cor in test_damping_effect():
    print(f"C: {c}, COR: {cor}")
