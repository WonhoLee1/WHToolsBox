import mujoco
import numpy as np

def test_damping_effect_scaled():
    # Scale K down to MuJoCo friendly range
    k = 1000000
    mass = 0.03
    rho = 7850
    size = 0.01
    
    results = []
    # Test damping
    # Critical damping C = 2 * sqrt(1e6 * 0.03) = 2 * 173 = 346
    dampings = [0.0, 10, 50, 100, 200, 500]
    
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

print("Damping Effect on COR for Steel Sphere (K=1e6):")
for c, cor in test_damping_effect_scaled():
    print(f"C: {c}, COR: {cor}")
