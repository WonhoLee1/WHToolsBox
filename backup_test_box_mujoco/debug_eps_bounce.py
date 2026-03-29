import mujoco
import numpy as np
import math

def test_eps_foam_bounce():
    material_name = "EPS_Foam"
    geometry_type = "sphere"
    dimension_size = 0.01
    drop_height = 0.2
    
    # Values from the output
    mass = 0.000084
    k_used = 14.7
    p1 = -k_used
    p2 = 0.0
    density = 20.0
    
    safe_dt = 0.001
    z_offset = dimension_size
    solref_plane = "0.001 0"

    xml = f"""
    <mujoco>
        <option integrator="implicitfast" timestep="{safe_dt:.7f}"/>
        <worldbody>
            <light pos="0 0 2" dir="0 0 -1"/>
            <geom type="plane" size="5 5 0.01" solref="{solref_plane}"/>
            <body name="test_obj" pos="0 0 {drop_height}">
                <freejoint/>
                <geom type="sphere" 
                      size="{dimension_size}" 
                      density="{density}" solref="{p1:.1f} {p2:.4f}" solimp="0.8 0.95 0.001"
                      solmix="1000"/> 
            </body>
        </worldbody>
    </mujoco>"""
    
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    max_h = 0.0
    contact = False
    rebounded = False
    steps = 1000

    print(f"Starting simulation: Mass={mass:.6f}, K={k_used:.2f}, dt={safe_dt}")

    for i in range(steps):
        mujoco.mj_step(model, data)
        current_h = data.qpos[2] - z_offset
        
        if i % 50 == 0:
            print(f"Step {i}: h={current_h:.6f}, v={data.qvel[2]:.6f}")

        if not contact and current_h < 0.002: 
            contact = True
            print(f"Contact at step {i}, h={current_h:.6f}")
        
        if contact:
            if current_h > max_h: 
                max_h = current_h
            if data.qvel[2] > 0.05:
                rebounded = True
            if rebounded and data.qvel[2] < -0.01:
                print(f"Rebound finished at step {i}, max_h={max_h:.6f}")
                break

    cor = math.sqrt(max(0, max_h) / (drop_height - z_offset))
    print(f"Final COR: {cor:.3f}")

if __name__ == "__main__":
    test_eps_foam_bounce()
