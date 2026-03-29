import mujoco
import numpy as np
import matplotlib.pyplot as plt
from run_ParameterFitterContact2 import MuJoCoForceSpaceOptimizer
import os
import math

def check_penetration_depth(mat_name, dimension_size=0.01, drop_height=0.2):
    opt = MuJoCoForceSpaceOptimizer()
    geometry_type = "sphere"
    
    k_real = opt.calculate_physical_stiffness(mat_name, geometry_type, dimension_size)
    mass = opt.calculate_mass(mat_name, geometry_type, dimension_size)
    density = opt.material_library[mat_name]["rho"]
    
    # K값 제한(Softened) 버전 (run_ParameterFitterContact1.py의 로직)
    target_dt = 0.001
    max_allowed_k = mass * ((2 * math.pi) / (15.0 * target_dt)) ** 2
    k_soft = min(k_real, max_allowed_k)
    p1_soft = -k_soft
    
    # 1. Softened 버전 (질량 보정 안됨 or K값이 낮아짐)
    xml_soft = f"""
    <mujoco>
        <option integrator="implicitfast" timestep="{target_dt:.7f}"/>
        <worldbody>
            <light pos="0 0 2" dir="0 0 -1"/>
            <geom type="plane" size="5 5 0.01" solref="0.001 0.05"/>
            <body name="obj" pos="0 0 {drop_height}">
                <freejoint/>
                <geom type="sphere" size="{dimension_size}" density="{density}" 
                      solref="{p1_soft:.1f} 0" solimp="0.8 0.95 0.001" solmix="1000"/> 
            </body>
        </worldbody>
    </mujoco>"""
    
    # 2. Force-Space (질량 보정됨) 버전 (run_ParameterFitterContact2.py의 로직)
    omega_n = math.sqrt(k_real / max(mass, 1e-9))
    safe_dt = min(target_dt, (2 * math.pi / omega_n) / 30.0)
    p1_force = -(k_real / max(mass, 1e-9))
    
    xml_force = f"""
    <mujoco>
        <option integrator="implicitfast" timestep="{safe_dt:.7f}"/>
        <worldbody>
            <light pos="0 0 2" dir="0 0 -1"/>
            <geom type="plane" size="5 5 0.01" solref="0.001 0.05"/>
            <body name="obj" pos="0 0 {drop_height}">
                <freejoint/>
                <geom type="sphere" size="{dimension_size}" density="{density}" 
                      solref="{p1_force:.1f} 0" solimp="0.8 0.95 0.001" solmix="1000"/> 
            </body>
        </worldbody>
    </mujoco>"""
    
    def run_sim(xml, dt):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        heights = []
        times = []
        
        # 0.4초만 시뮬레이션 (낙하 ~ 첫 바운스)
        for i in range(int(0.3 / dt)):
            mujoco.mj_step(model, data)
            current_h = data.qpos[2] - dimension_size
            heights.append(current_h * 1000) # mm 단위
            times.append(i * dt)
            
        return np.array(times), np.array(heights)


    
    t_soft, h_soft = run_sim(xml_soft, target_dt)
    t_force, h_force = run_sim(xml_force, safe_dt)
    
    min_h_soft = np.min(h_soft)
    min_h_force = np.min(h_force)
    
    return t_soft, h_soft, min_h_soft, t_force, h_force, min_h_force

if __name__ == "__main__":
    mats = ["Steel", "EPS_Foam"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, mat in enumerate(mats):
        t_soft, h_soft, min_s, t_force, h_force, min_f = check_penetration_depth(mat)
        
        ax = axes[i]
        
        # 확대 포인트 (충돌 부근만)
        mask_soft = (t_soft > 0.18) & (t_soft < 0.22)
        mask_force = (t_force > 0.18) & (t_force < 0.22)

        ax.plot(t_soft[mask_soft], h_soft[mask_soft], label=f'Softened (No Mass Norm)\nMax Pen: {abs(min_s):.4f} mm', linestyle='--')
        ax.plot(t_force[mask_force], h_force[mask_force], label=f'Force-Space (Mass Norm)\nMax Pen: {abs(min_f):.4f} mm')
        ax.axhline(0, color='r', linestyle=':', alpha=0.5)
        ax.set_title(f'Penetration Depth Comparison: {mat}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Height from floor (mm)')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('penetration_comparison.png')
