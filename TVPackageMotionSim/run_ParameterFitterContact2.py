import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math

# 출력 포맷 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.6f}'.format)

class MuJoCoForceSpaceOptimizer:
    """
    재료의 절대적 물성(영률, 밀도)을 유지하면서 
    질량 변화에 따른 물리적 정합성을 확보하는 Force-Space 최적화 클래스입니다.
    """

    def __init__(self, integrator_type="implicitfast", time_step_value=0.001, sol_imp_parameter="0.8 0.95 0.001"):
        self.integrator_type = integrator_type
        self.time_step_value = time_step_value
        self.sol_imp_parameter = sol_imp_parameter
        
        # 11종 공학 재료 데이터 (E: GPa, rho: kg/m^3)
        self.material_library = {
            "Steel": {"E": 200.0, "rho": 7850}, "Aluminum": {"E": 70.0, "rho": 2700},
            "Copper": {"E": 120.0, "rho": 8960}, "Glass": {"E": 70.0, "rho": 2500},
            "Plastic_PC": {"E": 2.3, "rho": 1200}, "TPU": {"E": 0.1, "rho": 1200},
            "Hard_Rubber": {"E": 0.05, "rho": 1100}, "Soft_Rubber": {"E": 0.005, "rho": 1000},
            "Paper": {"E": 0.2, "rho": 800}, "EPS_Foam": {"E": 0.01, "rho": 20},
            "PSA_Adhesive": {"E": 0.001, "rho": 900}
        }

    def calculate_physical_stiffness(self, material_name, geometry_type, dimension_size):
        """현실의 힘(Force) 단위 강성 K (N/m)를 계산합니다."""
        young_modulus_pa = self.material_library[material_name]["E"] * 1e9
        effective_modulus = young_modulus_pa / (1 - 0.3**2)
        
        if geometry_type == "sphere":
            # Hertzian Contact 등가 강성: K = 2 * E* * sqrt(R * d_ref)
            return 2 * effective_modulus * np.sqrt(dimension_size * (dimension_size * 0.01))
        
        # 박스류: K = E * A / L (단순화)
        return effective_modulus * (2 * dimension_size[0] if isinstance(dimension_size, list) else dimension_size)

    def calculate_mass(self, material_name, geometry_type, dimension_size):
        """물체의 기하학적 형상에 따른 질량(kg)을 계산합니다."""
        density = self.material_library[material_name]["rho"]
        if geometry_type == "sphere":
            volume = (4/3) * np.pi * (dimension_size**3)
        else:
            volume = 8 * dimension_size[0] * dimension_size[1] * dimension_size[2]
        return density * volume

    def run_simulation(self, test_damping_real, material_name, geometry_type, dimension_size, drop_height, k_real, mass):
        """
        [핵심] 물리적 강성과 감쇠를 질량으로 나누어 MuJoCo 가속도 파라미터로 변환합니다.
        """
        # MuJoCo 주입값 p1, p2 (Acceleration Space로 변환)
        p1 = -(k_real / max(mass, 1e-9))
        p2 = -(test_damping_real / max(mass, 1e-9))

        # 적응형 타임스텝 (고유진동수 기준 30배 정밀도)
        omega_n = math.sqrt(k_real / max(mass, 1e-9))
        safe_dt = min(self.time_step_value, (2 * math.pi / omega_n) / 30.0)

        z_offset = dimension_size if geometry_type == "sphere" else dimension_size[2]
        
        xml = f"""
        <mujoco>
            <option integrator="{self.integrator_type}" timestep="{safe_dt:.7f}"/>
            <worldbody>
                <light pos="0 0 2" dir="0 0 -1"/>
                <geom type="plane" size="5 5 0.01" solref="0.001 0.05"/>
                <body name="obj" pos="0 0 {drop_height}">
                    <freejoint/>
                    <geom type="{'sphere' if geometry_type=='sphere' else 'box'}" 
                          size="{dimension_size if isinstance(dimension_size, float) else ' '.join(map(str, dimension_size))}" 
                          density="{self.material_library[material_name]['rho']}" 
                          solref="{p1:.1f} {p2:.6f}" solimp="{self.sol_imp_parameter}" solmix="1000"/> 
                </body>
            </worldbody>
        </mujoco>"""
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        max_h, contact = 0.0, False
        
        for _ in range(int(1.2 / safe_dt)):
            mujoco.mj_step(model, data)
            current_h = data.qpos[2] - z_offset
            if not contact and current_h < 0.002: contact = True
            if contact:
                if current_h > max_h: max_h = current_h
                if data.qvel[2] < -0.01 and max_h > 0.001: break # 최고점 도달 후 하강 시 중단

        return math.sqrt(max(0, max_h) / (drop_height - z_offset))

    def optimize_batch(self, geometry_type, dimension_size, drop_height):
        cor_targets = np.around(np.arange(0.1, 1.0, 0.1), 1)
        results = []

        for mat in self.material_library.keys():
            k_real = self.calculate_physical_stiffness(mat, geometry_type, dimension_size)
            mass = self.calculate_mass(mat, geometry_type, dimension_size)
            
            # 임계 감쇠 (현실의 힘 기준): Cc = 2 * sqrt(K * m)
            Cc_real = 2.0 * math.sqrt(k_real * mass)
            
            print(f"\n[최적화] 재질: {mat} | K_real: {k_real:.1e} | Mass: {mass:.2e} kg")
            row = {"Material": mat, "K_real": k_real, "Mass": mass, "Cc_real": Cc_real}

            for target in cor_targets:
                def obj(c_real): 
                    return self.run_simulation(c_real, mat, geometry_type, dimension_size, drop_height, k_real, mass) - target
                
                try:
                    # 현실의 댐핑 계수 C_real을 직접 탐색 (0 ~ Cc_real의 20배)
                    opt_c_real = brentq(obj, 0, Cc_real * 20, xtol=1e-4)
                    row[f"COR_{target:.1f}"] = round(opt_c_real, 4)
                except ValueError:
                    row[f"COR_{target:.1f}"] = 0.0
            
            results.append(row)
        
        return pd.DataFrame(results).set_index("Material")

# ==========================================
# 실행 및 결과 분석
# ==========================================
if __name__ == "__main__":
    opt = MuJoCoForceSpaceOptimizer()
    
    # 2cm 구(r=0.01), 20cm 낙하
    final_df = opt.optimize_batch("sphere", 0.01, 0.2)
    
    print("\n" + "="*80)
    print("   [최종 결과] 현실의 힘(Force) 기준 재질별 최적 Damping (C_real) 테이블")
    print("="*80)
    print(final_df.to_string())
