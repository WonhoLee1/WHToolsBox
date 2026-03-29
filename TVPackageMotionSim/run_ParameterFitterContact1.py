import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.6f}'.format)
# 11종 공학 재료 표준 물성 라이브러리
# E_gpa: 탄성계수 (Young's Modulus, GPa)
# rho: 밀도 (Density, kg/m^3)
# target_cor: 문헌 기준 표준 반발계수 (Coefficient of Restitution)

MATERIAL_LIBRARY = {
    "Steel": {
        "E_gpa": 200.0, 
        "rho": 7850, 
        "target_cor": 0.92,
        "description": "Hardened Bearing Steel"
    },
    "Glass": {
        "E_gpa": 70.0, 
        "rho": 2500, 
        "target_cor": 0.91,
        "description": "Tempered Glass"
    },
    "Copper": {
        "E_gpa": 120.0, 
        "rho": 8960, 
        "target_cor": 0.60,
        "description": "Pure Annealed Copper"
    },
    "Aluminum": {
        "E_gpa": 70.0, 
        "rho": 2700, 
        "target_cor": 0.78,
        "description": "Aluminum Alloy 6061"
    },
    "Plastic_PC": {
        "E_gpa": 2.3, 
        "rho": 1200, 
        "target_cor": 0.62,
        "description": "Polycarbonate Engineering Plastic"
    },
    "TPU": {
        "E_gpa": 0.1, 
        "rho": 1200, 
        "target_cor": 0.47,
        "description": "Thermoplastic Polyurethane (Elastomer)"
    },
    "Hard_Rubber": {
        "E_gpa": 0.05, 
        "rho": 1100, 
        "target_cor": 0.45,
        "description": "High Durometer Rubber"
    },
    "Soft_Rubber": {
        "E_gpa": 0.005, 
        "rho": 1000, 
        "target_cor": 0.15,
        "description": "Low Durometer Cushion Rubber"
    },
    "Paper": {
        "E_gpa": 0.2, 
        "rho": 800, 
        "target_cor": 0.05,
        "description": "Compressed Paper Stack"
    },
    "EPS_Foam": {
        "E_gpa": 0.01, 
        "rho": 20, 
        "target_cor": 0.10,
        "description": "Expanded Polystyrene Foam"
    },
    "PSA_Adhesive": {
        "E_gpa": 0.001, 
        "rho": 900, 
        "target_cor": 0.01,
        "description": "Pressure Sensitive Adhesive"
    }
}

class MuJoCoFullGeometryOptimizer:
    def __init__(self, integrator_type="implicitfast", time_step_value=0.001, sol_imp_parameter="0.9 0.95 0.001", apply_k_limit=True, e_scaling_coeff=10.0):
        self.integrator_type = integrator_type
        self.time_step_value = time_step_value
        self.sol_imp_parameter = sol_imp_parameter
        self.apply_k_limit = apply_k_limit
        self.e_scaling_coeff = e_scaling_coeff
        self.material_library = {
            "Steel": {"E": 200.0, "rho": 7850}, "Aluminum": {"E": 70.0, "rho": 2700},
            "Copper": {"E": 120.0, "rho": 8960}, "Glass": {"E": 70.0, "rho": 2500},
            "Plastic_PC": {"E": 2.3, "rho": 1200}, "TPU": {"E": 0.1, "rho": 1200},
            "Hard_Rubber": {"E": 0.05, "rho": 1100}, "Soft_Rubber": {"E": 0.005, "rho": 1000},
            "Paper": {"E": 0.2, "rho": 800}, "EPS_Foam": {"E": 0.01, "rho": 20},
            "PSA_Adhesive": {"E": 0.001, "rho": 900}
        }

    def calculate_stiffness_scale(self, material_name, geometry_type, dimension_size):
        young_modulus_pa = self.material_library[material_name]["E"] * 1e9
        eff_mod = young_modulus_pa / (1 - 0.3**2)
        if geometry_type == "sphere":
            return 2 * eff_mod * np.sqrt(dimension_size * (dimension_size * 0.01))
        return eff_mod * (2 * dimension_size[0] if isinstance(dimension_size, list) else dimension_size)

    def calculate_mass(self, material_name, geometry_type, dimension_size):
        density = self.material_library[material_name]["rho"]
        if geometry_type == "sphere":
            volume = (4/3) * np.pi * (dimension_size**3)
        else:
            volume = 8 * dimension_size[0] * dimension_size[1] * dimension_size[2]
        return density * volume

    def run_restitution_simulation(self, test_p2, material_name, geometry_type, dimension_size, drop_height, k_raw, mass):
        # p1 계산 (물리적 강성과 감쇠를 질량으로 정규화하여 MuJoCo 가속도 파라미터로 변환)
        m = max(mass, 1e-9)
        p1 = -(k_raw / m)
        p2 = -(max(test_p2, 0.0) / m)

        # 시스템의 고유 진동수를 바탕으로 시뮬레이션 폭발을 막기 위한 적응형 dt 계산 (마진을 15->30으로 늘려 정밀도 향상)
        omega_n = math.sqrt(k_raw / max(mass, 1e-9))
        period_n = 2 * math.pi / omega_n
        safe_dt = min(self.time_step_value, period_n / 30.0)

        if geometry_type == "sphere":
            z_offset = dimension_size
        else:
            z_offset = dimension_size[2]
        
        solref_plane = "0.001 0.0005" # 바닥면 소량의 감쇠를 추가하여 가벼운 물체의 수치적 불안정성 억제. 
        density = self.material_library[material_name]["rho"]

        xml = f"""
        <mujoco>
            <option integrator="{self.integrator_type}" timestep="{safe_dt:.7f}"/>
            <worldbody>
                <light pos="0 0 2" dir="0 0 -1"/>
                <geom type="plane" size="5 5 0.01" solref="{solref_plane}"/>
                <body name="test_obj" pos="0 0 {drop_height}">
                    <freejoint/>
                    <geom type="{geometry_type if geometry_type=='sphere' else 'box'}" 
                          size="{dimension_size if isinstance(dimension_size, float) else ' '.join(map(str, dimension_size))}" 
                          density="{density}" solref="{p1:.6f} {p2:.6f}" solimp="{self.sol_imp_parameter}"/> 
                </body>
            </worldbody>
        </mujoco>"""
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        max_h = 0.0
        contact = False
        rebounded = False
        # 매우 연질인 경우(EPS_Foam 등) 접촉 시간이 길어지므로 넉넉한 steps 확보
        # K가 낮아질수록 반동 속도가 느려지므로 충분히 기다려야 정점이 잡힙니다.
        total_time = max(2.5, (math.sqrt(drop_height/4.9)*4.0) + (period_n * 20.0))
        steps = int(total_time / safe_dt)

        for _ in range(steps):
            mujoco.mj_step(model, data)
            current_h = data.qpos[2] - z_offset

            if not contact and current_h < 0.002: 
                contact = True
            
            # 접촉 후 최대 높이 기록 및 탈출 로직
            if contact:
                if current_h > max_h: 
                    max_h = current_h
                
                # 반발 후 (튀어오르는 속도가 일정 이상 도달)
                if data.qvel[2] > 0.02:
                    rebounded = True
                    
                # 다시 낙하하기 시작하면 중단(최고점 도달 판단)
                if rebounded and data.qvel[2] < -0.01:
                    break

        return math.sqrt(max(0, max_h) / (drop_height - z_offset))

    def run_optimization_batch(self, geometry_type, dimension_size, drop_height):
        cor_targets = [round(x, 1) for x in np.arange(0.0, 1.1, 0.1)]
        results_data = []

        for mat_name in self.material_library.keys():
            # [사용자 요청] 안정적인 적분을 위해 탄성계수를 스케일링하여 강성을 낮춤
            k_raw = self.calculate_stiffness_scale(mat_name, geometry_type, dimension_size) / self.e_scaling_coeff
            
            rho = self.material_library[mat_name]["rho"]
            mass = self.calculate_mass(mat_name, geometry_type, dimension_size)
            
            # K값 제한 로직 (수치적 안정성을 보장하면서 최대한의 실제 강성 허용)
            if self.apply_k_limit:
                max_allowed_k = mass * (0.25 / (self.time_step_value**2)) / self.e_scaling_coeff
                k_used = min(k_raw, max_allowed_k)
            else:
                k_used = k_raw

            # 임계 감쇠(Critical Damping) 계산: Cc = 2 * sqrt(K * m)
            Cc = 2.0 * math.sqrt(k_used * mass)
            p1_val = -k_used
            
            row = {
                "Material": mat_name, "E (GPa)": self.material_library[mat_name]["E"],
                "Rho (kg/m3)": rho, "Mass (kg)": round(mass, 6), 
                "K_Value": round(k_used, 1), "p1_Value": round(p1_val, 4),
                "Cc_Value": round(Cc, 4)
            }
            
            limit_msg = "(Softened)" if k_used < k_raw else ""
            print(f"\n최적화 중... 재질: {mat_name} (Cc: {Cc:.2f}, K: {k_used:.1e}) {limit_msg}")

            for target in cor_targets:
                if target == 1.0:
                    row[f"COR_{target:.1f}"] = 0.0
                    continue
                
                search_min = 0.0
                search_max = Cc * 2.0
                
                def obj(p2): 
                    return self.run_restitution_simulation(p2, mat_name, geometry_type, dimension_size, drop_height, k_used, mass) - target
                
                try:
                    # 정규화된 댐핑 p2 탐색 (Cc의 100배까지 넓게 탐격)
                    opt_p2 = brentq(obj, search_min, search_max, xtol=1e-4)
                except ValueError:
                    v_min = obj(search_min) + target
                    v_max = obj(search_max) + target
                    opt_p2 = search_min if abs(v_min - target) < abs(v_max - target) else search_max
                    print(f"  [경고] {mat_name} COR={target} 탐색 실패 (범위: {search_min:.1f}~{search_max:.1f}, COR결과: {v_min:.3f}~{v_max:.3f}). {opt_p2:.2f} 로 대체.")

                row[f"COR_{target:.1f}"] = round(opt_p2, 3)
                
            results_data.append(row)
            
        return pd.DataFrame(results_data).set_index("Material")

    def run_restitution_simulation_positive(self, timeconst, dampratio, material_name, geometry_type, dimension_size, drop_height, mass):
        """
        MuJoCo의 양수 입력 방식(timeconst, dampratio)을 사용하여 시뮬레이션을 수행합니다.
        """
        dt = self.time_step_value
        z_offset = dimension_size if isinstance(dimension_size, (float, int)) else dimension_size[2]
        density = self.material_library[material_name]["rho"]

        xml = f"""
        <mujoco>
            <option integrator="{self.integrator_type}" timestep="{dt:.7f}"/>
            <worldbody>
                <light pos="0 0 2" dir="0 0 -1"/>
                <geom type="plane" size="5 5 0.01" solref="0.001 0.0001"/>
                <body name="test_obj" pos="0 0 {drop_height}">
                    <freejoint/>
                    <geom type="{'sphere' if geometry_type=='sphere' else 'box'}" 
                          size="{dimension_size if isinstance(dimension_size, (float, int)) else ' '.join(map(str, dimension_size))}" 
                          density="{density}" solref="{timeconst:.6f} {dampratio:.6f}" solimp="{self.sol_imp_parameter}"/> 
                </body>
            </worldbody>
        </mujoco>"""
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        max_h = 0.0
        contact = False
        rebounded = False
        # 양수 방식은 접촉 시간이 timeconst에 비례함. 충분한 시간 확보.
        total_time = max(2.5, (math.sqrt(drop_height/4.9)*4.0) + (timeconst * 20.0))
        steps = int(total_time / dt)

        for _ in range(steps):
            mujoco.mj_step(model, data)
            current_h = data.qpos[2] - z_offset

            if not contact and current_h < 0.002: 
                contact = True
            
            if contact:
                if current_h > max_h: 
                    max_h = current_h
                if data.qvel[2] > 0.02:
                    rebounded = True
                if rebounded and data.qvel[2] < -0.01:
                    break

        return math.sqrt(max(0, max_h) / (drop_height - z_offset))

    def run_optimization_batch_positive(self, geometry_type, dimension_size, drop_height):
        """
        양수 입력 방식(timeconst, dampratio)에 대한 최적화 배치를 수행합니다.
        timeconst를 각 재질의 물리적 주기(sqrt(m/K))를 기반으로 계산하여 재질별 특성을 반영합니다.
        """
        cor_targets = [round(x, 1) for x in np.arange(0.0, 1.1, 0.1)]
        results_data = []
        
        # 수치적 안전 마진 (적어도 timestep의 2배는 되어야 안정적임)
        min_timeconst = self.time_step_value * 2.0

        for mat_name in self.material_library.keys():
            mass = self.calculate_mass(mat_name, geometry_type, dimension_size)
            rho = self.material_library[mat_name]["rho"]
            k_raw = self.calculate_stiffness_scale(mat_name, geometry_type, dimension_size)
            
            # Hertzian Contact Theory 기반 접촉 시간(Tc) 계산
            v_impact = math.sqrt(2 * 9.81 * drop_height)
            
            # [사용자 요청] 수치적 안정성을 위해 탄성계수를 스케일링하여 강성을 낮춤
            E_val = (self.material_library[mat_name]["E"] * 1e9) / self.e_scaling_coeff  # Pa
            
            nu = 0.3  # Poisson's ratio
            E_eff = E_val / (1 - nu**2)
            R = dimension_size if geometry_type == "sphere" else 0.1 # radius
            
            # Hertz 접촉 시간 공식 (Tc)
            # Tc = 2.9432 * (m^2 / (R * E_eff^2 * v_impact))^(1/5)
            hertz_tc = 2.9432 * ((mass**2) / (R * (E_eff**2) * v_impact))**0.2
            
            # 이론적인 접촉 시간을 timeconst(d)로 사용 (안정성을 위해 Tc 그대로 사용)
            theoretical_d = hertz_tc
            
            # 수치적 안정성을 위해 최소값(Safety Floor) 적용
            actual_d = max(theoretical_d, min_timeconst)
            
            print(f"\n[양수 최적화] 재질: {mat_name} | timeconst: {actual_d:.6f} {'(Safety Floor)' if actual_d == min_timeconst else ''} (Hertz Tc_scaled: {hertz_tc:.6f})")
            row = {
                "Material": mat_name, "Mass (kg)": round(mass, 6),
                "Used_TimeConst": round(actual_d, 6)
            }

            for target in cor_targets:
                if target == 1.0:
                    row[f"COR_{target:.1f}"] = 0.0
                    continue
                
                search_min_r = 0.0
                search_max_r = 50.0  # dampratio 범위를 5.0 -> 50.0으로 대폭 확대
                
                def obj(r):
                    return self.run_restitution_simulation_positive(actual_d, r, mat_name, geometry_type, dimension_size, drop_height, mass) - target
                
                try:
                    # brentq 탐색
                    opt_r = brentq(obj, search_min_r, search_max_r, xtol=1e-4)
                except ValueError:
                    v_min = obj(search_min_r) + target
                    v_max = obj(search_max_r) + target
                    opt_r = search_min_r if abs(v_min - target) < abs(v_max - target) else search_max_r
                    print(f"  [경고] {mat_name} COR={target} 탐색 실패 (범위: {search_min_r:.1f}~{search_max_r:.1f}, COR결과: {v_min:.3f}~{v_max:.3f}). r={opt_r:.2f} 로 대체.")

                row[f"COR_{target:.1f}"] = round(opt_r, 4)
            
            results_data.append(row)
            
        return pd.DataFrame(results_data).set_index("Material")

def create_test_xmls_from_results(df, optimizer, radius=0.1, drop_height=0.2, targets=[0.2, 0.5, 0.8]):
    """최적화 결과 DataFrame과 Optimizer 인스턴스를 사용하여 테스트 XML을 생성합니다."""
    output_dir = r"D:\PythonCodeStudy\mujoco-3.5.0-windows-x86_64\model\WH_boxdrop"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 시각화 색상 설정
    colors = {
        "Steel": "0.3 0.3 0.3 1", "Aluminum": "0.8 0.8 0.8 1", "Copper": "0.8 0.4 0.1 1",
        "Glass": "0.5 0.8 0.9 0.5", "Plastic_PC": "0.1 0.3 0.8 1", "TPU": "0.8 0.1 0.1 1",
        "Hard_Rubber": "0.1 0.1 0.1 1", "Soft_Rubber": "0.2 0.2 0.2 1", "Paper": "0.9 0.9 0.9 1",
        "EPS_Foam": "0.95 0.95 0.8 1", "PSA_Adhesive": "0.9 0.9 0.9 0.5"
    }
    for target in targets:
        col_name = f"COR_{target:.1f}"
        xml_content = f"""<mujoco model="Material_COR_{target:.1f}_Test">
    <option integrator="{optimizer.integrator_type}" timestep="{optimizer.time_step_value}"/>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    </asset>

    <visual>
        <map force="0.1" zfar="30"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <scale forcewidth="0.05" contactwidth="0.1" contactheight="0.05"/>
    </visual>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.05" material="matplane" solref="0.001 0.0005"/>
"""
        for i, (mat_name, row) in enumerate(df.iterrows()):
            mass = row["Mass (kg)"]
            m = max(mass, 1e-9)
            p1 = row["p1_Value"] / m  # 실제 XML 주입 시에만 /mass 처리하여 가속도 공간으로 맵핑
            p2 = -row[col_name] / m
            xml_content += f"""
        <body name="{mat_name}_body" pos="{(i - 5) * 0.3:.3f} 0.0 {drop_height}">
            <freejoint name="{mat_name}_joint"/>
            <geom name="{mat_name}_geom" type="sphere" size="{radius}" density="{row['Rho (kg/m3)']:.1f}" 
                  solref="{p1:.6f} {p2:.6f}" solimp="{optimizer.sol_imp_parameter}" rgba="{colors.get(mat_name, '0.5 0.5 0.5 1')}"/>
        </body>"""
        xml_content += "\n    </worldbody>\n</mujoco>"
        
        file_path = os.path.join(output_dir, f"Material_Test_COR_{target:.1f}.xml")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
    
    print(f"\n[성공] 3종의 테스트용 XML 파일이 생성되었습니다 (음수 방식): {output_dir}")

def create_test_xmls_from_results_positive(df_pos, optimizer, radius=0.1, drop_height=0.2, targets=[0.2, 0.5, 0.8]):
    """양수 최적화 결과 DataFrame을 사용하여 테스트 XML을 생성합니다."""
    output_dir = r"D:\PythonCodeStudy\mujoco-3.5.0-windows-x86_64\model\WH_boxdrop"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {
        "Steel": "0.3 0.3 0.3 1", "Aluminum": "0.8 0.8 0.8 1", "Copper": "0.8 0.4 0.1 1",
        "Glass": "0.5 0.8 0.9 0.5", "Plastic_PC": "0.1 0.3 0.8 1", "TPU": "0.8 0.1 0.1 1",
        "Hard_Rubber": "0.1 0.1 0.1 1", "Soft_Rubber": "0.2 0.2 0.2 1", "Paper": "0.9 0.9 0.9 1",
        "EPS_Foam": "0.95 0.95 0.8 1", "PSA_Adhesive": "0.9 0.9 0.9 0.5"
    }
    
    for target in targets:
        col_name = f"COR_{target:.1f}"
        xml_content = f"""<mujoco model="Material_POS_COR_{target:.1f}_Test">
    <option integrator="{optimizer.integrator_type}" timestep="{optimizer.time_step_value}"/>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    </asset>

    <visual>
        <map force="0.1" zfar="30"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <scale forcewidth="0.05" contactwidth="0.1" contactheight="0.05"/>
    </visual>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.05" material="matplane" solref="0.001 0.0005"/>
"""
        for i, (mat_name, row) in enumerate(df_pos.iterrows()):
            timeconst = row["Used_TimeConst"]
            dampratio = row[col_name]
            xml_content += f"""
        <body name="{mat_name}_body" pos="{(i - 5) * 0.3:.3f} 0.0 {drop_height}">
            <freejoint name="{mat_name}_joint"/>
            <geom name="{mat_name}_geom" type="sphere" size="{radius}" density="{row['Rho (kg/m3)'] if 'Rho (kg/m3)' in row else optimizer.material_library[mat_name]['rho']:.1f}" 
                  solref="{timeconst:.6f} {dampratio:.6f}" solimp="{optimizer.sol_imp_parameter}" rgba="{colors.get(mat_name, '0.5 0.5 0.5 1')}"/>
        </body>"""
        xml_content += "\n    </worldbody>\n</mujoco>"
        
        file_path = os.path.join(output_dir, f"Material_Test_POS_COR_{target:.1f}.xml")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
    
    print(f"\n[성공] 3종의 테스트용 XML 파일이 생성되었습니다 (양수 방식): {output_dir}")

if __name__ == "__main__":
    # 실무적 관점의 타겟 timestep (0.001s) 을 입력받고 강성 스케일링(Softening) 제한 옵션을 켭니다.
    target_dt = float(input("목표 타겟 timestep을 입력하세요 (예: 0.001 또는 0.002): ") or "0.001")
    opt = MuJoCoFullGeometryOptimizer(time_step_value=target_dt, apply_k_limit=True, e_scaling_coeff=100.0)
    res = opt.run_optimization_batch("sphere", 0.1, 1.0)
    cor_cols = [f"COR_{x:.1f}" for x in np.around(np.arange(0.0, 1.1, 0.1), 1)]
    display_cols = ["E (GPa)", "Rho (kg/m3)", "Mass (kg)", "K_Value", "p1_Value", "Cc_Value"] + cor_cols
    print("\n[최적화 결과 테이블]")
    print(res[display_cols].T.to_string())
    res[display_cols].to_csv("sphere_damping_mapping.csv", float_format="%.6f")

    # 2. [솔레프 조합] 절대 힘(Absolute Force) 기준 p1, p2 조합 (시각적 가독성 용이)
    print("\n[solref (p1 - p2) 조합 테이블 (가독성을 위한 순수 Force-Space 기준)]")
    res_solref = res.copy()
    for col in cor_cols:
        res_solref[col] = res.apply(lambda row: f"{row['p1_Value']:.1f} - {-row[col]:.3f}", axis=1)
    print(res_solref[cor_cols].T.to_string())

    # 3. [솔레프 조합] 질량 놈(Mass Norm) 기준 p1/m, p2/m 조합 (실제 MuJoCo 구동용 가속도 공간)
    print("\n[solref (p1/m - p2/m) 조합 테이블 (실제 MuJoCo 맵핑용 /mass 적용)]")
    res_norm = res.copy()
    for col in cor_cols:
        res_norm[col] = res.apply(lambda row: f"{(row['p1_Value']/row['Mass (kg)']):.1f} - {(-row[col]/row['Mass (kg)']):.3f}", axis=1)
    print(res_norm[cor_cols].T.to_string())

    # test xml 생성 (음수 입력 방식)
    create_test_xmls_from_results(res, opt, radius=0.1, drop_height=1.0)

    # 4. [solref 양수 입력] timeconst, dampratio 조합 테이블
    print("\n" + "="*80)
    print("   [양수 입력 방식] solref='timeconst dampratio' 최적화 결과")
    print("="*80)
    res_pos = opt.run_optimization_batch_positive("sphere", 0.1, 1.0)
    
    print("\n[solref (timeconst - dampratio) 조합 테이블 (양수 입력용)]")
    res_pos_table = res_pos.copy()
    for col in cor_cols:
        res_pos_table[col] = res_pos.apply(lambda row: f"{row['Used_TimeConst']:.4f} - {row[col]:.4f}", axis=1)
    print(res_pos_table[cor_cols].T.to_string())

    # 5. 독립 함수 호출하여 XML 생성 (기존 방식 유지)    
    create_test_xmls_from_results_positive(res_pos, opt, radius=0.1, drop_height=1.0)
    