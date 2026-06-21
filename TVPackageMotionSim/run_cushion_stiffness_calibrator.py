# -*- coding: utf-8 -*-
"""
[WHTOOLS] Cushion Stiffness Calibration Tool (Negative solref Mode)
실제 낙하 시뮬레이션 설정(config.json 등)의 weld solref에 입력할 수 있는 
음수 절대 강성/감쇠 방식(solref="-stiffness -damping")을 적용하여 캘리브레이션을 수행합니다.
각 레이어에 freejoint를 주어 움직임을 보장하되, 측면 가이드 벽으로 좌굴을 방지합니다.
"""

import sys
import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib
    plt.rcParams.update({'font.size': 9})
except ImportError:
    pass

def create_3layer_cushion_xml(weld_stiffness, weld_damping, W, H, D):
    """
    실제 폼 치수 W, H, D를 반영한 3개 층 쿠션 폼 압축 시험 XML을 생성합니다.
    weld constraint의 solref에 음수 절대 강성(-stiffness)과 절대 감쇠(-damping)를 정의합니다.
    각 레이어는 freejoint를 가져 구속되지 않고 움직일 수 있으며, 측면 가이드 벽이 이탈을 막습니다.
    상단 램은 mocap body로 동작하여 kinematic 제어됩니다.
    """
    layer_dz = D / 6.0
    layer_thickness = D / 3.0
    
    xml_str = f"""<mujoco model="cushion_3layer_calibration">
  <compiler angle="degree" coordinate="local"/>
  <option integrator="implicitfast" timestep="0.0001" gravity="0 0 0">
    <flag contact="enable"/>
  </option>
  
  <default>
    <!-- 양단 평면과의 접촉: 침투가 최소화된 높은 강성 접촉 계수 부여 -->
    <geom friction="0.9" solref="0.001 1.0" solimp="0.99 0.999 0.001"/>
  </default>
  
  <worldbody>
    <light pos="0 0 {D*2}" dir="0 0 -1"/>
    
    <!-- 하단 압축 평면 (고정) -->
    <body name="clamp_B" pos="0 0 {-D/2 - 0.01}">
      <geom name="clamp_geom" type="box" size="{W/2} {H/2} 0.01" rgba="1 0 0 0.5" contype="1" conaffinity="1"/>
    </body>
    
    <!-- 쿠션 폼 3개 층 (Z축 슬라이드 조인트로 구속하여 수평 흔들림/회전 차단) -->
    <body name="layer_0" pos="0 0 {-layer_thickness}">
      <joint name="joint_l0" type="slide" axis="0 0 1"/>
      <geom name="layer_0_geom" type="box" size="{W/2} {H/2} {layer_dz}" mass="1.0" rgba="0.8 0.8 0.8 0.6" contype="1" conaffinity="1"/>
    </body>
    
    <body name="layer_1" pos="0 0 0">
      <joint name="joint_l1" type="slide" axis="0 0 1"/>
      <geom name="layer_1_geom" type="box" size="{W/2} {H/2} {layer_dz}" mass="1.0" rgba="0.8 0.8 0.8 0.6" contype="1" conaffinity="1"/>
    </body>
    
    <body name="layer_2" pos="0 0 {layer_thickness}">
      <joint name="joint_l2" type="slide" axis="0 0 1"/>
      <geom name="layer_2_geom" type="box" size="{W/2} {H/2} {layer_dz}" mass="1.0" rgba="0.8 0.8 0.8 0.6" contype="1" conaffinity="1"/>
    </body>
    
    <!-- 상단 압축 평면 (mocap body로 설정하여 변위를 완벽히 kinematic 제어) -->
    <body name="ram" mocap="true" pos="0 0 {D/2 + 0.01}">
      <geom name="ram_geom" type="box" size="{W/2} {H/2} 0.01" rgba="0 1 0 0.5" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  
  <equality>
    <!-- 층간 Weld 연결: solref에 음수 절대 파라미터를 인가 -->
    <weld name="weld_0_1" body1="layer_0" body2="layer_1" solref="-{weld_stiffness} -{weld_damping}" solimp="0.99 0.999 0.001"/>
    <weld name="weld_1_2" body1="layer_1" body2="layer_2" solref="-{weld_stiffness} -{weld_damping}" solimp="0.99 0.999 0.001"/>
  </equality>
  
  <contact>
    <!-- 층간 물리적 접촉을 배제하여 weld constraint에 의해서만 하중이 전달되게 함 -->
    <exclude body1="layer_0" body2="layer_1"/>
    <exclude body1="layer_1" body2="layer_2"/>
  </contact>
</mujoco>
"""
    return xml_str

def run_compression_simulation(xml_str, target_disp, duration=1.0):
    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)
    
    n_steps = int(duration / m.opt.timestep)
    
    disp_hist = []
    force_hist = []
    
    # mocap body ID 및 clamp_geom geom ID 취득
    ram_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ram")
    mocap_id = m.body_mocapid[ram_body_id]
    clamp_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "clamp_geom")
    
    # 초기 Z 위치 기록
    initial_z = d.mocap_pos[mocap_id][2]
    
    mujoco.mj_forward(m, d)
    
    for step in range(n_steps):
        t = d.time
        ratio = 0.5 * (1.0 - np.cos(np.pi * (t / duration)))
        
        # mocap Z 좌표 직접 갱신
        d.mocap_pos[mocap_id][2] = initial_z + target_disp * ratio
        
        try:
            mujoco.mj_step(m, d)
            if np.isnan(d.qpos).any() or np.isnan(d.qvel).any():
                break
        except Exception:
            break
            
        if step % 10 == 0:
            if np.isnan(d.qpos).any():
                break
            actual_disp = abs(d.mocap_pos[mocap_id][2] - initial_z)
            
            # clamp_geom 과 관련된 모든 contact force의 normal force 합산
            act_force = 0.0
            for c_idx in range(d.ncon):
                contact = d.contact[c_idx]
                if contact.geom1 == clamp_geom_id or contact.geom2 == clamp_geom_id:
                    c_force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, c_idx, c_force)
                    act_force += c_force[0]
                    
            disp_hist.append(actual_disp)
            force_hist.append(act_force)
            
    disp_arr = np.array(disp_hist)
    force_arr = np.array(force_hist)
    
    if len(disp_arr) > 0 and (np.isnan(disp_arr).any() or np.isnan(force_arr).any()):
        return np.array([]), np.array([])
        
    if len(force_arr) > 0:
        force_arr = force_arr - force_arr[0]
        disp_arr = disp_arr - disp_arr[0]
        
    return disp_arr, force_arr


def evaluate_stiffness_for_value(k_val, W, H, D):
    # 각 레이어의 질량이 1.0kg이므로 임계 감쇠 c = 2 * sqrt(k * m)
    # 수치적 안정성을 극대화하기 위해 임계 감쇠 기준 적용
    c_val = 2.0 * np.sqrt(k_val)
    xml_str = create_3layer_cushion_xml(k_val, c_val, W, H, D)
    target_disp = -D * 0.1
    disp, force = run_compression_simulation(xml_str, target_disp=target_disp, duration=1.0)
    
    print(f"[DEBUG] k_val={k_val:.1f}, disp_len={len(disp)}, force_len={len(force)}")
    if len(disp) > 0:
        print(f"[DEBUG] disp[0]={disp[0]:.6f}, disp[-1]={disp[-1]:.6f}, force[0]={force[0]:.1f}, force[-1]={force[-1]:.1f}")
        
    if len(disp) < 10 or disp[-1] < 1e-5:
        return 0.0, disp, force
        
    area = W * H
    E_estimated = (force[-1] * D) / (area * disp[-1])
    return E_estimated, disp, force

def main():
    print("=" * 70)
    print(" [WHTOOLS] 3-Layer Cushion Stiffness Calibrator (Negative solref Mode)")
    print("=" * 70)
    
    is_piped = not sys.stdin.isatty()
    
    try:
        if is_piped:
            input_line = sys.stdin.readline().strip()
            parts = [p.strip() for p in input_line.split(",") if p.strip()]
            target_E_mpa = float(parts[0]) if len(parts) > 0 else 4.5
            W = float(parts[1]) if len(parts) > 1 else 1.0
            H = float(parts[2]) if len(parts) > 2 else 1.0
            D = float(parts[3]) if len(parts) > 3 else 1.0
        else:
            user_input = input("Enter target Elastic Modulus (E) in MPa (default 4.5): ").strip()
            target_E_mpa = float(user_input) if user_input else 4.5
            
            w_input = input("Enter cushion Width (W) in meters (default 1.0): ").strip()
            W = float(w_input) if w_input else 1.0
            
            h_input = input("Enter cushion Height (H) in meters (default 1.0): ").strip()
            H = float(h_input) if h_input else 1.0
            
            d_input = input("Enter cushion Thickness (D) in meters (default 1.0): ").strip()
            D = float(d_input) if d_input else 1.0
            
    except ValueError:
        target_E_mpa, W, H, D = 4.5, 1.0, 1.0, 1.0
        
    target_E = target_E_mpa * 1e6
    print(f">> Target Elastic Modulus : {target_E_mpa:.2f} MPa")
    print(f">> Cushion Size (W x H x D): {W:.3f} m x {H:.3f} m x {D:.3f} m")
    print(f">> Cross-section Area (A)  : {W*H:.6f} m^2")
    print(f">> Compression Target (10%): {D*0.1:.4f} m (displacement)")
    
    # 2. 이분법 탐색 구간 초기화 (절대 강성 K의 범위)
    # 3개 레이어의 weld 직렬 연결을 고려하여 k_approx = 2 * (E * A) / D 로 계산
    k_approx = (2.0 * target_E * W * H) / D
    # MuJoCo 수치 강성 척도를 감안하여 탐색 범위를 대폭 확대
    low_k = 10.0
    high_k = k_approx * 10.0
    
    best_k = None
    final_E = 0
    final_disp, final_force = None, None
    
    max_steps = 25
    tolerance_pct = 0.5
    
    print("\n[Bisection Search] Starting search for Optimal Weld stiffness...")
    print("-" * 75)
    print(f"{'Step':^6} | {'Proposed Joint K (N/m)':^24} | {'Est. Modulus (E)':^18} | {'Error (%)':^12}")
    print("-" * 75)
    
    for step in range(max_steps):
        mid_k = (low_k + high_k) / 2.0
        E_est, disp, force = evaluate_stiffness_for_value(mid_k, W, H, D)
        E_mpa = E_est / 1e6
        
        error_pct = abs(E_est - target_E) / target_E * 100.0 if E_est > 0 else 999.0
        print(f" {step+1:02d}   | {mid_k:24,.2f} | {E_mpa:14.4f} MPa | {error_pct:9.2f}%")
        
        if E_est > 0:
            best_k = mid_k
            final_E = E_est
            final_disp = disp
            final_force = force
            
        if error_pct < tolerance_pct and E_est > 0:
            print("-" * 75)
            print("Target reached within acceptable tolerance limit!")
            break
            
        if E_est > target_E:
            high_k = mid_k
        else:
            low_k = mid_k
    else:
        print("-" * 75)
        print("Maximum search steps reached.")
        
    # 실제 입력될 음수 파라미터 값 변환
    best_c = 2.0 * np.sqrt(best_k)
    neg_solref_str = f"-{best_k:.0f} -{best_c:.0f}"
    
    print(f"\nOptimal Weld stiffness (K): {best_k:,.2f} N/m")
    print(f"Calibrated Elastic Modulus (E): {final_E/1e6:.4f} MPa (Error: {abs(final_E - target_E)/target_E*100:.3f}%)")
    print(f"*** Actual input solref value: \"{neg_solref_str}\" ***")
    
    # 3. 결과 그래프 저장
    plt.figure(figsize=(8, 5))
    if final_disp is not None and len(final_disp) > 0:
        plt.plot(final_disp * 100, final_force / 1000, 'b-', linewidth=2.5, label=f"Simulated (K = {best_k:.2e} N/m)")
    
    # 선형 목표선 플롯
    target_strain = np.linspace(0, 0.1, 100)
    target_force_kn = (target_E * (W*H) * target_strain) / 1000.0
    plt.plot(target_strain * D * 100, target_force_kn, 'r--', linewidth=2, label=f"Target E = {target_E_mpa:.2f} MPa")
    
    plt.xlabel("Compression Displacement [cm] (10% max strain)")
    plt.ylabel("Reaction Force [kN]")
    plt.title(f"Cushion Compression Calibration (Validated E = {final_E/1e6:.3f} MPa)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    out_png = "cushion_stiffness_calibration.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f">> Calibration plot saved to: {os.path.abspath(out_png)}")
    
    # 4. 텍스트 보고서 요약
    out_txt = "cushion_stiffness_report.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(" [WHTOOLS] Cushion 3-Layer Stiffness Calibration Report\n")
        f.write("=" * 60 + "\n")
        f.write(f" - Cushion Size (W x H x D)  : {W:.4f} m x {H:.4f} m x {D:.4f} m\n")
        f.write(f" - Target Elastic Modulus (E) : {target_E_mpa:.4f} MPa\n")
        f.write(f" - Calibrated Weld stiffness  : {best_k:.2f} N/m\n")
        f.write(f" - Calibrated Weld damping    : {best_c:.2f} Ns/m\n")
        f.write(f" - Validated Elastic Modulus  : {final_E/1e6:.4f} MPa\n")
        f.write(f" - Validation Error           : {abs(final_E - target_E)/target_E*100:.4f} %\n")
        f.write("-" * 60 + "\n")
        f.write(" Actual XML Weld config line (with negative parameters):\n")
        f.write(f' <weld body1="layer_i" body2="layer_j" solref="{neg_solref_str}" solimp="0.99 0.999 0.001"/>\n')
        f.write("=" * 60 + "\n")
    print(f">> Report summary saved to: {os.path.abspath(out_txt)}")

if __name__ == "__main__":
    main()
