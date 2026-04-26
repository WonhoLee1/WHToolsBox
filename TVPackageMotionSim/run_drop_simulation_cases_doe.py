# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulation v6.0 - Autonomous Structural Analysis & Multi-Format Export
MuJoCo 시뮬레이션 결과를 최소한의 정보(마커 궤적)만으로 자율 분석하고,
VTKHDF 및 GLB 포맷으로 결과를 내보내는 차세대 파이프라인입니다.
"""

import os
import sys
import numpy as np
import pickle
import jax.numpy as jnp
from datetime import datetime
from typing import Any

# [WHTOOLS] UTF-8 인코딩 강제 설정 (표준 출력/에러)
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

# [WHTOOLS] 경로 설정
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path: sys.path.append(curr_dir)

from run_drop_simulator import DropSimulator
from run_discrete_builder import get_default_config, get_rgba_by_name, calculate_plate_twist_weld_params
from run_drop_simulator.whts_mapping import get_assembly_data_from_sim
from run_drop_simulator.whts_multipostprocessor_engine import (
    ShellDeformationAnalyzer, 
    PlateAssemblyManager, 
    PlateConfig,
    scale_result_to_mm
)
from run_drop_simulator.whts_exporter import WHToolsExporter
from run_drop_simulator.whts_multipostprocessor_ui import QtVisualizerV2
from PySide6 import QtWidgets
import run_drop_simulator.whts_multipostprocessor_engine as eng
print(f"🔍 [DEBUG] Engine Path: {eng.__file__}", flush=True)

def doe_process_pipeline(modeling_func, enable_post_ui: bool = False):
    """
    DOE (Design of Experiments) 프로세스 파이프라인.
    주어진 모델링 함수를 실행하여 MuJoCo 시뮬레이션을 수행하고, 그 결과를 처리합니다.
    """
    print("\n" + "="*85)
    print(f"🚀 DOE Process Pipeline (v6.0 Autonomous): {modeling_func.__name__}")
    print("="*85)
    
    sim = modeling_func() # modeling_func은 DropSimulator 인스턴스를 반환해야 합니다.
    if sim is None or sim.result is None:
        print("❌ Simulation failed.")
        return

    if sim.config.get("batch_run_save_figures", False) or sim.config.get("batch_run_show_figures", False):
        save_batch_figures(sim)

def save_batch_figures(sim):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import koreanize_matplotlib
    
    cfg = sim.config
    out_dir = str(sim.output_dir) if hasattr(sim, 'output_dir') else cfg.get("output_dir", "doe_results")
    os.makedirs(out_dir, exist_ok=True)
    
    times = np.array(sim.time_history)
    plt.rc('font', size=9)
    
    # 1. Corner Displacement (X, Y, Z)
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    c_pos = np.array(sim.corner_pos_hist) # (N, 8, 3)
    c_disp = c_pos - c_pos[0:1, :, :] # Displacement from initial position
    axes_names = ['X', 'Y', 'Z']
    
    for ax_idx in range(3):
        for i in range(8):
            axs1[ax_idx].plot(times, c_disp[:, i, ax_idx], label=f'C{i+1}')
        axs1[ax_idx].set_title(f'Corner Displacement ({axes_names[ax_idx]})')
        axs1[ax_idx].set_ylabel('Displacement [m]')
        axs1[ax_idx].grid(True)
    axs1[0].legend(loc='upper right', fontsize='small', ncol=4)
    axs1[2].set_xlabel('Time [s]')
    fig1.tight_layout()
    if cfg.get("batch_run_save_figures", False):
        fig1.savefig(os.path.join(out_dir, 'corner_displacement.png'), dpi=150)

    # 2. Corner Velocity (X, Y, Z)
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    c_vel = np.array(sim.corner_vel_hist) # (N, 8, 3)
    for ax_idx in range(3):
        for i in range(8):
            axs2[ax_idx].plot(times, c_vel[:, i, ax_idx], label=f'C{i+1}')
        axs2[ax_idx].set_title(f'Corner Velocity ({axes_names[ax_idx]})')
        axs2[ax_idx].set_ylabel('Velocity [m/s]')
        axs2[ax_idx].grid(True)
    axs2[0].legend(loc='upper right', fontsize='small', ncol=4)
    axs2[2].set_xlabel('Time [s]')
    fig2.tight_layout()
    if cfg.get("batch_run_save_figures", False):
        fig2.savefig(os.path.join(out_dir, 'corner_velocity.png'), dpi=150)
        
    # 3. Forces
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs3[0].plot(times, sim.air_drag_hist, color='blue', label='Drag Force')
    if hasattr(sim, 'air_squeeze_hist'):
        axs3[0].plot(times, sim.air_squeeze_hist, color='cyan', linestyle='--', label='Squeeze Force')
    axs3[0].set_title('Aerodynamic Forces')
    axs3[0].set_ylabel('Force [N]')
    axs3[0].legend(loc='upper right', fontsize='small')
    axs3[0].grid(True)
    
    axs3[1].plot(times, sim.ground_impact_hist, color='red')
    axs3[1].set_title('Ground Contact Force')
    axs3[1].set_ylabel('Force [N]')
    axs3[1].set_xlabel('Time [s]')
    axs3[1].grid(True)
    
    fig3.tight_layout()
    if cfg.get("batch_run_save_figures", False):
        fig3.savefig(os.path.join(out_dir, 'forces.png'), dpi=150)
        
    # 4. Corner Deformation
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 4))
    if hasattr(sim, 'corner_impact_hist') and len(sim.corner_impact_hist) > 0:
        c_impact = np.array(sim.corner_impact_hist)
        for i in range(8):
            ax4.plot(times, c_impact[:, i], label=f'C{i+1}')
        ax4.set_title('Corner Deformation')
        ax4.set_ylabel('Deformation')
        ax4.set_xlabel('Time [s]')
        ax4.legend(loc='upper right', fontsize='small', ncol=4)
        ax4.grid(True)
        fig4.tight_layout()
        if cfg.get("batch_run_save_figures", False):
            fig4.savefig(os.path.join(out_dir, 'corner_deformation.png'), dpi=150)
            
    # 5. Spatial Trajectories (2x2)
    fig5 = plt.figure(figsize=(12, 12))
    ax_xy = fig5.add_subplot(2, 2, 1)
    ax_xz = fig5.add_subplot(2, 2, 2)
    ax_yz = fig5.add_subplot(2, 2, 3)
    ax_3d = fig5.add_subplot(2, 2, 4, projection='3d')
    
    pivot_idx = np.argmin(c_pos[0, :, 2])
    
    for i in range(8):
        lw = 2.5 if i == pivot_idx else 1.0
        alpha = 1.0 if i == pivot_idx else 0.5
        label = f'C{i+1} (Pivot)' if i == pivot_idx else f'C{i+1}'
        
        ax_xy.plot(c_pos[:, i, 0], c_pos[:, i, 1], label=label, linewidth=lw, alpha=alpha)
        ax_xz.plot(c_pos[:, i, 0], c_pos[:, i, 2], label=label, linewidth=lw, alpha=alpha)
        ax_yz.plot(c_pos[:, i, 1], c_pos[:, i, 2], label=label, linewidth=lw, alpha=alpha)
        ax_3d.plot(c_pos[:, i, 0], c_pos[:, i, 1], c_pos[:, i, 2], label=label, linewidth=lw, alpha=alpha)
        
    ax_xy.set_title('XY Plane Trajectory')
    ax_xy.set_xlabel('X [m]')
    ax_xy.set_ylabel('Y [m]')
    ax_xy.grid(True)
    ax_xy.legend(loc='best', fontsize='small')
    
    ax_xz.set_title('XZ Plane Trajectory')
    ax_xz.set_xlabel('X [m]')
    ax_xz.set_ylabel('Z [m]')
    ax_xz.grid(True)
    
    ax_yz.set_title('YZ Plane Trajectory')
    ax_yz.set_xlabel('Y [m]')
    ax_yz.set_ylabel('Z [m]')
    ax_yz.grid(True)
    
    ax_3d.set_title('3D Trajectory (ISO)')
    ax_3d.set_xlabel('X [m]')
    ax_3d.set_ylabel('Y [m]')
    ax_3d.set_zlabel('Z [m]')
    ax_3d.view_init(elev=30, azim=45)
    
    fig5.tight_layout()
    if cfg.get("batch_run_save_figures", False):
        fig5.savefig(os.path.join(out_dir, 'corner_trajectories.png'), dpi=150)
            
    if cfg.get("batch_run_show_figures", False):
        print(f"📊 [WHTOOLS] Opening batch figures. Close windows to continue.")
        plt.show()
    else:
        plt.close('all')

def doe_modeling_case_1_setup(enable_viewer: bool = False, enable_post_ui: bool = False):
    """
    [V5.2.8.4] v4의 test_run_case_1 설정을 100% 계승하고 v5용 옵션을 추가함
    [Case 1] 표준 낙하 테스트 (Golden Case)
    가장 안정적인 물리 계수와 형상 정보를 포함하는 기준 케이스입니다.

    Args:
        enable_viewer (bool): MuJoCo Viewer(GUI) 활성화 여부.
        enable_post_ui (bool): 후처리 대시보드(PySide6) 활성화 여부.
    [V5.2.8.4] v4의 test_run_case_1 설정을 100% 계승하고 v5용 옵션을 추가함
    [Case 1] 표준 낙하 테스트 (Golden Case)
    가장 안정적인 물리 계수와 형상 정보를 포함하는 기준 케이스입니다.

    심플모드로 openchell, chassis 변형 미고려하고, cushion은 이산화 최소화 한다.
    """
    print("\n" + "="*85)
    print("🚀 Running Case 1: Standard Corner 2-3-5 (0.5m)")
    print("="*85)
    
    cfg = get_default_config()
    cfg["use_viewer"] = enable_viewer  # 인터랙티브 모드 활성화 (컨트롤 패널 표시)
    
    # [1. GEOMETRY OPTIONS] : 외관 및 어셈블리 형상 정의
    cfg["box_w"] = 2.056          # 박스 외곽 가로 치수 [m]
    cfg["box_h"] = 1.200          # 박스 외곽 세로 치수 [m]
    cfg["box_d"] = 0.178          # 박스 외곽 깊이 치수 [m]
    cfg["box_thick"] = 0.008      # 박스 골판지(더블 월 등) 두께 [m]
    cfg["assy_w"] = 1.892         # 제품(TV) 어셈블리 가로 [m]
    cfg["assy_h"] = 1.082         # 제품(TV) 어셈블리 세로 [m]
    cfg["cush_gap"] = 0.001       # 쿠션과 제품 사이의 조립 공극(Tolerance) [m]
    # [Geometry]
    cfg["opencell_d"] = 0.012        # Open Cell의 두께 (기본값: 12mm)
    cfg["opencellcoh_d"] = 0.002     # Open Cell Cohesion(Tape 등)의 두께
    cfg["chassis_d"] = 0.035         # Chassis의 두께
    cfg["occ_ithick"] = 0.030        # 인터페이스 두께 관련 변수로 추정
    # [2. DROP ENV] : 낙하 시나리오 및 환경 설정 (use_postprocess_ui는 DropSimulator 내부에서 처리)
    cfg["drop_mode"] = "LTL" # 낙하 테스트 모드 (LTL: Less than Truckload)
    cfg["drop_direction"] = "Corner 2-3-5" # 낙하시 지향 방향 (코너 낙하)
    cfg["drop_height"] = 0.3      # 자유 낙하 높이 [m]

    # [WHTOOLS] PREMIUM VISUALS: Fog & Infinite Ground Effect
    cfg["visual"] = {
        "fogstart": 3.0,              # 3m 지점부터 안개 시작
        "fogend": 10.0,               # 10m 지점에서 완전 안개 (지면 경계 삭제)
        "skybox_rgba": "0.6 0.6 0.6", # 밝은 그레이 배경 (기존보다 밝게)
    }

    # [3. COMPONENTS OPTIONS] : 각 컴포넌트의 설정 (Meshing, Weld, Mass) 통합 관리
    cfg["components"] = {
        "paper"         : {"div": [5, 5, 3], "use_weld": True, "mass": 4.0,  "rgba": get_rgba_by_name("paper", 1.0)},
        "cushion"       : {"div": [5, 5, 3], "use_weld": True, "mass": 3.0,  "rgba": "0.8 0.8 0.8 0.6"},
        "opencell"      : {"div": [4, 4, 1], "use_weld": False, "mass": 5.0,  "rgba": get_rgba_by_name("black", 1.0)},
        "opencellcoh"   : {"div": [4, 4, 1], "use_weld": False, "mass": 0.1,  "rgba": get_rgba_by_name("red", 0.4)},
        "chassis"       : {"div": [4, 4, 1], "use_weld": False, "mass": 10.0, "rgba": "0.0 0.2 0.4 1.0"},
    }
    cfg["include_paperbox"] = False        # 종이 박스 메쉬 모델 활성화

    # [4. CONTACT & PAIR PARAMETERS] : 명시적 접촉 쌍 설정 (A1/A2 통합 점검)
    common_friction = [0.7, 0.7]
    p_solref = [-55000.0,-800.0]
    p_solimp = [0.10, 0.95, 0.02, 0.5, 2]
    cfg["contacts"] = {
        ("ground", "cushion")       : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.02, 0.5, 2]},
        ("ground", "cushion_edge")  : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.02, 0.5, 2]},
        ("ground", "paper")         : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.02, 0.5, 2]},
        ("cushion", "opencell")     : {"friction": common_friction, "solref": p_solref, "solimp": p_solimp},
        ("cushion", "chassis")      : {"friction": common_friction, "solref": p_solref, "solimp": p_solimp},        
        ("cushion", "paper")        : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.02, 0.5, 2]},
    }
    
    # [WHTOOLS] 실제 물성(E, h_real) 기반 인장 강성 산출 + 실험치(f_target) 기반 벤딩 튜닝
    # Chassis: 0.6t, E=170GPa / OpenCell: 1.0t, E=70GPa
    k_oc, d_oc, ts_oc = calculate_plate_twist_weld_params(
        mass=cfg["components"]["opencell"]["mass"],
        width=cfg["assy_w"],
        height=cfg["assy_h"],
        thickness=cfg["opencell_d"],
        div=cfg["components"]["opencell"]["div"],
        E_real=70e9,          # 실제 유리 탄성계수
        real_thickness=0.001, # 실제 유리 두께 (1mm)
        target_freq_hz=1.0,   # 판 전체 목표 벤딩 진동수 (1Hz)
        zeta=0.05
    )
    k_chas, d_chas, ts_chas = calculate_plate_twist_weld_params(
        mass=cfg["components"]["chassis"]["mass"],
        width=cfg["assy_w"],
        height=cfg["assy_h"],
        thickness=cfg["chassis_d"],
        div=cfg["components"]["chassis"]["div"],
        E_real=170e9,         # 실제 Chassis 탄성계수 (Steel계열)
        real_thickness=0.0006, # 실제 Chassis 두께 (0.6mm)
        target_freq_hz=4.0,   # 판 전체 목표 벤딩 진동수 (4Hz)
        zeta=0.05
    )

    cfg["welds"] = {
        "paper"          : {"solref": [0.010, 1.00], "solimp": [0.10, 0.95, 0.01, 0.5, 2]},
        "cushion"        : {"solref": p_solref, "solimp": p_solimp},
        "cushion_corner" : {"solref": p_solref, "solimp": p_solimp},
        "opencell"       : {"solref": [k_oc, d_oc], "solimp": [0.10, 0.95, 0.1, 0.5, 2], "torquescale": ts_oc},
        "opencellcoh"    : {"solref": [-15000.0, -500.0], "solimp": [0.10, 0.95, 0.01, 0.5, 2]},
        "chassis"        : {"solref": [k_chas, d_chas], "solimp": [0.10, 0.99, 0.1, 0.5, 2], "torquescale": ts_chas},
    }
    
    # [5. PLASTICITY & HARDENING]
    cfg["enable_plasticity"]    = True
    cfg["plasticity_ratio"]     = 0.3
    cfg["cush_yield_pressure"]  = 1500.0
    cfg["plastic_hardening_modulus"] = 30000.0
    
    # [6. MASS TOTALS] : (전체 합계: 25.0kg)
    # [6. MASS TOTALS & AUTO BALANCING]
    cfg["components_balance"] = {
        "target_mass": 42.2,
        "target_inertia": [3.0, 8.0, 14.0],
        "target_cog": [0.001, 0.007, 0.010],  # 10cm 편심 배치 시도
        "count": 8
    }
    # analyze_and_balance_components가 실행되면, 위 설정을 바탕으로 aux 질량이 생성되어 component_aux에 추가됩니다.

    # [7. GROUND PROPERTIES]
    # (Unused legacy keys removed)

    # [8. SOLVER & REPORTING OPTIONS]
    cfg["sim_integrator"] = "implicitfast"
    cfg["sim_timestep"]   = 0.0012
    cfg["sim_iterations"] = 50
    cfg["sim_noslip_iterations"] = 0
    cfg["sim_tolerance"]  = 1e-5
    cfg["sim_gravity"]    = [0, 0, -9.81]
    cfg["sim_nthread"]    = 4
    cfg["reporting_interval"] = 0.0024
    cfg["sim_duration"] = 2.0

    # [9. AIR FLUIDICS]
    cfg["enable_air_drag"]    = True
    cfg["enable_air_squeeze"] = True

    # [10. OUTPUT PATH]
    # 결과 저장 루트 폴더. 실제 저장 경로는 "{result_base_dir}/rds-{timestamp}/"가 됩니다.
    cfg["result_base_dir"] = "doe_results"            # 기본값: 실행 폴더 아래 results/D{날짜}_{시간}
    
    # [11. BATCH RUN REPORTING]
    cfg["batch_run_save_figures"] = True
    cfg["batch_run_show_figures"] = True
    
    # [WHTOOLS] 0. 내부 컴포넌트 관성 측정 및 Auto-Balancing 확인
    from run_discrete_builder.whtb_physics import analyze_and_balance_components
    cfg = analyze_and_balance_components(cfg, verbose=True)

    # 4. 시뮬레이션 실행
    sim = DropSimulator(config=cfg)
    sim.simulate()
    return sim

if __name__ == "__main__":
    # [WHTOOLS] 시뮬레이션 및 후처리 UI/Viewer 활성화 여부 설정
    RUN_VIEWER = False # MuJoCo Viewer (GUI) 활성화
    RUN_POST_UI = False # Post-Processing Dashboard (PySide6) 활성화

    # Case 1 기반으로 v6 파이프라인 실행
    doe_process_pipeline(
        lambda: doe_modeling_case_1_setup(enable_viewer=RUN_VIEWER, enable_post_ui=RUN_POST_UI),
        enable_post_ui=RUN_POST_UI
    )
