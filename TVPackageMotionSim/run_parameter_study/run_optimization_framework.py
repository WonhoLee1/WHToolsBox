# -*- coding: utf-8 -*-
"""
[WHTOOLS] Optimization & DOE Framework Main Launcher (v1.0)
이 스크립트는 최적화 및 DOE 프레임워크를 구동하는 메인 진입점입니다.
기본 config.json 설정 파일이 존재하지 않는 경우 자동으로 default 설정을 생성하여 셋업을 완료합니다.
"""

import os
import sys
import json
from pathlib import Path

# UTF-8 인코딩 강제
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

# [WHTOOLS] 경로 설정 보완 (run_parameter_study용)
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)  # TVPackageMotionSim
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
grandparent_dir = os.path.dirname(parent_dir)  # WHToolsBox
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

from run_discrete_builder.whtb_config import get_default_config, save_config
from whts_optimization_ui import main as ui_main


def ensure_default_config():
    """
    TVPackageMotionSim 폴더 내에 config.json 설정 파일이 존재하지 않는 경우
    v6.0 test_case_1_setup 설정을 바탕으로 config.json 파일을 자동 생성합니다.
    """
    config_path = Path(parent_dir) / "config.json"
    if not config_path.exists():
        print(f"🔍 [WHTOOLS Launcher] 'config.json' not found at {config_path}. Generating default config...", flush=True)
        try:
            # 기본 설정 로드
            default_cfg = get_default_config()
            
            # [1. GEOMETRY OPTIONS]
            default_cfg["box_w"] = 2.056
            default_cfg["box_h"] = 1.200
            default_cfg["box_d"] = 0.178
            default_cfg["box_thick"] = 0.008
            default_cfg["assy_w"] = 1.892
            default_cfg["assy_h"] = 1.082
            default_cfg["cush_gap"] = 0.0001
            default_cfg["opencell_d"] = 0.012
            default_cfg["opencellcoh_d"] = 0.002
            default_cfg["chassis_d"] = 0.035
            default_cfg["occ_ithick"] = 0.030

            # [2. DROP ENV]
            default_cfg["drop_mode"] = "LTL"
            default_cfg["drop_direction"] = "Corner 2-3-5"
            default_cfg["drop_height"] = 0.3
            default_cfg["use_postprocess_ui"] = False
            default_cfg["use_viewer"] = True
            default_cfg["initial_tilt_deg"] = 0.0
            default_cfg["initial_tilt_azimuth_deg"] = 0.0

            # [8. SOLVER & REPORTING OPTIONS]
            default_cfg["sim_integrator"] = "implicitfast"
            default_cfg["sim_timestep"]   = 0.0010
            default_cfg["sim_iterations"] = 50
            default_cfg["sim_noslip_iterations"] = 0
            default_cfg["sim_tolerance"]  = 1e-5
            default_cfg["sim_gravity"]    = [0, 0, -9.81]
            default_cfg["sim_nthread"]    = 4
            default_cfg["reporting_interval"] = 0.0020
            default_cfg["sim_duration"] = 1.0  # 배치 해석 시간을 단축하기 위해 1.0초 기본 세팅

            # [9. AIR FLUIDICS]
            default_cfg["enable_air_drag"]    = True
            default_cfg["enable_air_squeeze"] = True

            # [PREMIUM VISUALS]
            default_cfg["visual"] = {
                "fogstart": 3.0,
                "fogend": 10.0,
                "skybox_rgba": "0.6 0.6 0.6",
            }

            # [3. COMPONENTS OPTIONS]
            from run_discrete_builder import get_rgba_by_name
            default_cfg["components"] = {
                "paper"         : {"div": [5, 5, 3], "use_weld": True, "mass": 4.0,  "rgba": get_rgba_by_name("paper", 1.0)},
                "cushion"       : {"div": [5, 5, 3], "use_weld": True, "mass": 3.0,  "rgba": "0.8 0.8 0.8 0.6"},
                "opencell"      : {"div": [4, 4, 1], "use_weld": True, "mass": 5.0,  "rgba": get_rgba_by_name("black", 1.0)},
                "opencellcoh"   : {"div": [4, 4, 1], "use_weld": True, "mass": 0.1,  "rgba": get_rgba_by_name("red", 0.4), "enable_btm_weld": False},
                "chassis"       : {"div": [4, 4, 1], "use_weld": True, "mass": 10.0, "rgba": "0.0 0.2 0.4 1.0"},
            }
            default_cfg["include_paperbox"] = False

            # [4. CONTACT & PAIR PARAMETERS]
            common_friction = [0.3, 0.3]
            p_solref = [-25000.0, -200.0]
            p_solimp = [0.90, 0.95, 0.001, 0.5, 2]
            default_cfg["contacts"] = {
                "ground,cushion"       : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": p_solimp},
                "ground,cushion_edge"  : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": p_solimp},
                "ground,paper"         : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": p_solimp},
                "cushion,opencell"     : {"friction": common_friction, "solref": p_solref, "solimp": p_solimp},
                "cushion_edge,opencell": {"friction": common_friction, "solref": p_solref, "solimp": p_solimp},
                "cushion,chassis"      : {"friction": common_friction, "solref": p_solref, "solimp": p_solimp},
                "cushion_edge,chassis" : {"friction": common_friction, "solref": p_solref, "solimp": p_solimp},
                "cushion,paper"        : {"friction": common_friction, "solref": [0.001, 1.0], "solimp": p_solimp},
            }

            # [WELD PARAMETERS CALCULATION]
            from run_discrete_builder import calculate_plate_twist_weld_params
            k_oc, d_oc, ts_oc = calculate_plate_twist_weld_params(
                mass=default_cfg["components"]["opencell"]["mass"],
                width=default_cfg["assy_w"],
                height=default_cfg["assy_h"],
                thickness=default_cfg["opencell_d"],
                div=default_cfg["components"]["opencell"]["div"],
                E_real=70e9,
                real_thickness=0.001,
                target_freq_hz=1.0,
                zeta=0.05
            )
            k_chas, d_chas, ts_chas = calculate_plate_twist_weld_params(
                mass=default_cfg["components"]["chassis"]["mass"],
                width=default_cfg["assy_w"],
                height=default_cfg["assy_h"],
                thickness=default_cfg["chassis_d"],
                div=default_cfg["components"]["chassis"]["div"],
                E_real=170e9,
                real_thickness=0.0006,
                target_freq_hz=4.0,
                zeta=0.05
            )

            default_cfg["welds"] = {
                "paper"          : {"solref": [0.010, 1.00], "solimp": [0.10, 0.95, 0.01, 0.5, 2]},
                "cushion"        : {"solref": p_solref, "solimp": p_solimp},
                "cushion_corner" : {"solref": p_solref, "solimp": p_solimp},
                "opencell"       : {"solref": [k_oc, d_oc], "solimp": [0.10, 0.95, 0.1, 0.5, 2], "torquescale": ts_oc},
                "opencellcoh"    : {"solref": [-50000.0, -500.0], "solimp": [0.10, 0.95, 0.005, 0.5, 2]},
                "chassis"        : {"solref": [k_chas, d_chas], "solimp": [0.10, 0.99, 0.1, 0.5, 2], "torquescale": ts_chas},
                "auxboxmass"     : {"solref": [0.001, 1.0], "solimp": [0.1, 0.95, 0.001, 0.5, 2], "torquescale": 100.0},
            }

            # [5. PLASTICITY & HARDENING]
            default_cfg["enable_plasticity"]    = True
            default_cfg["plasticity_ratio"]     = 0.3
            default_cfg["cush_yield_pressure"]  = 3500.0
            default_cfg["plastic_hardening_modulus"] = 30000.0

            # [6. MASS TOTALS & INERTIA CORRECTION]
            default_cfg["components_balance"] = {
                "target_mass"   : 42.2,
                "target_inertia": [5.013, 14.005, 18.8, 0.289, -0.0073, -0.0252],   # 6성분 규격 완벽 대응
                "target_cog"    : [0.0034228, -0.01196665, 0.0059899],
            }

            # 물리 파라미터 동기화 및 밸런싱 수행
            from run_discrete_builder.whtb_physics import analyze_and_balance_components
            default_cfg = analyze_and_balance_components(default_cfg, verbose=False)

            save_config(default_cfg, config_path)
            print(f"✅ Created default configuration (v6 test_case_1): {config_path}", flush=True)
        except Exception as e:
            print(f"❌ Failed to create default config: {e}", flush=True)


if __name__ == "__main__":
    # 1. config.json 존재유무 파악 및 자동 생성
    ensure_default_config()

    # 2. whts_optimization_ui 메인 기동 (Gooey 구동)
    ui_main()
