# -*- coding: utf-8 -*-
"""
[WHTOOLS] Configuration & Physics Synchronization Module (v4.5)
'test_run_case_1'의 사양을 기본값으로 100% 반영하며, 지연 동기화(Late-Binding)를 통해
수치적 안정성을 보장합니다.
"""

from typing import Any, Dict, Optional

def sync_phys_config(config: Dict[str, Any]):
    """
    [CRITICAL] 루트 레벨의 물리 상수를 시뮬레이션용 복합 맵(solref, mat_*)으로 동기화합니다.
    사용자 오버라이드 이후 호출되어 'Stale' 데이터 문제를 방지합니다.
    """
    # 1. 네이밍 호환성 레이어 (Legacy -> Standard)
    mapping = {
        "oc_div": "opencell_div",
        "oc_use_weld": "opencell_use_weld",
        "occ_div": "opencellcoh_div",
        "occ_use_weld": "opencellcoh_use_weld",
        "chas_div": "chassis_div",
        "chas_use_weld": "chassis_use_weld",
        "oc_d": "opencell_d",
        "occ_d": "opencellcoh_d",
        "chas_d": "chassis_d"
    }
    for old_key, new_key in mapping.items():
        if old_key in config and new_key not in config:
            config[new_key] = config[old_key]
        elif new_key in config and old_key not in config:
            config[old_key] = config[new_key]

    # 2. Solver 문자열 조립 (XML 빌더용)
    # Chassis (TV)
    config["chassis_weld_solref"] = f"{config['chassis_weld_solref_timec']} {config['chassis_weld_solref_damprr']}"
    config["chassis_weld_solimp"] = f"{config['chassis_weld_solimp_pos']} {config['chassis_weld_solimp_width']} {config['chassis_weld_solimp_mid']} {config['chassis_weld_solimp_low']} {config['chassis_weld_solimp_high']}"
    
    # Open Cell
    config["opencell_weld_solref"] = f"{config['opencell_weld_solref_timec']} {config['opencell_weld_solref_damprr']}"
    config["opencell_weld_solimp"] = f"{config['opencell_weld_solimp_pos']} {config['opencell_weld_solimp_width']} {config['opencell_weld_solimp_mid']} {config['opencell_weld_solimp_low']} {config['opencell_weld_solimp_high']}"
    
    # Cushion Weld
    config["cush_weld_solref"] = f"{config['cush_weld_solref_timec']} {config['cush_weld_solref_damprr']}"
    config["cush_weld_corner_solref"] = f"{config['cush_weld_corner_solref_timec']} {config['cush_weld_corner_solref_damprr']}"

    # paper weld
    config["paper_weld_solref"] = f"{config['paper_solref_timec']} {config['paper_solref_damprr']}"
    
    # 3. Material Maps (Builder 호환용)
    config["mat_paper"] = {
        "rgba": "0.5 0.3 0.2 1",
        "friction": f"{config['paper_friction']}",
        "solref": f"{config['paper_weld_solref']}",
        "solimp": f"{config['paper_weld_solimp']}"
    }

    config["mat_cush"] = {
        "rgba": "1 1 1 0.4",
        "friction": f"{config['cush_friction']}",
        "solref": f"{config['cush_contact_solref']}",
        "solimp": f"{config['cush_contact_solimp']}",
        "corner_solref": f"{config['cush_corner_solref']}",
        "corner_solimp": f"{config['cush_corner_solimp']}"
    }

    config["mat_cell"] = {
        "rgba": "0.1 0.1 0.1 1",
        "friction": "0.5",
        "solref": f"{config['opencell_weld_solref']}",
        "solimp": f"{config['opencell_weld_solimp']}"
    }

    config["mat_tape"] = {
        "rgba": "1 0.1 0.1 0.4",
        "friction": "0.8",
        "weld_solref": f"{config['opencell_weld_solref']}",
        "weld_solimp": f"{config['opencell_weld_solimp']}"
    }

    config["mat_tv"] = {
        "rgba": "0.1 0.1 0.1 1",
        "friction": "0.5",
        "weld_solref": f"{config['chassis_weld_solref']}",
        "weld_solimp": f"{config['chassis_weld_solimp']}"
    }

def get_default_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    [WHTOOLS] 'test_run_case_1' 사양을 골자로 하는 기본 설정을 반환합니다.
    """
    # 1. 초기 기본값 설정 (Case 1 100% 반영)
    config = {
        # [Geometry]
        "box_w": 1.841, "box_h": 1.103, "box_d": 0.170, "box_thick": 0.008,
        "assy_w": 1.670, "assy_h": 0.960, "cush_gap": 0.005,
        "opencell_d": 0.012, "opencellcoh_d": 0.002, "chassis_d": 0.035, "occ_ithick": 0.030,

        # [Drop Env]
        "drop_mode": "LTL", "drop_direction": "Corner 2-3-5", "drop_height": 0.5,
        "sim_duration": 2.0, "include_paperbox": False, "include_cushion": True,
        "use_postprocess_ui": True, "use_postprocess_v2": False, "use_viewer": True,

        # [Meshing]
        "chassis_div": [5, 5, 1], "chassis_use_weld": True,
        "opencell_div": [5, 5, 1], "opencell_use_weld": True,
        "opencellcoh_div": [5, 5, 1], "opencellcoh_use_weld": True,
        "cush_div": [5, 5, 3], "cush_use_weld": True,
        "box_div": [5, 5, 2], "box_use_weld": False,

        # [Solver Specs]
        "sim_integrator": "implicitfast", "sim_timestep": 0.0012, "sim_iterations": 50,
        "sim_noslip_iterations": 0, "sim_tolerance": 1e-5, "sim_impratio": 1.0,
        "sim_gravity": [0, 0, -9.81],

        # [Weld Physics Constants]
        "cush_weld_solref_timec": 0.008, "cush_weld_solref_damprr": 0.8,
        "cush_weld_corner_solref_timec": 0.008, "cush_weld_corner_solref_damprr": 0.8,
        "opencell_weld_solref_timec": 0.005, "opencell_weld_solref_damprr": 0.5,
        "chassis_weld_solref_timec": 0.002, "chassis_weld_solref_damprr": 0.5,
        "chassis_weld_solimp_pos": 0.1, "chassis_weld_solimp_width": 0.95, "chassis_weld_solimp_mid": 0.005, "chassis_weld_solimp_low": 0.5, "chassis_weld_solimp_high": 2.0,
        "opencell_weld_solimp_pos": 0.1, "opencell_weld_solimp_width": 0.95, "opencell_weld_solimp_mid": 0.005, "opencell_weld_solimp_low": 0.5, "opencell_weld_solimp_high": 2.0,
        "paper_solref_timec": 0.01, "paper_solref_damprr": 0.8,
        "paper_weld_solimp":"0.1 0.95 0.005 0.5 2",

        # [Contact Specs]
        "cush_friction": 0.8, "paper_friction": 0.8, "ground_friction": 1.0,
        "cush_contact_solref": "0.01 0.8", "cush_contact_solimp": "0.1 0.95 0.005 0.5 2",
        "cush_corner_solref": "0.01 0.8", "cush_corner_solimp": "0.1 0.95 0.005 0.5 2",
        "ground_solref": "0.002 0.001", "ground_solimp": "0.9 0.99 0.001",
        

        # [Plasticity]
        "enable_plasticity": True, "plasticity_ratio": 0.5, "cush_yield_pressure": 1000.0,
        "plastic_hardening_modulus": 2000.0, "plastic_color_limit": 0.08, "plastic_max_strain": 0.5,
        "debug_plasticity": False,

        # [Mass Totals]
        "mass_paper": 4.0, "mass_cushion": 2.0, "mass_oc": 5.0, "mass_occ": 0.1, "mass_chassis": 10.0,
        "target_mass": 25.0, "enable_target_balancing": True, "num_balancing_masses": 8,
        "chassis_aux_masses": [
            {"name": "InertiaAux_Single", "size": [0.1, 0.1, 0.1], "mass": 3.9, "pos": [0, 0, 0]}
        ],

        # [Light/Visuals]
        "light_main_diffuse": "0.8 0.8 0.8", "light_main_ambient": "0.3 0.3 0.3",
        "light_sub_diffuse": "0.4 0.4 0.4", "light_head_ambient": "0.4 0.4 0.4", "light_head_diffuse": "0.8 0.8 0.8",

        # [Air Fluidics]
        "air_density": 1.225, "air_viscosity": 1.81e-5, "air_cd_drag": 1.05, "air_cd_viscous": 0.01,
        "air_coef_squeeze": 0.2, "air_squeeze_hmax": 0.20, "air_squeeze_hmin": 0.005,
        "enable_air_drag": True, "enable_air_squeeze": False,
    }

    # 2. 사용자 설정 덮어쓰기
    if user_config:
        config.update(user_config)

    # 3. [CRITICAL] 물리 파라미터 동기화 (Late-Binding) 
    sync_phys_config(config)

    return config
