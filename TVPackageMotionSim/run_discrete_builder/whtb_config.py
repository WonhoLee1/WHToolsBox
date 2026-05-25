# -*- coding: utf-8 -*-
"""
[WHTOOLS] Configuration & Physics Synchronization Module (v4.5)
'test_run_case_1'의 사양을 기본값으로 100% 반영하며, 지연 동기화(Late-Binding)를 통해
수치적 안정성을 보장합니다.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple

# 현재 config 스키마 버전 — 키 구조가 바뀔 때마다 올림
CONFIG_VERSION = 1

# ─── 버전별 마이그레이션 함수 ────────────────────────────────────────────────
# migrate_v{N}_to_v{N+1}(cfg) 형태로 추가하면 자동으로 체인 적용됨

def _migrate_v0_to_v1(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """v0(버전 필드 없음) → v1: chassis_cog/chassis_moi 키 표준화"""
    # 구버전에서 다른 이름으로 저장됐을 수 있는 키 마이그레이션 예시
    if "cog" in cfg and "chassis_cog" not in cfg:
        cfg["chassis_cog"] = cfg.pop("cog")
    if "moi" in cfg and "chassis_moi" not in cfg:
        cfg["chassis_moi"] = cfg.pop("moi")
    return cfg

_MIGRATIONS: Dict[int, Any] = {
    0: _migrate_v0_to_v1,
}

def _apply_migrations(cfg: Dict[str, Any], from_version: int) -> Dict[str, Any]:
    """from_version부터 CONFIG_VERSION까지 마이그레이션을 순차 적용합니다."""
    v = from_version
    while v < CONFIG_VERSION:
        fn = _MIGRATIONS.get(v)
        if fn:
            cfg = fn(cfg)
        v += 1
    return cfg

# ─── 직렬화 안전 변환 ────────────────────────────────────────────────────────

def _make_serializable(obj: Any) -> Any:
    """numpy / JAX 배열 등 JSON 비직렬화 타입을 Python 기본형으로 변환합니다."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj

# ─── Public API ──────────────────────────────────────────────────────────────

def save_config(cfg: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    현재 config를 버전 메타데이터와 함께 JSON으로 저장합니다.
    numpy 배열 등 비직렬화 타입은 자동 변환합니다.
    """
    meta = {
        "_version": CONFIG_VERSION,
        "_saved_at": datetime.now().isoformat(timespec="seconds"),
        "_app": "WHToolsBox-DropSim",
    }
    payload = {**meta, **_make_serializable(cfg)}
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def load_config(path: Union[str, Path],
                user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    JSON 파일을 읽어 버전 마이그레이션 → 기본값 머지 → 물리 동기화를 수행합니다.

    누락된 키는 get_default_config() 기본값으로 채워지고,
    파일에 없는 신규 키도 자동으로 추가됩니다.

    Args:
        path: JSON 파일 경로
        user_config: 추가로 덮어쓸 설정 (선택)

    Returns:
        완전히 동기화된 config dict
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    # 메타 필드 분리
    file_version = int(raw.get("_version", 0))
    saved = {k: v for k, v in raw.items() if not k.startswith("_")}

    # 버전 마이그레이션
    if file_version < CONFIG_VERSION:
        saved = _apply_migrations(saved, file_version)

    # 기본값으로 시작 → 저장된 값으로 덮어쓰기 → 추가 오버라이드
    merged = _build_default_dict()
    merged.update(saved)
    if user_config:
        merged.update(user_config)

    # 물리 파라미터 동기화
    sync_phys_config(merged)
    return merged

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
    comp = config.get("components", {})
    
    config["mat_paper"] = {
        "rgba": comp.get("paper", {}).get("rgba", "0.5 0.3 0.2 1"),
        "friction": f"{config['paper_friction']}",
        "solref": f"{config['paper_weld_solref']}",
        "solimp": f"{config['paper_weld_solimp']}"
    }

    config["mat_cush"] = {
        "rgba": comp.get("cushion", {}).get("rgba", "0.9 0.9 0.9 0.5"),
        "friction": f"{config['cush_friction']}",
        "solref": f"{config['cush_contact_solref']}",
        "solimp": f"{config['cush_contact_solimp']}",
        "corner_solref": f"{config['cush_corner_solref']}",
        "corner_solimp": f"{config['cush_corner_solimp']}"
    }

    config["mat_cell"] = {
        "rgba": comp.get("opencell", {}).get("rgba", "0.1 0.1 0.1 1.0"),
        "friction": "0.5",
        "solref": f"{config['opencell_weld_solref']}",
        "solimp": f"{config['opencell_weld_solimp']}"
    }

    config["mat_tape"] = {
        "rgba": comp.get("opencellcoh", {}).get("rgba", "1 0.1 0.1 0.4"),
        "friction": "0.8",
        "weld_solref": f"{config['opencell_weld_solref']}",
        "weld_solimp": f"{config['opencell_weld_solimp']}"
    }

    config["mat_tv"] = {
        "rgba": comp.get("chassis", {}).get("rgba", "0.5 0.5 0.5 1.0"),
        "friction": "0.5",
        "weld_solref": f"{config['chassis_weld_solref']}",
        "weld_solimp": f"{config['chassis_weld_solimp']}"
    }

def get_friction_standard(mu: Union[float, List[float], Tuple[float, ...]], dim: int = 5) -> List[float]:
    """
    [WHTOOLS] 마찰 계수를 MuJoCo 표준 차원(dim)에 맞게 정규화합니다.
    
    Args:
        mu (float, list, tuple): 입력 마찰 계수. 단일 값 또는 시퀀스 가능.
        dim (int): 목표 차원 (기본값 5: MuJoCo Sliding2, Torsional1, Rolling2).
        
    Returns:
        List[float]: 규격화된 마찰 계수 리스트.
    """
    if isinstance(mu, (list, tuple)):
        result = [float(x) for x in mu]
    else:
        result = [float(mu)]
    
    # 부족한 차원 보완 (Tangential 2개, Torsional 1개, Rolling 2개 기준)
    # 기본값: [Tangential, Tangential, Torsional(0.005), Rolling(0.0001), Rolling(0.0001)]
    defaults = [0.0, 0.0, 0.005, 0.0001, 0.0001]
    
    # 1. Tangential이 1개만 있으면 복사하여 2개로 확장
    if len(result) == 1:
        result.append(result[0])
        
    # 2. 지정된 dim까지 default로 채움
    while len(result) < dim:
        idx = len(result)
        if idx < len(defaults):
            result.append(defaults[idx])
        else:
            result.append(0.0)
            
    return result[:dim]

def _build_default_dict() -> Dict[str, Any]:
    """sync 없이 순수 기본값 dict만 반환합니다. load_config 내부에서 사용."""
    return {
        # [Geometry]
        "box_w": 1.841, "box_h": 1.103, "box_d": 0.170, "box_thick": 0.008,
        "assy_w": 1.670, "assy_h": 0.960, "cush_gap": 0.005,
        "opencell_d": 0.012, "opencellcoh_d": 0.002, "chassis_d": 0.035, "occ_ithick": 0.030,

        # [Drop Env]
        "drop_mode": "LTL", "drop_direction": "Corner 2-3-5", "drop_height": 0.5,
        "sim_duration": 2.0, "include_paperbox": False, "include_cushion": True,
        "use_postprocess_ui": True, "use_postprocess_v2": False, "use_viewer": True,
        "initial_tilt_deg": 0.0, "initial_tilt_azimuth_deg": 0.0, # 낙하 자세 미세 틸트 (deg)

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
        # Ref. Model에서 가져온 chassis CoG [x,y,z] (m) 및 MoI [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] (kg·m²)
        "chassis_cog": None, "chassis_moi": None,
        "chassis_aux_masses": [
            {"name": "InertiaAux_Single", "size": [0.1, 0.1, 0.1], "mass": 3.9, "pos": [0, 0, 0]}
        ],

        # [Light/Visuals]
        "light_main_diffuse": "0.7 0.7 0.7", "light_main_ambient": "0.3 0.3 0.3",
        "light_sub_diffuse": "0.3 0.3 0.3", "light_head_ambient": "0.1 0.1 0.1", "light_head_diffuse": "0.5 0.5 0.5",

        # [Air Fluidics]
        "air_density": 1.225, "air_viscosity": 1.81e-5, "air_cd_drag": 1.05, "air_cd_viscous": 0.01,
        "air_coef_squeeze": 0.2, "air_squeeze_hmax": 0.20, "air_squeeze_hmin": 0.005,
        "enable_air_drag": True, "enable_air_squeeze": False,
    }

def get_default_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    [WHTOOLS] 'test_run_case_1' 사양을 골자로 하는 기본 설정을 반환합니다.
    """
    config = _build_default_dict()

    # 사용자 설정 덮어쓰기
    if user_config:
        config.update(user_config)

    # [CRITICAL] 물리 파라미터 동기화 (Late-Binding)
    sync_phys_config(config)

    return config
