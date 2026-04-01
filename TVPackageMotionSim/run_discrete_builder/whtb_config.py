from typing import Optional, Dict, Any

def get_default_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    MuJoCo 이산형 낙하 모델링을 위한 기본 파라미터 설정을 생성하고 사용자 설정과 병합합니다.
    
    이 함수는 시뮬레이션 환경, 물리 엔진 옵션, 부품별 기하학적 치수 및 재질 물성치를 
    통합적으로 관리하는 'Source of Truth' 역할을 수행합니다. 모든 수치는 SI 단위(m, kg, s, N)를 기준합니다.

    Args:
        user_config (dict, optional): 사용자가 오버라이드하고자 하는 설정값 딕셔너리. 
                                     기본값(None)일 경우 내부의 디폴트 값을 사용합니다.

    Returns:
        dict: 모든 계산과 병합이 완료된 최종 설정 딕셔너리.
    """
    if user_config is None:
        user_config = {}
        
    # -----------------------------------------------------------------
    # [0] 낙하 모드 및 방향 설정 (Scenario Settings)
    # -----------------------------------------------------------------
    # drop_mode: 'PARCEL' (택배 낙하), 'LTL' (대형 화물), 'CUSTOM' 등
    drop_mode = user_config.get("drop_mode", "PARCEL")
    # drop_direction: 'front', 'bottom', 'top-left-corner' 등 (whtb_utils.parse_drop_target 참고)
    drop_direction = user_config.get("drop_direction", "front")
    
    # -----------------------------------------------------------------
    # [1] 시뮬레이션/솔버 기본 파라미터 (MuJoCo Solver Options)
    # -----------------------------------------------------------------
    sim_integrator = user_config.get("sim_integrator", "implicitfast") # 솔버 알고리즘 (안정성 위주)
    sim_timestep   = user_config.get("sim_timestep", 0.001)           # 시뮬레이션 타임스텝 (초)
    sim_iterations = user_config.get("sim_iterations", 80)            # 최대 컨택 반복 횟수
    sim_noslip_iterations = user_config.get("sim_noslip_iterations", 5) # 슬립 방지용 반복 횟수
    sim_tolerance  = user_config.get("sim_tolerance", 1e-6)           # 솔버 수렴 오차 한계
    sim_impratio   = user_config.get("sim_impratio", 1.0)             # 충격량 비중
    sim_gravity    = user_config.get("sim_gravity", [0, 0, -9.81])    # 중력 가속도 (m/s^2)
    sim_nthread    = user_config.get("sim_nthread", 4)                # 멀티스레딩 지원 코어 수

    # -----------------------------------------------------------------
    # [2] 부품별 기본 물리 파라미터 (Material Physics)
    # -----------------------------------------------------------------
    # solref/solimp 파라미터 가이드:
    # - solref [timeconst, dampratio]: 작을수록 딱딱하고 에너지가 덜 빠짐.
    # - solimp [dmin, dmax, width, mid, power]: 접촉력의 임피던스 곡선을 결정.
    
    # Cushion (완충재: EPS/Foam)
    cush_solref_timec = user_config.get("cush_solref_timec", 0.03) # 완충재 유연성 (0.03~0.05 권장)
    cush_solref_damprr = user_config.get("cush_solref_damprr", 0.8)   # 완충재 감쇠비 (0.8~1.0 권장)
    cush_solimp_dmin = user_config.get("cush_solimp_dmin", 0.1)
    cush_solimp_dmax = user_config.get("cush_solimp_dmax", 0.95)
    cush_solimp_width = user_config.get("cush_solimp_width", 0.005)
    cush_solimp_mid = user_config.get("cush_solimp_mid", 0.5)
    cush_solimp_power = user_config.get("cush_solimp_power", 2.0)
    cush_yield_stress = user_config.get("cush_yield_stress", 0.1) # MPa 단위 항복 강도 (소성 변형 시작점)
    
    # Tape (부착용 테이프: Cohesive/Adhesive)
    tape_solref_timec = user_config.get("tape_solref_timec", 0.01) # 테이프의 인장 강성 관련
    tape_solref_damprr = user_config.get("tape_solref_damprr", 1.0)
    tape_solimp_dmin = user_config.get("tape_solimp_dmin", 0.1)
    tape_solimp_dmax = user_config.get("tape_solimp_dmax", 0.99)
    tape_solimp_width = user_config.get("tape_solimp_width", 0.001)
    tape_solimp_mid = user_config.get("tape_solimp_mid", 0.5)
    tape_solimp_power = user_config.get("tape_solimp_power", 2.0)
    
    # Cell (TV 패널부)
    cell_solref_timec = user_config.get("cell_solref_timec", 0.01) # 고강성 세라믹/글라스 특성
    cell_solref_damprr = user_config.get("cell_solref_damprr", 0.3)
    cell_solimp_dmin = user_config.get("cell_solimp_dmin", 0.5)
    cell_solimp_dmax = user_config.get("cell_solimp_dmax", 0.95)
    cell_solimp_width = user_config.get("cell_solimp_width", 0.001)
    cell_solimp_mid = user_config.get("cell_solimp_mid", 0.5)
    cell_solimp_power = user_config.get("cell_solimp_power", 2.0)
    
    # TV Chassis (기구물/프레임)
    tv_solref_timec = user_config.get("tv_solref_timec", 0.002) # 매우 높은 금속 강성
    tv_solref_damprr = user_config.get("tv_solref_damprr", 0.5)
    tv_solimp_dmin = user_config.get("tv_solimp_dmin", 0.1)
    tv_solimp_dmax = user_config.get("tv_solimp_dmax", 0.95)
    tv_solimp_width = user_config.get("tv_solimp_width", 0.005)
    tv_solimp_mid = user_config.get("tv_solimp_mid", 0.5)
    tv_solimp_power = user_config.get("tv_solimp_power", 2.0)
    
    # Ground (바닥 접촉면)
    ground_solref_timec = user_config.get("ground_solref_timec", 0.001)
    ground_solref_damprr = user_config.get("ground_solref_damprr", 0.0001)
    ground_solref = f"{ground_solref_timec} {ground_solref_damprr}"
    if "ground_solref" in user_config and "ground_solref_timec" not in user_config and "ground_solref_damprr" not in user_config:
        ground_solref = user_config["ground_solref"]

    # -----------------------------------------------------------------
    # [3] 내부 문자열 조립 (String Formatting for MuJoCo XML)
    # -----------------------------------------------------------------
    cush_solref = f"{cush_solref_timec} {cush_solref_damprr}"
    tape_solref = f"{tape_solref_timec} {tape_solref_damprr}"
    cell_solref = f"{cell_solref_timec} {cell_solref_damprr}"
    tv_solref = f"{tv_solref_timec} {tv_solref_damprr}"
    
    cush_solimp = f"{cush_solimp_dmin} {cush_solimp_dmax} {cush_solimp_width} {cush_solimp_mid} {cush_solimp_power}"
    tape_solimp = f"{tape_solimp_dmin} {tape_solimp_dmax} {tape_solimp_width} {tape_solimp_mid} {tape_solimp_power}"
    cell_solimp = f"{cell_solimp_dmin} {cell_solimp_dmax} {cell_solimp_width} {cell_solimp_mid} {cell_solimp_power}"
    tv_solimp = f"{tv_solimp_dmin} {tv_solimp_dmax} {tv_solimp_width} {tv_solimp_mid} {tv_solimp_power}"

    # -----------------------------------------------------------------
    # [4] Weld(용접) 및 Contact(접촉) 파라미터 분리 적용
    # -----------------------------------------------------------------
    # Weld: 동일 부품 혹은 연결된 부품 간의 고정 구속조건 성능
    cush_weld_solref = f"{user_config.get('cush_weld_solref_timec', cush_solref_timec)} {user_config.get('cush_weld_solref_damprr', cush_solref_damprr)}"
    tape_weld_solref = f"{user_config.get('tape_weld_solref_timec', tape_solref_timec)} {user_config.get('tape_weld_solref_damprr', tape_solref_damprr)}"
    
    # OpenCell 및 Chassis는 명시적 명칭(opencell, chassis)을 우선 확인
    oc_w_s = user_config.get('opencell_weld_solref_timec', user_config.get('cell_weld_solref_timec', cell_solref_timec))
    oc_w_d = user_config.get('opencell_weld_solref_damprr', user_config.get('cell_weld_solref_damprr', cell_solref_damprr))
    cell_weld_solref = f"{oc_w_s} {oc_w_d}"
    
    ch_w_s = user_config.get('chassis_weld_solref_timec', user_config.get('tv_weld_solref_timec', tv_solref_timec))
    ch_w_d = user_config.get('chassis_weld_solref_damprr', user_config.get('tv_weld_solref_damprr', tv_solref_damprr))
    tv_weld_solref = f"{ch_w_s} {ch_w_d}"

    cush_weld_solimp = cush_solimp 
    tape_weld_solimp = tape_solimp
    cell_weld_solimp = cell_solimp
    tv_weld_solimp = tv_solimp

    # Contact: 다른 바디와 부딪힐 때 적용되는 성능
    cush_contact_solref = user_config.get("cush_contact_solref", cush_solref)
    cush_contact_solimp = user_config.get("cush_contact_solimp", cush_solimp)
    cush_corner_solref = user_config.get("cush_corner_solref", ground_solref)
    cush_corner_solimp = user_config.get("cush_corner_solimp", "0.95 0.999 0.001 0.5 2.0")
    
    # 특수 목적: 코너/에지 부위의 국부적 충격 제어용
    cush_weld_corner_solref_timec = user_config.get("cush_weld_corner_solref_timec", None)
    cush_weld_corner_solref_damprr = user_config.get("cush_weld_corner_solref_damprr", 1.0)
    cush_weld_corner_solref = None
    if cush_weld_corner_solref_timec is not None:
        cush_weld_corner_solref = f"{cush_weld_corner_solref_timec} {cush_weld_corner_solref_damprr}"
    
    # -----------------------------------------------------------------
    # [5] 주요 기하학적 치수 및 질량 (Dimensions & Mass)
    # -----------------------------------------------------------------
    # 외부 박스 치수
    box_w = user_config.get("box_w", 2.0) 
    box_h = user_config.get("box_h", 1.4)
    box_d = user_config.get("box_d", 0.25)
    box_thick = user_config.get("box_thick", 0.01) # 박스 골판지 두께
    
    # 박스 내경에 맞춘 완충재 초기 크기
    cush_w = box_w - 2 * box_thick
    cush_h = box_h - 2 * box_thick
    cush_d = box_d - 2 * box_thick

    # 내부 조립체(Assy: 패널+프레임 등)의 외곽 크기
    assy_w = user_config.get("assy_w", cush_w - 0.3)
    assy_h = user_config.get("assy_h", cush_h - 0.3)
    
    # 완충재 질량 계산 (밀도가 입력된 경우 체적 비례 산출)
    cush_density = user_config.get("cush_density", None)
    mass_cushion = user_config.get("mass_cushion", 1.0)
    if cush_density is not None:
        if user_config.get("unit_size") is not None: 
            us = user_config.get("unit_size")
            mass_cushion = cush_density * (us[0] * us[1] * us[2])
        else: 
            assy_d = 0.020 + 0.005 + 0.050 # 내부 Assy 합계 두께 가정
            ext_vol = cush_w * cush_h * cush_d
            int_vol = assy_w * assy_h * assy_d
            cush_vol = max(0.01, ext_vol - int_vol)
            mass_cushion = cush_density * cush_vol
    
    # -----------------------------------------------------------------
    # [6] 최종 설정 딕셔너리 생성 (Assembly)
    # -----------------------------------------------------------------
    config = {
        "drop_mode": drop_mode,
        "drop_direction": drop_direction,
        "drop_height": user_config.get("drop_height", 0.5), # 낙하 높이 (m)
        "include_paperbox": True,
        "include_cushion": True,
        "sim_duration": user_config.get("sim_duration", 2.0),
        
        "box_w": box_w, "box_h": box_h, "box_d": box_d,
        "box_thick": box_thick, "cush_gap": 0.001,
        "assy_w": assy_w, "assy_h": assy_h,
        "oc_d"      : 0.020, "occ_d"     : 0.005, "chas_d"    : 0.050, "occ_ithick": 0.050,

        # 격자 분할 수 (이산화 해상도)
        "box_div": user_config.get("box_div", [5, 4, 3]),
        "cush_div": user_config.get("cush_div", [5, 4, 3]),
        "oc_div": user_config.get("oc_div", [5, 4, 1]),
        "occ_div": user_config.get("occ_div", [5, 4, 1]),
        "chassis_div": user_config.get("chassis_div", [5, 4, 1]),
        "assy_div": user_config.get("assy_div", [5, 4, 1]), 

        # 구속조건(Weld) 사용 스위치 (False로 설정 시 접촉만으로 시뮬레이션: Zero-Weld 모드)
        "box_use_weld": user_config.get("box_use_weld", True),
        "cush_use_weld": user_config.get("cush_use_weld", True),
        "oc_use_weld": user_config.get("oc_use_weld", True),
        "occ_use_weld": user_config.get("occ_use_weld", True),
        "chassis_use_weld": user_config.get("chassis_use_weld", True),
        
        "print_corner_plasticity": user_config.get("print_corner_plasticity", False),

        # 부품별 질량 (단위: kg)
        "mass_paper": 4.0, "mass_cushion": mass_cushion,
        "mass_oc": 5.0, "mass_occ": 0.1, "mass_chassis": 10.0,

        "cush_weld_solref": cush_weld_solref, "cush_weld_solimp": cush_weld_solimp,
        "cush_contact_solref": cush_contact_solref, "cush_contact_solimp": cush_contact_solimp,
        "cush_corner_solref": cush_corner_solref, "cush_corner_solimp": cush_corner_solimp,
        "cush_weld_corner_solref": cush_weld_corner_solref,

        # 재료 데이터 매핑 (XML <default> 태그용)
        "mat_paper": {"rgba": "0.7 0.6 0.4 0.9", "solref": "0.01 1.0", "solimp": "0.1 0.95 0.005 0.5 2", "contype": "1", "conaffinity": "1", "friction": "0.8"},
        "mat_cush" : {
            "rgba": "0.9 0.9 0.9 0.5", 
            "weld_solref": cush_weld_solref, "weld_solimp": cush_weld_solimp, 
            "solref": cush_contact_solref, "solimp": cush_contact_solimp,
            "corner_solref": cush_corner_solref, "corner_solimp": cush_corner_solimp,
            "contype": "1", "conaffinity": "1", "friction": user_config.get("cush_friction", 0.8)
        },
        "mat_tape" : {
            "rgba": "1.0 0.1 0.1 0.8", "weld_solref": tape_weld_solref, "weld_solimp": tape_weld_solimp, 
            "solref": tape_solref, "solimp": tape_solimp, "contype": "1", "conaffinity": "1", "friction": "0.8"
        },
        "mat_cell" : {
            "rgba": "0.1 0.1 0.1 1.0", "weld_solref": cell_weld_solref, "weld_solimp": cell_weld_solimp, 
            "solref": cell_solref, "solimp": cell_solimp, "contype": "1", "conaffinity": "1", "friction": "0.8"
        },
        "mat_tv"   : {
            "rgba": "0.1 0.5 0.8 1.0", "weld_solref": tv_weld_solref, "weld_solimp": tv_weld_solimp, 
            "solref": tv_solref, "solimp": tv_solimp, "contype": "1", "conaffinity": "1", "friction": "0.8"
        },
        
        "ground_solref": ground_solref, 
        "ground_solimp": user_config.get("ground_solimp", "0.90 0.99 0.01 0.5 2"),
        "ground_friction": user_config.get("ground_friction", 0.2), 

        "light_head_ambient" : "0.28 0.28 0.28", "light_head_diffuse" : "0.56 0.56 0.56",
        "light_main_ambient" : "0.21 0.21 0.21", "light_main_diffuse" : "0.49 0.49 0.49", "light_sub_diffuse"  : "0.21 0.21 0.21",

        "air_density"      : 1.225, "air_viscosity"    : 1.81e-5, "air_cd_drag"      : 1.05, "air_cd_viscous"   : 0.0, "air_squeeze_hmin" : 0.001,
        "enable_air_drag"    : True, "enable_air_squeeze" : True,
        
        "chassis_aux_masses": user_config.get("chassis_aux_masses", []),

        "sim_integrator": sim_integrator, "sim_timestep": sim_timestep, "sim_iterations": sim_iterations,
        "sim_noslip_iterations": sim_noslip_iterations, "sim_impratio": sim_impratio, "sim_tolerance": sim_tolerance, "sim_gravity": sim_gravity, "sim_nthread": sim_nthread,
        "reporting_interval": user_config.get("reporting_interval", 0.005), # 데이터 저장 주기 (초)

        "cush_yield_stress": cush_yield_stress,
        "cush_solref_timec": cush_solref_timec, "cush_solref_damprr": cush_solref_damprr,
        "cush_solimp_dmin": cush_solimp_dmin, "cush_solimp_dmax": cush_solimp_dmax, "cush_solimp_width": cush_solimp_width, "cush_solimp_mid": cush_solimp_mid, "cush_solimp_power": cush_solimp_power,
        "tape_solref_timec": tape_solref_timec, "tape_solref_damprr": tape_solref_damprr,
        "tape_solimp_dmin": tape_solimp_dmin, "tape_solimp_dmax": tape_solimp_dmax, "tape_solimp_width": tape_solimp_width, "tape_solimp_mid": tape_solimp_mid, "tape_solimp_power": tape_solimp_power,
        "cell_solref_timec": cell_solref_timec, "cell_solref_damprr": cell_solref_damprr,
        "cell_solimp_dmin": cell_solimp_dmin, "cell_solimp_dmax": cell_solimp_dmax, "cell_solimp_width": cell_solimp_width, "cell_solimp_mid": cell_solimp_mid, "cell_solimp_power": cell_solimp_power,
        "tv_solref_timec": tv_solref_timec, "tv_solref_damprr": tv_solref_damprr,
        "tv_solimp_dmin": tv_solimp_dmin, "tv_solimp_dmax": tv_solimp_dmax, "tv_solimp_width": tv_solimp_width, "tv_solimp_mid": tv_solimp_mid, "tv_solimp_power": tv_solimp_power,
        "ground_solref_timec": ground_solref_timec, "ground_solref_damprr": ground_solref_damprr,
        "cush_weld_solref_timec": user_config.get('cush_weld_solref_timec', cush_solref_timec),
        "cush_weld_solref_damprr": user_config.get('cush_weld_solref_damprr', cush_solref_damprr),
    }
    
    # 사용자가 명시적으로 전달한 설정값 최우선 덮어쓰기
    for k, v in user_config.items(): config[k] = v
        
    # [7] 파라미터 최종 동기화 (Final Variable Synchronization)
    # 개별 성분(stiff, damp 등)을 수정했을 경우 MuJoCo XML 문자열로 즉각 반영하기 위한 후처리.
    
    # 바닥 동기화
    g_s = config.get("ground_solref_timec", 0.001); g_d = config.get("ground_solref_damprr", 0.0001)
    config["ground_solref"] = f"{g_s} {g_d}"
    
    # 재질별 동기화 루프
    for prefix in ["cush", "tape", "cell", "tv"]:
        # (A) Solref 재조립
        s_val = config.get(f"{prefix}_solref_timec"); d_val = config.get(f"{prefix}_solref_damprr")
        if s_val is not None and d_val is not None:
            config[f"{prefix}_solref"] = f"{s_val} {d_val}"
            
        # (A-2) [SPECIAL] Weld Solref 재조립 (WHTOOLS 전용 하드코딩 명칭 지원)
        w_prefix = prefix
        if prefix == "cell": w_prefix = "opencell"
        if prefix == "tv":   w_prefix = "chassis"
        
        ws_val = config.get(f"{w_prefix}_weld_solref_timec")
        if ws_val is None: ws_val = config.get(f"{prefix}_weld_solref_timec")
        wd_val = config.get(f"{w_prefix}_weld_solref_damprr")
        if wd_val is None: wd_val = config.get(f"{prefix}_weld_solref_damprr")
        
        if ws_val is not None and wd_val is not None:
            config[f"{prefix}_weld_solref"] = f"{ws_val} {wd_val}"
            
        # (B) Solimp 재조립 (5개 성분 필수)
        i_dmin  = config.get(f"{prefix}_solimp_dmin", 0.1)
        i_dmax  = config.get(f"{prefix}_solimp_dmax", 0.95)
        i_width = config.get(f"{prefix}_solimp_width", 0.005)
        i_mid   = config.get(f"{prefix}_solimp_mid", 0.5)
        i_power = config.get(f"{prefix}_solimp_power", 2.0)
        config[f"{prefix}_solimp"] = f"{i_dmin} {i_dmax} {i_width} {i_mid} {i_power}"

    # (C) Weld 전용 동기화 (내부 구속조건 성능 제어)
    for prefix in ["cush", "tape", "cell", "tv"]:
        target_key = f"{prefix}_weld_solref"
        s = config.get(f"{prefix}_weld_solref_timec", config.get(f"{prefix}_solref_timec"))
        d = config.get(f"{prefix}_weld_solref_damprr", config.get(f"{prefix}_solref_damprr"))
        config[target_key] = f"{s} {d}"
        config[f"{prefix}_weld_solimp"] = config.get(f"{prefix}_weld_solimp", config[f"{prefix}_solimp"])

    # (D) 중첩 딕셔너리(mat_*) 최종 업데이트 (XML 생성기에서 이 데이터를 참조함)
    config["cush_contact_solref"] = config.get("cush_contact_solref", config["cush_solref"])
    config["cush_contact_solimp"] = config.get("cush_contact_solimp", config["cush_solimp"])
    config["cush_corner_solref"]    = config.get("cush_corner_solref", config["ground_solref"])
    config["cush_corner_solimp"]    = config.get("cush_corner_solimp", "0.95 0.999 0.001 0.5 2.0")
    
    if "cush_weld_corner_solref_timec" in config and config["cush_weld_corner_solref_timec"] is not None:
        cw_s = config["cush_weld_corner_solref_timec"]; cw_d = config.get("cush_weld_corner_solref_damprr", 1.0)
        config["cush_weld_corner_solref"] = f"{cw_s} {cw_d}"
    else: config["cush_weld_corner_solref"] = config.get("cush_weld_corner_solref", None)
    
    if "mat_cush" in config:
        config["mat_cush"]["solref"]      = config["cush_contact_solref"]
        config["mat_cush"]["solimp"]      = config["cush_contact_solimp"]
        config["mat_cush"]["corner_solref"] = config["cush_corner_solref"]
        config["mat_cush"]["corner_solimp"] = config["cush_corner_solimp"]
        config["mat_cush"]["weld_solref"] = config["cush_weld_solref"]
        config["mat_cush"]["weld_solimp"] = config["cush_weld_solimp"]
        config["mat_cush"]["corner_weld_solref"] = config.get("cush_weld_corner_solref")
        config["mat_cush"]["friction"]    = config.get("cush_friction", config["mat_cush"].get("friction", "0.8"))

    mat_map = {"tape": "mat_tape", "cell": "mat_cell", "tv": "mat_tv"}
    for prefix, mat_key in mat_map.items():
        if mat_key in config:
            config[mat_key]["solref"]      = config[f"{prefix}_solref"]
            config[mat_key]["solimp"]      = config[f"{prefix}_solimp"]
            config[mat_key]["weld_solref"] = config[f"{prefix}_weld_solref"]
            config[mat_key]["weld_solimp"] = config[f"{prefix}_weld_solimp"]
            f_key = f"{prefix}_friction"
            if f_key in config: config[mat_key]["friction"] = config[f_key]

    return config
