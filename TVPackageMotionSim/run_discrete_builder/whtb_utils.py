import numpy as np
import math
from typing import List, Tuple, Union, Optional, Any

def get_local_pose(vec: Union[List[float], np.ndarray], drop_height: float, rot_axis: np.ndarray, angle_rad: float, corner_dist: float) -> np.ndarray:
    """최상위 루트(BPackagingBox) 구동을 위한 글로벌 포즈 계산용"""
    v = np.array(vec)
    v_rot = v * np.cos(angle_rad) + np.cross(rot_axis, v) * np.sin(angle_rad) + rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_rad))
    return v_rot + np.array([0, 0, drop_height + corner_dist])

def calculate_solref(K: float, C: float) -> Tuple[float, float]:
    """
    물리적인 강성(Stiffness, K)과 점성 감쇠(Damping, C)를 입력받아
    MuJoCo의 기본 solref 양수 입력방식인 (timeconst, dampratio)로 유추/변환하는 헬퍼 함수입니다.
    """
    if K <= 0:
        raise ValueError("강성(Stiffness) K는 0보다 커야 합니다.")
        
    # 부드러울수록(K가 작을수록) timeconst는 커짐 (최소 하한 0.002)
    timeconst = 1.0 / math.sqrt(K)
    
    if timeconst < 0.002:
        timeconst = 0.002
        K_safe = 1.0 / (timeconst**2)
    else:
        K_safe = K
        
    # 임계 감쇠(Critical Damping)를 1.0으로 하는 감쇠비(dampratio) 산출
    dampratio = C / (2.0 * math.sqrt(K_safe))
    
    # 소수점 5자리까지만 정리하여 반환
    return round(timeconst, 5), round(dampratio, 5)

def parse_drop_target(mode_str: str, direction_str: str, box_w: float, box_h: float, box_d: float) -> np.ndarray:
    """
    낙하 모드와 방향 문자열을 파싱하여 바닥에 닿을 로컬 좌표(타겟 점)를 반환합니다.
    - drop_mode: PARCEL, LTL, CUSTOM (공정/시험 기준 분류용)
    - drop_direction: 'Face 1', 'Edge 3-4', 'Corner 2-3-5', '235', 'front-bottom-left' 등
    """
    import re
    mode = str(mode_str).upper()
    direct = str(direction_str).lower().replace('face', '').replace('edge', '').replace('corner', '').strip()
    
    # [WHTOOLS ISTA 6-Amazon Mapping] - Y=Height, Z=Depth 기준
    # 1: Top (+Y), 2: Bottom (-Y)
    # Parcel(G): 3/4=Sides(±X), 5/6=FrontBack(±Z)
    # LTL(H): 3/4=RearFront(±Z), 5/6=Sides(±X)
    
    parcel_map = {
        1:[0,1,0], 2:[0,-1,0], # Top/Bottom
        3:[1,0,0], 4:[-1,0,0], # Right/Left Side
        5:[0,0,1], 6:[0,0,-1]  # Front/Back
    }
    ltl_map = {
        1:[0,1,0], 2:[0,0,-1], # Top/Back (Rear)
        3:[0,-1,0], 4:[0,0,1], # Bottom/Front Screen
        5:[1,0,0], 6:[-1,0,0]  # Right/Left Side
    }
    
    face_map = ltl_map if mode == 'LTL' else parcel_map
    vec = np.array([0.0, 0.0, 0.0])
    
    # 숫자(1~6)가 포함된 경우 해당 맵핑 적용
    nums = [int(n) for n in re.findall(r'[1-6]', direct)]
    if nums:
        for n in nums: 
            vec += np.array(face_map.get(n, [0,0,0]))
    
    # 키워드 기반 토큰 파싱 (숫자가 없거나 보조 설명인 경우)
    if np.linalg.norm(vec) < 1e-6 or any(kw in direct for kw in ['front', 'rear', 'top', 'bottom', 'left', 'right']):
        tokens = [t.strip() for t in re.split(r'[-,\s]+', direct)]
        for tk in tokens:
            if 'front' in tk or 'screen' in tk: vec[2] = 1.0   # Front/Screen -> Z+
            elif 'rear' in tk or 'back' in tk: vec[2] = -1.0  # Rear/Back -> Z-
            elif 'top' in tk: vec[1] = 1.0                    # Top -> Y+
            elif 'bottom' in tk: vec[1] = -1.0                # Bottom -> Y-
            elif 'left' in tk: vec[0] = -1.0                  # Left -> X-
            elif 'right' in tk: vec[0] = 1.0                  # Right -> X+
    
    if np.linalg.norm(vec) < 1e-6:
        # Default fallback
        if mode == 'LTL': vec = np.array([0.0, -1.0, 0.0]) # Bottom
        else: vec = np.array([0.0, -1.0, 0.0]) # Bottom (ISTA Default)

    # 부호 {-1, 0, 1} 상태의 vec을 박스 절반 크기에 곱해 실제 외곽점(Target Corner) 산출
    # X=Width/2, Y=Height/2, Z=Depth/2
    target_pt = np.array([vec[0] * box_w/2, vec[1] * box_h/2, vec[2] * box_d/2])
    
    if np.linalg.norm(target_pt) < 1e-6:
        target_pt = np.array([0, -box_h/2, 0]) # Defacto Bottom
        
    return target_pt

def get_rgba_by_name(color_name: str, alpha: float = 1.0) -> str:
    """
    [WHTOOLS] 영문 색상명과 투명도(alpha)를 입력받아 MuJoCo용 RGBA 문자열을 반환합니다.
    
    Args:
        color_name (str): 색상 명칭 (예: 'white', 'red', 'gray', 'paper', 'black' 등)
        alpha (float): 투명도 (0.0 ~ 1.0)
        
    Returns:
        str: "R G B A" 형식의 문자열 (예: "1.0 0.0 0.0 0.5")
    """
    colors = {
        "white":   [1.0, 1.0, 1.0],
        "gray":    [0.5, 0.5, 0.5],
        "black":   [0.1, 0.1, 0.1],
        "red":     [1.0, 0.0, 0.0],
        "green":   [0.0, 1.0, 0.0],
        "blue":    [0.0, 0.0, 1.0],
        "yellow":  [1.0, 1.0, 0.0],
        "cyan":    [0.0, 1.0, 1.0],
        "magenta": [1.0, 0.0, 1.0],
        "orange":  [1.0, 0.5, 0.0],
        "purple":  [0.5, 0.0, 0.5],
        "brown":   [0.5, 0.3, 0.2],
        "paper":   [0.5, 0.3, 0.2],
        "cushion": [0.9, 0.9, 0.9],
        "metal":   [0.7, 0.7, 0.7],
        "glass":   [0.1, 0.1, 0.1], # Dark black for open cell
    }
    
    name = color_name.lower().strip()
    rgb = colors.get(name, [0.8, 0.8, 0.8]) # Default Light Gray
    
    return f"{rgb[0]} {rgb[1]} {rgb[2]} {alpha}"

def calculate_plate_twist_weld_params(mass: float, width: float, height: float, thickness: float, 
                                     div: Tuple[int, int],
                                     E_real: float = None, real_thickness: float = None,
                                     nu: float = 0.22, zeta: float = 0.05,
                                     target_freq_hz: float = None,
                                     base_freq_hz: float = None,
                                     verbose: bool = True):
    """
    [WHTOOLS] 판재의 1차 트위스트 고유진동수를 기반으로 MuJoCo solref 및 torquescale을 산출합니다.
    """
    Nx, Ny = div[0], div[1]
    L, W, h_mj = max(width, height), min(width, height), thickness
    
    # 1. 베이스 진동수 (Axial) 결정
    method_base = "Manual"
    if E_real is not None and real_thickness is not None:
        # 물리적 인장 강성 기반: K = E * (W * h_real) / L
        k_axial = E_real * (W * real_thickness) / L
        f_base_global = (1.0 / (2.0 * np.pi)) * np.sqrt(k_axial / mass)
        method_base = f"Physical (E={E_real/1e9:.1f}GPa, h={real_thickness*1000:.2f}mm)"
    elif base_freq_hz is not None:
        f_base_global = base_freq_hz
        method_base = "Direct Input"
    else:
        f_base_global = target_freq_hz if target_freq_hz is not None else 10.0
        method_base = "Fallback to Target"

    # 2. 목표 진동수 (Bending) 결정
    method_target = "Experimental"
    if target_freq_hz is not None:
        f_target_global = target_freq_hz
    else:
        # 이론치 계산 (트위스트 모드)
        rho = mass / (width * height * h_mj)
        E_for_twist = E_real if E_real is not None else 7e10
        h_for_twist = real_thickness if real_thickness is not None else h_mj
        f_target_global = (1.0 / (2.0 * np.pi)) * (12.43 * h_for_twist / (L * W)) * np.sqrt(E_for_twist / (rho * (1 - nu**2)))
        method_target = "Theoretical (Twist)"

    # 3. 격자 분할 보정 (Local Equivalent)
    S = np.sqrt(Nx**2 + Ny**2)
    f_target_local = f_target_global * S
    f_base_local = f_base_global * S

    # [WHTOOLS] 수치적 안정성 캡 (Simulation Stability Cap)
    # 일반적인 dt=0.001s 환경에서 200Hz 이상의 국부 강성은 폭발을 유발함
    max_f_safe = 200.0 
    is_capped = False
    if f_base_local > max_f_safe:
        f_base_local = max_f_safe
        is_capped = True
    
    # Target(Bending)은 Base(Axial)보다 클 수 없음
    f_target_local = min(f_target_local, f_base_local)

    # 4. 파라미터 계산 (Base Local 기준)
    omega_base = 2 * np.pi * f_base_local
    solref_k = -(omega_base ** 2)
    solref_d = -(2 * zeta * omega_base)
    
    # 5. 토크 스케일 계산 (Bending 강성비)
    torquescale = (f_target_local / f_base_local)**2 if f_base_local > 0 else 1.0
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"📐 [WHTOOLS] Stiffness Calculation Report")
        if is_capped:
            print(f"⚠️  [WARNING] Frequency capped at {max_f_safe}Hz for stability!")
        print(f"{'='*50}")
        print(f"🔹 Global Targets:")
        print(f"   - Axial Freq (Base)   : {f_base_global:6.2f} Hz  [{method_base}]")
        print(f"   - Bending Freq (Target): {f_target_global:6.2f} Hz  [{method_target}]")
        print(f"🔹 Discretization (div={div}):")
        print(f"   - Scaling Factor (S)  : {S:6.2f}x")
        print(f"   - Local Axial Freq    : {f_base_local:6.2f} Hz")
        print(f"   - Local Bending Freq  : {f_target_local:6.2f} Hz")
        print(f"🔹 MuJoCo Parameters:")
        print(f"   - solref (K, D)       : [{solref_k:8.1f}, {solref_d:8.1f}]")
        print(f"   - torquescale         : {torquescale:10.6f} (Bending Ratio)")
        print(f"{'='*50}\n")
    
    return solref_k, solref_d, torquescale


def mat2axisangle(R: np.ndarray) -> np.ndarray:
    """Rotation Matrix -> Axis-Angle (x, y, z, deg)"""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        # 180 degree rotation case
        # Find the largest diagonal element
        idx = np.argmax(np.diag(R))
        axis = np.zeros(3)
        axis[idx] = 1.0
        # In a real 180 case, axis = sqrt((diag+1)/2) etc. but for MuJoCo simple is ok if trace is handled.
        # Actually MuJoCo handles small angles but 180 is special.
        # This is a simplified fallback.
    else:
        axis /= axis_norm
    return np.concatenate([axis, [np.degrees(angle)]])

def get_drop_orientation_matrix(target_pt: np.ndarray, ref_vec: np.ndarray, global_ref_target: np.ndarray = np.array([0, -1, 0])) -> np.ndarray:
    """
    [WHTOOLS] 낙하 자세 정렬을 위한 회전 행렬을 생성합니다.
    1. target_pt가 글로벌 -Z 방향(바닥)을 향하도록 1차 회전 (Shortest Rotation)
    2. 글로벌 Z축을 기준으로 회전하여 ref_vec의 수평 투영 성분이 global_ref_target을 향하도록 2차 회전
    """
    # 1. Primary Alignment: target_pt -> [0, 0, -1]
    t = target_pt / (np.linalg.norm(target_pt) + 1e-12)
    k = np.array([0, 0, -1])
    
    axis = np.cross(t, k)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-9:
        if t[2] < 0: R1 = np.eye(3)
        else: R1 = np.diag([1, -1, -1]) # 180 deg around X
    else:
        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(t, k), -1, 1))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R1 = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
    # 2. Secondary Alignment: Rotate around global Z
    v1 = R1 @ ref_vec
    v1_proj = np.array([v1[0], v1[1], 0])
    v1_proj_norm = np.linalg.norm(v1_proj)
    
    if v1_proj_norm < 1e-6:
        return R1 # Projection is too small, skip secondary rotation
        
    v1_proj /= v1_proj_norm
    tar = global_ref_target / (np.linalg.norm(global_ref_target) + 1e-12)
    
    # R_psi @ v1_proj = tar
    # psi = atan2(v1_proj x tar . z, v1_proj . tar)
    cos_psi = np.dot(v1_proj, tar)
    sin_psi = v1_proj[0] * tar[1] - v1_proj[1] * tar[0]
    
    R_psi = np.array([[cos_psi, -sin_psi, 0],
                      [sin_psi,  cos_psi, 0],
                      [0,        0,       1]])
    
    return R_psi @ R1
