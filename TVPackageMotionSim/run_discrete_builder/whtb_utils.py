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
