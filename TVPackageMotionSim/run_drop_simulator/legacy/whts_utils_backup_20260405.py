import numpy as np
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from run_discrete_builder import create_model

def compute_corner_kinematics(
    center_pos: np.ndarray, 
    center_mat: np.ndarray, 
    center_vel: np.ndarray, 
    center_acc: np.ndarray, 
    box_w: float, box_h: float, box_d: float
) -> List[Dict[str, np.ndarray]]:
    """
    조립체 중심의 위치, 회전, 속도, 가속도 데이터로부터 8개 모서리 꼭지점의 
    글로벌 위치/속도/가속도를 강체 운동학(Rigid Body Kinematics) 공식으로 역산합니다.
    
    Args:
        center_pos (np.ndarray): 중심점의 3차원 글로벌 위치 [x, y, z]
        center_mat (np.ndarray): 3x3 회전 행렬 (body xmat)
        center_vel (np.ndarray): 6자유도 속도 [wx, wy, wz, vx, vy, vz]
        center_acc (np.ndarray): 6자유도 가속도 [alpha_x, alpha_y, alpha_z, ax, ay, az]
        box_w (float): 박스의 가로 길이 (Width, m)
        box_h (float): 박스의 세로 길이 (Height, m)
        box_d (float): 박스의 깊이 (Depth, m)
    
    Returns:
        List[Dict]: 8개 모서리의 {'pos': ndarray, 'vel': ndarray, 'acc': ndarray} 리스트
    """
    w = center_vel[0:3]     # 각속도 (Angular velocity)
    v = center_vel[3:6]     # 선속도 (Linear velocity)
    alpha = center_acc[0:3] # 각가속도 (Angular acceleration)
    a = center_acc[3:6]     # 선가속도 (Linear acceleration)
    
    corners_local = []
    # 8개 꼭지점의 로컬 좌표 생성
    for x in [-box_w / 2, box_w / 2]:
        for y in [-box_h / 2, box_h / 2]:
            for z in [-box_d / 2, box_d / 2]:
                corners_local.append(np.array([x, y, z]))
    
    results = []
    for loc in corners_local:
        # 글로벌 오프셋 벡터 (r = R * r_local)
        r = center_mat @ loc
        
        # 선속도 공식: v_p = v_cg + w × r
        v_corner = v + np.cross(w, r)
        
        # 선가속도 공식: a_p = a_cg + α × r + w × (w × r)
        a_corner = a + np.cross(alpha, r) + np.cross(w, np.cross(w, r))
        
        results.append({
            'pos': center_pos + r,
            'vel': v_corner,
            'acc': a_corner
        })
    return results

def calculate_required_aux_masses(
    config: Dict[str, Any],
    target_mass: float, 
    target_cog: Optional[Union[List[float], np.ndarray]] = None, 
    target_moi: Optional[Union[List[float], np.ndarray]] = None, 
    num_masses: int = 8
) -> List[Dict[str, Any]]:
    """
    설계 목표치(Target Mass, CoG, MoI)를 달성하기 위해 필요한 추가 보정 질량(Aux Masses)의 
    최적 위치와 크기를 역산합니다.
    
    Args:
        config (Dict): 현재 시뮬레이션 설정 (박스 크기 등 참조용)
        target_mass (float): 목표로 하는 총 질량 (kg)
        target_cog (List[float], optional): 목표 질량 중심 좌표 (m)
        target_moi (List[float], optional): 목표 관성 모멘트 (Ixx, Iyy, Izz, kg*m^2)
        num_masses (int): 배치할 보정 질량체의 개수 (지원: 1, 2, 4, 8)
    
    Returns:
        List[Dict]: 보정 질량체 리스트 [{'name': str, 'pos': [x,y,z], 'mass': float, 'size': [w,h,d]}]
    """
    # [1] 기저 모델(보정 전)의 관성 데이터 확보
    temp_cfg = config.copy()
    temp_cfg["chassis_aux_masses"] = []
    # Builder를 이용해 임시 모델의 질량/관성 측정
    _, m_base, c_base, i_base, _ = create_model("temp_inertia_balancer.xml", config=temp_cfg, logger=lambda x: None)
    
    m_base = float(m_base)
    c_base = np.array(c_base)
    i_base = np.array(i_base)
    
    t_mass = target_mass if target_mass is not None else m_base
    t_cog  = np.array(target_cog) if (target_cog is not None and len(target_cog) == 3) else c_base
    t_moi  = np.array(target_moi) if (target_moi is not None and len(target_moi) == 3) else None
    
    # 추가 필요 질량 (Target - Current)
    m_aux = t_mass - m_base
    if m_aux < 0:
        # 목표 질량이 현재보다 작으면 최소한의 질량으로 CoG만 보정 시도
        m_aux = 1e-4 
        
    # 보정 질량계의 평균 중심 좌표 (M_total * C_total = M_base * C_base + M_aux * C_aux)
    pos_aux = (t_cog * t_mass - m_base * c_base) / (m_aux if m_aux > 0 else 1e-6)
    
    # 박스 바운딩 박스 제한 (내부 안착 유도, 90% 마진)
    bw, bh, bd = config.get('box_w', 2.0), config.get('box_h', 1.4), config.get('box_d', 0.25)
    limit_x, limit_y, limit_z = bw/2.0 * 0.9, bh/2.0 * 0.9, bd/2.0 * 0.9
    
    def clip_pos(p):
        return [
            float(np.clip(p[0], -limit_x, limit_x)),
            float(np.clip(p[1], -limit_y, limit_y)),
            float(np.clip(p[2], -limit_z, limit_z))
        ]

    aux_masses = []
    
    # 배치 로직: 보충 질량 개수에 따라 기하학적으로 배분
    if num_masses <= 1 or t_moi is None:
        aux_masses.append({
            "name" : "InertiaAux_Single",
            "pos"  : clip_pos(pos_aux),
            "mass" : float(m_aux),
            "size" : [0.01, 0.01, 0.01]
        })
    elif num_masses == 2:
        m_each = m_aux / 2.0
        i_needed = t_moi[1] - i_base[1] if t_moi is not None else 0
        dx = math.sqrt(max(0.005, i_needed / (2.0 * m_each)))
        for sx in [-1, 1]:
            p = [pos_aux[0] + sx * dx, pos_aux[1], pos_aux[2]]
            aux_masses.append({"name": f"InertiaAux_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})
            
    elif num_masses == 4:
        m_each = m_aux / 4.0
        # XY 평면 분산 배치
        dx = math.sqrt(max(0.005, (t_moi[1] - i_base[1]) / (4.0 * m_each))) if t_moi is not None else 0.05
        dy = math.sqrt(max(0.005, (t_moi[0] - i_base[0]) / (4.0 * m_each))) if t_moi is not None else 0.05
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                p = [pos_aux[0] + sx * dx, pos_aux[1] + sy * dy, pos_aux[2]]
                aux_masses.append({"name": f"InertiaAux_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})
                
    else: # Default 8 masses
        m_each = m_aux / 8.0
        # 평행축 정리를 이용한 정밀 배분 (Axis-symmetric)
        def get_shift(m, i_target, i_base, c_target, c_base):
            d = c_target - c_base
            i_at_t = i_base + m_base * np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])
            res = (i_target - i_at_t) / (m_aux if m_aux > 0 else 1)
            return np.sqrt(np.maximum(0.001, res))

        shifts = get_shift(m_base, t_moi, i_base, t_cog, c_base) if t_moi is not None else np.array([0.05, 0.05, 0.05])
        dx, dy, dz = shifts[0], shifts[1], shifts[2]
        
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    p = [pos_aux[0] + sx * dx, pos_aux[1] + sy * dy, pos_aux[2] + sz * dz]
                    aux_masses.append({"name": f"InertiaAux_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})
                    
    return aux_masses
