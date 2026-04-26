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
    num_masses: int = 8,
    base_mci: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """
    설계 목표치(Target Mass, CoG, MoI)를 달성하기 위해 필요한 추가 보정 질량(Aux Masses)의 
    최적 위치와 크기를 역산합니다.
    """
    # [WHTOOLS] 재귀 방지: 외부에서 base 정보를 주거나, 직접 계산(재귀 없는 버전) 수행
    if base_mci is not None:
        m_base, c_base, i_base = base_mci
    else:
        # [WHTOOLS] Circular Import 방지를 위해 로컬 임포트 사용
        from run_discrete_builder.whtb_physics import _get_assembly_inertia_base
        temp_cfg = config.copy()
        temp_cfg["chassis_aux_masses"] = []
        temp_cfg["component_aux"] = {}
        m_base, c_base, i_base, _ = _get_assembly_inertia_base(temp_cfg)
    
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
            "name" : "AutoBalance_Single",
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
            aux_masses.append({"name": f"AutoBalance_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})
            
    elif num_masses == 4:
        m_each = m_aux / 4.0
        # XY 평면 분산 배치
        dx = math.sqrt(max(0.005, (t_moi[1] - i_base[1]) / (4.0 * m_each))) if t_moi is not None else 0.05
        dy = math.sqrt(max(0.005, (t_moi[0] - i_base[0]) / (4.0 * m_each))) if t_moi is not None else 0.05
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                p = [pos_aux[0] + sx * dx, pos_aux[1] + sy * dy, pos_aux[2]]
                aux_masses.append({"name": f"AutoBalance_{len(aux_masses)+1}", "pos": clip_pos(p), "mass": m_each, "size": [0.01]*3})
                
    else: # Default 8 masses
        m_each = m_aux / 8.0
        
        # [WHTOOLS] 8개 질량 분산 배치를 위한 연립 방정식 해결
        # I_xx_contribution = M_aux * (dy^2 + dz^2)
        # I_yy_contribution = M_aux * (dx^2 + dz^2)
        # I_zz_contribution = M_aux * (dx^2 + dy^2)
        
        # 1. 기저 모델을 타겟 CoG로 이동시켰을 때의 관성 (평행축 정리)
        d = t_cog - c_base
        i_at_t = i_base + m_base * np.array([d[1]**2 + d[2]**2, d[0]**2 + d[2]**2, d[0]**2 + d[1]**2])
        
        # 2. 보조 질량계가 담당해야 할 추가 관성량
        di = t_moi - i_at_t
        
        # 3. 연립 방정식 해결: A=dy^2+dz^2, B=dx^2+dz^2, C=dx^2+dy^2
        A, B, C = di / (m_aux if m_aux > 0 else 1.0)
        
        # dx^2 = (B + C - A) / 2
        # dy^2 = (A + C - B) / 2
        # dz^2 = (A + B - C) / 2
        dx2 = (B + C - A) / 2.0
        dy2 = (A + C - B) / 2.0
        dz2 = (A + B - C) / 2.0
        
        # 물리적 한계 체크 (관성이 너무 낮으면 COG에 밀착)
        dx = math.sqrt(max(1e-6, dx2))
        dy = math.sqrt(max(1e-6, dy2))
        dz = math.sqrt(max(1e-6, dz2))
        
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    p = [pos_aux[0] + sx * dx, pos_aux[1] + sy * dy, pos_aux[2] + sz * dz]
                    aux_masses.append({
                        "name": f"AutoBalance_{len(aux_masses)+1}", 
                        "pos": clip_pos(p), 
                        "mass": m_each, 
                        "size": [0.01, 0.01, 0.01]
                    })
                    
    return aux_masses
