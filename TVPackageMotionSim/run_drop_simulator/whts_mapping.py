import numpy as np
from typing import Dict, List, Any, Tuple
from .whts_data import DropSimResult

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] -> Rotation Matrix [3, 3]"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])

def get_face_index_logic(face_name: str, max_indices: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    [WHTOOLS Unified Coordinate Mapping]
    명칭별 격자 인덱스(i,j,k) 및 면 법선 방향(normal)을 정의합니다.
    """
    max_i, max_j, max_k = max_indices
    logic = {
        "Top":    {"axis": "j", "idx": max_j, "normal": [0,1,0],  "plane": ("i","k")},
        "Bottom": {"axis": "j", "idx": 0,     "normal": [0,-1,0], "plane": ("i","k")},
        "Front":  {"axis": "k", "idx": max_k, "normal": [0,0,1],  "plane": ("i","j")},
        "Rear":   {"axis": "k", "idx": 0,     "normal": [0,0,-1], "plane": ("i","j")},
        "Left":   {"axis": "i", "idx": 0,     "normal": [-1,0,0], "plane": ("k","j")},
        "Right":  {"axis": "i", "idx": max_i, "normal": [1,0,0],  "plane": ("k","j")}
    }
    return logic.get(face_name)

def extract_face_markers(result: DropSimResult, part_name: str, p_size: Tuple[float, float, float] = None, mode: str = 'statistical') -> Tuple[Dict, Dict]:
    """
    [WHTOOLS Hybrid Marker Extraction v5.6.0]
    - mode='statistical': PCA/SVD 기반 통계적 정렬 (MoCap 데이터 대응)
    - mode='kinematic': 본체 회전 행렬 기반 역학적 정렬 (시뮬레이션 Truth 대응)
    """
    comp_name = part_name.lower()
    
    # [WHTOOLS] Flexible Name Mapping (Substring support)
    # If exact match fails, look for a key that is a substring of comp_name or vice-versa
    found_key = None
    if hasattr(result, 'components'):
        if comp_name in result.components:
            found_key = comp_name
        else:
            # Try to find a partial match (e.g., 'cushion' inside 'bcushion')
            for key in result.components.keys():
                if key in comp_name or comp_name in key:
                    found_key = key
                    break
    
    if found_key is None:
        return {}, {}

    body_map = result.components[found_key]
    indices = body_map.keys()
    if not indices: return {}, {}

    # [V5.6.2] 인덱스 정규화 (단일 블록인 경우 0으로 기본값 설정)
    if not indices:
        max_i, max_j, max_k = 0, 0, 0
    else:
        max_i = max(idx[0] for idx in indices) if indices else 0
        max_j = max(idx[1] for idx in indices) if indices else 0
        max_k = max(idx[2] for idx in indices) if indices else 0

    face_prefix = {"Front": "F", "Rear": "R", "Left": "L", "Right": "Ri", "Top": "T", "Bottom": "B"}
    face_markers = {face: {} for face in face_prefix}
    face_offsets = {face: {} for face in face_prefix}
    
    pos_hist = np.array(result.pos_hist)
    quat_hist = np.array(result.quat_hist)
    n_frames = pos_hist.shape[0]

    for face_name in face_prefix:
        face_logic = get_face_index_logic(face_name, (max_i, max_j, max_k))
        if not face_logic: continue
        
        node_accumulator = {}
        target_axis, target_idx = face_logic["axis"], face_logic["idx"]

        # 마커 노출 및 3D 데이터 수집
        for (i, j, k), body_id in body_map.items():
            is_on_face = (target_axis == "i" and i == target_idx) or \
                         (target_axis == "j" and j == target_idx) or \
                         (target_axis == "k" and k == target_idx)
            if not is_on_face: continue
            
            dx, dy, dz = result.block_half_extents[body_id]
            norm_vec = np.array(face_logic["normal"])
            p_axes = face_logic["plane"]
            d_val = {"i": dx, "j": dy, "k": dz}
            
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    lv = np.zeros(3)
                    lv_idx = 0 if target_axis=="i" else 1 if target_axis=="j" else 2
                    lv[lv_idx] = norm_vec[lv_idx] * d_val[target_axis]
                    
                    p1_idx = 0 if p_axes[0]=="i" else 1 if p_axes[0]=="j" else 2
                    p2_idx = 0 if p_axes[1]=="i" else 1 if p_axes[1]=="j" else 2
                    lv[p1_idx], lv[p2_idx] = s1 * d_val[p_axes[0]], s2 * d_val[p_axes[1]]
                    
                    ni = [i, j, k]
                    ni[p1_idx] = ni[p1_idx] if s1 < 0 else ni[p1_idx] + 1
                    ni[p2_idx] = ni[p2_idx] if s2 < 0 else ni[p2_idx] + 1
                    node_idx = tuple(ni)
                    
                    if node_idx not in node_accumulator: node_accumulator[node_idx] = []
                    node_accumulator[node_idx].append((body_id, lv))

        for node_idx, contributions in node_accumulator.items():
            node_pos_final = np.zeros((n_frames, 3))
            for body_id, l_vec in contributions:
                for f in range(n_frames):
                    R = quat_to_mat(quat_hist[f, body_id])
                    node_pos_final[f] += pos_hist[f, body_id] + R @ l_vec
            node_pos_final /= len(contributions)
            m_name = f"{face_prefix[face_name]}_{node_idx[0]}_{node_idx[1]}_{node_idx[2]}"
            face_markers[face_name][m_name] = node_pos_final

        # -------------------------------------------------------------
        # [WHTOOLS] Coordinate Mapping (Projection & Alignment)
        # -------------------------------------------------------------
        all_node_names = sorted(list(face_markers[face_name].keys()))
        if not all_node_names: continue
        
        P0 = np.stack([face_markers[face_name][m][0] for m in all_node_names])
        c_P = np.mean(P0, axis=0)
        
        if mode == 'kinematic':
            # [Theory] 바디의 초기 회전 행렬을 직접 사용하여 기저 벡터 생성
            first_body_id = list(node_accumulator.values())[0][0][0]
            R0 = quat_to_mat(quat_hist[0, first_body_id])
            ax_map = {"i": [1,0,0], "j": [0,1,0], "k": [0,0,1]}
            p_axes = face_logic["plane"]
            e1, e2 = R0 @ np.array(ax_map[p_axes[0]]), R0 @ np.array(ax_map[p_axes[1]])
        else:
            # [Statistical] SVD 기반 주축 추출
            P_centered = P0 - c_P
            _, _, Vt = np.linalg.svd(P_centered)
            normal = Vt[2]
            if np.dot(normal, c_P) < 0: normal = -normal
            
            ax_map = {"i": [1,0,0], "j": [0,1,0], "k": [0,0,1]}
            p_axes = face_logic["plane"]
            h_ideal, v_ideal = np.array(ax_map[p_axes[0]]), np.array(ax_map[p_axes[1]])
            
            e1 = h_ideal - np.dot(h_ideal, normal) * normal
            e1 /= (np.linalg.norm(e1) + 1e-9)
            e2 = np.cross(normal, e1)
            if np.dot(e2, v_ideal) < 0: e2 = -e2
            
        # 2D 투영 및 정렬
        P2D = np.stack([np.array([np.dot(p - c_P, e1), np.dot(p - c_P, e2)]) for p in P0])
        
        if mode == 'statistical':
            # 통계 모드일 때만 장단축비 기반 90도 회전 보정 (MoCap 대응)
            if face_name in ["Top", "Bottom"]:   w_nom, h_nom = p_size[0], p_size[2] if p_size else (1,1)
            elif face_name in ["Front", "Rear"]: w_nom, h_nom = p_size[0], p_size[1] if p_size else (1,1)
            else:                                w_nom, h_nom = p_size[2], p_size[1] if p_size else (1,1)
            
            w_a = np.max(P2D[:,0]) - np.min(P2D[:,0])
            h_a = np.max(P2D[:,1]) - np.min(P2D[:,1])
            if (w_nom > h_nom and h_a > w_a) or (h_nom > w_nom and w_a > h_a):
                P2D = P2D[:, [1, 0]]

        for idx, m_name in enumerate(all_node_names):
            face_offsets[face_name][m_name] = P2D[idx]

    return face_markers, face_offsets

def get_assembly_data_from_sim(result: DropSimResult, target_parts: List[str], mode: str = 'statistical') -> Tuple[Dict, Dict]:
    total_markers, total_offsets = {}, {}
    for part in target_parts:
        p_c = result.config.get(part, {})
        p_size = (p_c.get("box_w", 1.0), p_c.get("box_h", 1.0), p_c.get("box_d", 1.0))
        m, o = extract_face_markers(result, part, p_size, mode=mode)
        total_markers[part], total_offsets[part] = m, o
    return total_markers, total_offsets

