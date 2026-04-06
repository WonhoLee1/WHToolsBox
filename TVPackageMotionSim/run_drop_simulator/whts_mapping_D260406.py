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
    - X (i): Width (가로)
    - Y (j): Height (높이)
    - Z (k): Depth (깊이 / Screen)
    """
    max_i, max_j, max_k = max_indices
    logic = {
        "Top":    {"axis": "j", "idx": max_j, "normal": [0,1,0],  "plane": ("i","k"), "offsets": ["dx","dz"]},
        "Bottom": {"axis": "j", "idx": 0,     "normal": [0,-1,0], "plane": ("i","k"), "offsets": ["dx","dz"]},
        "Front":  {"axis": "k", "idx": max_k, "normal": [0,0,1],  "plane": ("i","j"), "offsets": ["dx","dy"]},
        "Rear":   {"axis": "k", "idx": 0,     "normal": [0,0,-1], "plane": ("i","j"), "offsets": ["dx","dy"]},
        "Left":   {"axis": "i", "idx": 0,     "normal": [-1,0,0], "plane": ("j","k"), "offsets": ["dy","dz"]},
        "Right":  {"axis": "i", "idx": max_i, "normal": [1,0,0],  "plane": ("j","k"), "offsets": ["dy","dz"]}
    }
    return logic.get(face_name)

def extract_face_markers(result: DropSimResult, component_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    MuJoCo 이산화 블록의 면(Face)별 4개 꼭짓점(Vertex)을 추출하고, 
    중복되는 위치의 꼭짓점들을 평균화(Averaging)하여 고해상도 마커 데이터를 생성합니다.
    
    Returns:
        markers: { marker_name: position_history_array }
        offsets: { marker_name: (x, y) local_offset }
    """
    comp_name = component_name.lower()
    
    if not hasattr(result, 'components') or comp_name not in result.components:
        print(f"[whts_mapping] Component mapping for '{comp_name}' not found in result.components.")
        return {}, {} # 항상 튜플 반환 (Unpack 에러 방지)

    body_map = result.components[comp_name]
    indices = list(body_map.keys())
    if not indices: return {}, {}

    max_i = max(idx[0] for idx in indices)
    max_j = max(idx[1] for idx in indices)
    max_k = max(idx[2] for idx in indices)

    face_prefix = {"Front": "F", "Rear": "R", "Left": "L", "Right": "Ri", "Top": "T", "Bottom": "B"}
    face_markers = {face: {} for face in face_prefix}
    face_offsets = {face: {} for face in face_prefix}
    
    pos_hist = np.array(result.pos_hist)   # [N_frames, N_bodies, 3]
    quat_hist = np.array(result.quat_hist) # [N_frames, N_bodies, 4]
    n_frames = pos_hist.shape[0]

    # 각 면별로 '그리드 노드'를 정의하여 평균화 처리
    # node_accumulator: { face_name: { (node_i, node_j, node_k): [pos_list] } }
    
    for face_name in face_prefix:
        face_logic = get_face_index_logic(face_name, (max_i, max_j, max_k))
        if not face_logic: continue
        
        node_accumulator = {}
        target_axis = face_logic["axis"]
        target_idx = face_logic["idx"]

        for (i, j, k), body_id in body_map.items():
            # 격자 위치 확인
            is_on_face = False
            if target_axis == "i" and i == target_idx: is_on_face = True
            elif target_axis == "j" and j == target_idx: is_on_face = True
            elif target_axis == "k" and k == target_idx: is_on_face = True
            
            if not is_on_face: continue
            
            dx, dy, dz = result.block_half_extents[body_id]
            norm_vec = np.array(face_logic["normal"])
            
            # 면 법선 방향에 따른 4개 꼭짓점 생성 (로컬 좌표계)
            # Plane과 Offsets 정보를 사용하여 동적 생성
            p_axes = face_logic["plane"]  # e.g., ("i", "j")
            o_axes = face_logic["offsets"] # e.g., ["dx", "dy"]
            
            # 오프셋 값 매핑 (v5.3.7: i,j,k와 x,y,z 표기법 모두 지원하도록 통합)
            d_val = {"di": dx, "dj": dy, "dk": dz, "dx": dx, "dy": dy, "dz": dz}
            d1, d2 = d_val[o_axes[0]], d_val[o_axes[1]]
            
            # 로컬 위치 및 노드 인덱스 증분 정의
            local_vertices = []
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    # 1. 로컬 벡터 구성
                    lv = np.zeros(3)
                    # 법선 축 값 설정
                    lv_idx = 0 if target_axis=="i" else 1 if target_axis=="j" else 2
                    lv[lv_idx] = norm_vec[lv_idx] * d_val[f"d{target_axis}"]
                    # 평면 축 값 설정
                    p1_idx = 0 if p_axes[0]=="i" else 1 if p_axes[0]=="j" else 2
                    p2_idx = 0 if p_axes[1]=="i" else 1 if p_axes[1]=="j" else 2
                    lv[p1_idx] = s1 * d1
                    lv[p2_idx] = s2 * d2
                    
                    # 2. 노드 인덱스 구성
                    ni = [i, j, k]
                    ni[p1_idx] = ni[p1_idx] if s1 < 0 else ni[p1_idx] + 1
                    ni[p2_idx] = ni[p2_idx] if s2 < 0 else ni[p2_idx] + 1
                    
                    local_vertices.append((lv, tuple(ni)))

            # 모든 프레임에 대해 꼭짓점 월드 좌표 계산 및 누적
            for local_v, node_idx in local_vertices:
                if node_idx not in node_accumulator:
                    node_accumulator[node_idx] = []
                
                # 벡터화 연산을 위해 일단 블록 데이터 저장
                node_accumulator[node_idx].append((body_id, np.array(local_v)))

        # 2. 누적된 데이터 평균화하여 최종 마커 생성
        for node_idx, contributions in node_accumulator.items():
            # contributions: list of (body_id, local_vec)
            # 해당 노드의 모든 프레임 위치 계산: Mean( P_body + R_body * V_local )
            
            node_pos_final = np.zeros((n_frames, 3))
            for body_id, l_vec in contributions:
                # 이 바디의 모든 프레임 회전 행렬 계산 (성능을 위해 최적화 필요할 수 있으나 우선 직관적 구현)
                for f in range(n_frames):
                    R = quat_to_mat(quat_hist[f, body_id])
                    node_pos_final[f] += pos_hist[f, body_id] + R @ l_vec
            
            node_pos_final /= len(contributions)
            
            # 마커 이름 지정 (예: F_Node_i_j_k)
            m_name = f"{face_prefix[face_name]}_{node_idx[0]}_{node_idx[1]}_{node_idx[2]}"
            face_markers[face_name][m_name] = node_pos_final
            
        # -------------------------------------------------------------
        # [WHTOOLS Precision SVD Projection] 
        # SVD로 평면을 찾되, '바깥쪽 Normal'과 '표준 방향성' 제약을 가함
        # -------------------------------------------------------------
        all_node_names = sorted(list(face_markers[face_name].keys()))
        if not all_node_names: continue
        
        n_m = len(all_node_names)
        P0 = np.zeros((n_m, 3))
        for idx_m, m_name in enumerate(all_node_names):
            P0[idx_m] = face_markers[face_name][m_name][0]
            
        c_P = np.mean(P0, axis=0)
        P_centered = P0 - c_P
        U, S, Vt = np.linalg.svd(P_centered)
        
        # 1. 법선(Normal) 방향 제약: 박스 중심(0,0,0)에서 바깥쪽을 향하도록 함
        # Vt[2]는 평면의 법선. 마커 중심(c_P)이 원점 기준 법선 방향에 있어야 함
        normal = Vt[2]
        if np.dot(normal, c_P) < 0: normal = -normal
        
        # 2. 2D 플롯 축(Basis) 결정 제약: 이상적인 '가로', '세로' 벡터를 평면에 투영
        # WHTOOLS 표준: 평면의 첫 번째 축을 가로(h), 두 번째 축을 세로(v)로 기본 매핑
        h_sign = -1.0 if face_name in ["Rear", "Left"] else 1.0
        v_sign = -1.0 if face_name in ["Bottom", "Rear"] else 1.0 # Rear는 상하 반전 방지
        
        p_axes = face_logic["plane"]
        ax_map = {"i": [1,0,0], "j": [0,1,0], "k": [0,0,1]}
        h_ideal = np.array(ax_map[p_axes[0]]) * h_sign
        v_ideal = np.array(ax_map[p_axes[1]]) * v_sign
        
        # e1 (Horizontal): h_ideal을 평면에 투영
        e1 = h_ideal - np.dot(h_ideal, normal) * normal
        e1 /= (np.linalg.norm(e1) + 1e-9)
        
        # e2 (Vertical): normal과 e1의 외적 (오른손 법칙 유지하며 수직 확보)
        e2 = np.cross(normal, e1)
        if np.dot(e2, v_ideal) < 0: e2 = -e2
        
        # 3. 투영 수행 및 오프셋 저장 (t=0 기준 고정 맵 확보)
        for m_name in all_node_names:
            p_rel = face_markers[face_name][m_name][0] - c_P
            px = np.dot(p_rel, e1)
            py = np.dot(p_rel, e2)
            face_offsets[face_name][m_name] = np.array([px, py])

    return face_markers, face_offsets

def get_assembly_data_from_sim(result: DropSimResult, target_parts: List[str]) -> Tuple[Dict, Dict]:
    total_markers = {}
    total_offsets = {}
    for part in target_parts:
        m, o = extract_face_markers(result, part)
        total_markers[part] = m
        total_offsets[part] = o
    return total_markers, total_offsets
