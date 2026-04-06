# -*- coding: utf-8 -*-
"""
[WHTOOLS] TV Package Assembly Component Mapping (v4.x)
Source: GitHub (WonhoLee1/WHToolsBox) - refs/heads/D260329
"""

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
        node_accumulator = {} # 각 프레임별로 계산하기 위해 루프 내에서 관리하거나 벡터화 필요

        # 1. 각 블록 순회하며 해당 면의 4개 꼭짓점 기여분 수집
        for (i, j, k), body_id in body_map.items():
            # 이 블록이 해당 면에 속하는지 확인
            is_on_face = False
            # [v5.4.0] 사용자 요청 (Top=Front, Bottom=Rear) 방향에 맞춰 Y, Z 인덱스 스왑
            if face_name == "Front" and k == max_k: is_on_face = True
            elif face_name == "Rear" and k == 0: is_on_face = True
            elif face_name == "Left" and i == 0: is_on_face = True
            elif face_name == "Right" and i == max_i: is_on_face = True
            elif face_name == "Top" and j == 0: is_on_face = True
            elif face_name == "Bottom" and j == max_j: is_on_face = True

            if not is_on_face: continue

            # 블록 크기 정보
            dx, dy, dz = result.block_half_extents[body_id]

            # 면에 따른 4개 꼭짓점의 로컬 오프셋 정의
            # i, j, k 격자 시스템에서의 노드 인덱스 생성 (블록 i는 i와 i+1 노드 사이에 있음)
            # 로컬 좌표 매핑도 변경된 축에 맞춰 조정 (Y-Z 스왑 적용)
            # Front/Rear는 X-Y 평면 (Z 고정, 즉 k축), Top/Bottom은 X-Z 평면 (Y 고정, 즉 j축)으로 역할 변경
            if face_name in ["Top", "Bottom"]: # 원래 Front/Rear 역할 (Y축 고정)
                y_sign = -1 if face_name == "Top" else 1
                local_vertices = [
                    ([-dx, y_sign*dy, -dz], (i, j if face_name=="Top" else j+1, k)),
                    ([ dx, y_sign*dy, -dz], (i+1, j if face_name=="Top" else j+1, k)),
                    ([-dx, y_sign*dy,  dz], (i, j if face_name=="Top" else j+1, k+1)),
                    ([ dx, y_sign*dy,  dz], (i+1, j if face_name=="Top" else j+1, k+1))
                ]
            elif face_name in ["Left", "Right"]: # X 고정은 그대로
                x_sign = -1 if face_name == "Left" else 1
                local_vertices = [
                    ([x_sign*dx, -dy, -dz], (i if face_name=="Left" else i+1, j, k)),
                    ([x_sign*dx,  dy, -dz], (i if face_name=="Left" else i+1, j+1, k)),
                    ([x_sign*dx, -dy,  dz], (i if face_name=="Left" else i+1, j, k+1)),
                    ([x_sign*dx,  dy,  dz], (i if face_name=="Left" else i+1, j+1, k+1))
                ]
            else: # Front, Rear: 원래 Top/Bottom 역할 (Z축 고정)
                z_sign = 1 if face_name == "Front" else -1
                local_vertices = [
                    ([-dx, -dy, z_sign*dz], (i, j, k if face_name=="Rear" else k+1)),
                    ([ dx, -dy, z_sign*dz], (i+1, j, k if face_name=="Rear" else k+1)),
                    ([-dx,  dy, z_sign*dz], (i, j+1, k if face_name=="Rear" else k+1)),
                    ([ dx,  dy, z_sign*dz], (i+1, j+1, k if face_name=="Rear" else k+1))
                ]

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
        # [v5.3.17.1 FIX] 모든 노드 수집 후 물리 좌표 기반 오프셋 자동 추출 (SVD)
        # -------------------------------------------------------------
        all_node_names = sorted(list(face_markers[face_name].keys()))
        if not all_node_names: continue

        n_m = len(all_node_names)
        P0 = np.zeros((n_m, 3))
        for idx_m, m_name in enumerate(all_node_names):
            P0[idx_m] = face_markers[face_name][m_name][0] # t=0 시점

        # 주성분 분석 (SVD)를 통한 최적 평면 투영 및 크기 자동 산출
        c_P = np.mean(P0, axis=0)
        P_centered = P0 - c_P
        U, S, Vt = np.linalg.svd(P_centered)
        # Vt[:2]가 평면을 구성하는 가장 지배적인 두 주축임
        P_2d = P_centered @ Vt[:2, :].T 

        for idx_m, m_name in enumerate(all_node_names):
            face_offsets[face_name][m_name] = P_2d[idx_m]

    return face_markers, face_offsets

def get_assembly_data_from_sim(result: DropSimResult, target_parts: List[str]) -> Tuple[Dict, Dict]:
    total_markers = {}
    total_offsets = {}
    for part in target_parts:
        m, o = extract_face_markers(result, part)
        total_markers[part] = m
        total_offsets[part] = o
    return total_markers, total_offsets
