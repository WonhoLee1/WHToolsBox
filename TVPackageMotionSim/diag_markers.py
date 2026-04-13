"""
[WHTOOLS] 마커 추출 진단 스크립트
chassis 파트의 block_half_extents, pos_hist, 그리고 d_val 매핑을 출력하여
마커 위치 오류의 원인을 진단합니다.
"""
import os, sys, pickle, glob
import numpy as np

sys.path.append(os.getcwd())
from run_drop_simulator.whts_mapping import quat_to_mat, get_face_index_logic
from run_drop_simulator.whts_multipostprocessor_engine import scale_result_to_mm

def get_latest_result_dir():
    dirs = glob.glob("rds-*")
    if not dirs: return None
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]

def diagnose_markers(result):
    result = scale_result_to_mm(result)  # m -> mm 변환

    comp_names = list(result.components.keys())
    print(f"[DIAG] Available components: {comp_names}")

    # chassis 컴포넌트 찾기
    chassis_key = None
    for k in comp_names:
        if 'chassis' in k:
            chassis_key = k
            break

    if chassis_key is None:
        print("[DIAG] ERROR: 'chassis' component not found!")
        return

    body_map = result.components[chassis_key]
    pos_hist = np.array(result.pos_hist)
    quat_hist = np.array(result.quat_hist)

    print(f"\n[DIAG] Component: '{chassis_key}'")
    print(f"[DIAG] Total blocks: {len(body_map)}")
    print(f"\n[DIAG] Block index (i,j,k) -> body_id mapping:")

    indices = list(body_map.keys())
    max_i = max(idx[0] for idx in indices)
    max_j = max(idx[1] for idx in indices)
    max_k = max(idx[2] for idx in indices)
    print(f"  max_i={max_i}, max_j={max_j}, max_k={max_k}")

    # 초기 프레임의 각 바디 위치 출력
    print(f"\n[DIAG] Block positions at frame 0 (mm):")
    for (i, j, k), body_id in sorted(body_map.items()):
        pos0 = pos_hist[0, body_id]
        extents = result.block_half_extents.get(body_id, [0,0,0])
        print(f"  ({i},{j},{k}) body={body_id:3d}: pos={pos0.round(1)}, half_extents={np.round(extents,1)}")

    # Front 면 마커 계산 진단
    print(f"\n[DIAG] Front face (k=max_k={max_k}) marker positions:")
    face_logic = get_face_index_logic("Front", (max_i, max_j, max_k))
    target_axis = face_logic["axis"]
    target_idx  = face_logic["idx"]
    p_axes      = face_logic["plane"]
    norm_vec    = np.array(face_logic["normal"])

    print(f"  target_axis={target_axis}, target_idx={target_idx}, p_axes={p_axes}")

    node_accumulator = {}
    for (i, j, k), body_id in body_map.items():
        is_on_face = (target_axis == "i" and i == target_idx) or \
                     (target_axis == "j" and j == target_idx) or \
                     (target_axis == "k" and k == target_idx)
        if not is_on_face: continue

        dx, dy, dz = result.block_half_extents.get(body_id, [0.0, 0.0, 0.0])
        d_val = {"i": dx, "j": dy, "k": dz}

        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                lv = np.zeros(3)
                lv_idx = 0 if target_axis=="i" else 1 if target_axis=="j" else 2
                lv[lv_idx] = norm_vec[lv_idx] * d_val[target_axis]
                p1_idx = 0 if p_axes[0]=="i" else 1 if p_axes[0]=="j" else 2
                p2_idx = 0 if p_axes[1]=="i" else 1 if p_axes[1]=="j" else 2
                lv[p1_idx] = s1 * d_val[p_axes[0]]
                lv[p2_idx] = s2 * d_val[p_axes[1]]

                ni = [i, j, k]
                ni[p1_idx] = ni[p1_idx] if s1 < 0 else ni[p1_idx] + 1
                ni[p2_idx] = ni[p2_idx] if s2 < 0 else ni[p2_idx] + 1
                node_idx = tuple(ni)

                if node_idx not in node_accumulator: node_accumulator[node_idx] = []
                node_accumulator[node_idx].append((body_id, lv))

    print(f"\n  Total merged nodes: {len(node_accumulator)}")
    print(f"  Node world positions at frame 0:")
    for node_idx, contribs in sorted(node_accumulator.items()):
        pos_list = []
        for body_id, l_vec in contribs:
            R = quat_to_mat(quat_hist[0, body_id])
            p = pos_hist[0, body_id] + R @ l_vec
            pos_list.append(p)
        avg = np.mean(pos_list, axis=0)
        print(f"    node{node_idx}: avg_world_pos={avg.round(1)} mm  (from {len(contribs)} blocks)")

if __name__ == "__main__":
    target_dir = get_latest_result_dir()
    print(f"Loading: {target_dir}")
    pkl_path = os.path.join(target_dir, "simulation_result.pkl")
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    diagnose_markers(result)
