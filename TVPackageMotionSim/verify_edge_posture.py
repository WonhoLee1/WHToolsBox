import numpy as np
from run_discrete_builder.whtb_utils import parse_drop_target, get_drop_orientation_matrix, mat2axisangle

def verify():
    # 박스 치수
    box_w, box_h, box_d = 1.841, 1.103, 0.170
    drop_height = 0.5
    
    print("--- LTL Mode Edge/Corner Posture Verification ---")
    
    # 테스트 케이스들: 모든 12개 LTL Edge 케이스와 코너 케이스 추가
    test_cases = [
        ("LTL", "Edge 1-2"),
        ("LTL", "Edge 1-4"),
        ("LTL", "Edge 1-5"),
        ("LTL", "Edge 1-6"),
        ("LTL", "Edge 2-3"),
        ("LTL", "Edge 2-5"),
        ("LTL", "Edge 2-6"),
        ("LTL", "Edge 3-4"),
        ("LTL", "Edge 3-5"),
        ("LTL", "Edge 3-6"),
        ("LTL", "Edge 4-5"),
        ("LTL", "Edge 4-6"),
        ("LTL", "Corner 2-3-5"),
    ]
    
    for mode, direction in test_cases:
        print(f"\n[Mode: {mode}, Direction: {direction}]")
        target_pt = parse_drop_target(mode, direction, box_w, box_h, box_d)
        target_dist = np.linalg.norm(target_pt)
        print(f"  Target Point (Local): {target_pt}")
        
        # whtb_builder.py 의 로직 재현
        if mode == "LTL":
            ref_vec = np.array([0.0, 0.0, 1.0]) if target_pt[2] < 0 else np.array([0.0, 0.0, -1.0])
            R_final = get_drop_orientation_matrix(target_pt, ref_vec, global_ref_target=np.array([0, -1, 0]))
            print(f"  ref_vec: {ref_vec}")
        else:
            rot_axis = np.cross(target_pt, [0, 0, -target_dist])
            if np.linalg.norm(rot_axis) < 1e-6:
                rot_axis = np.array([1.0, 0.0, 0.0])
                angle_rad = 0.0 if target_pt[2] < 0 else np.pi
            else:
                rot_axis /= np.linalg.norm(rot_axis)
                dot_val = np.clip(np.dot(target_pt, [0, 0, -target_dist]) / (target_dist**2), -1, 1)
                angle_rad = np.arccos(dot_val)
            
            K = np.array([[0, -rot_axis[2], rot_axis[1]], 
                          [rot_axis[2], 0, -rot_axis[0]], 
                          [-rot_axis[1], rot_axis[0], 0]])
            R_final = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
            print(f"  rot_axis: {rot_axis}, angle_deg: {np.degrees(angle_rad):.4f}")
        print(f"  R_final:\n{R_final}")
        
        # target_pt 의 글로벌 위치 계산
        target_pt_global = R_final @ target_pt
        print(f"  Target Point (Global after R_final): {target_pt_global}")
        
        # 박스의 8개 꼭지점 정의
        half_w, half_h, half_d = box_w/2, box_h/2, box_d/2
        corners = {
            "C1": np.array([-half_w, -half_h, -half_d]),
            "C2": np.array([half_w, -half_h, -half_d]),
            "C3": np.array([half_w, half_h, -half_d]),
            "C4": np.array([-half_w, half_h, -half_d]),
            "C5": np.array([-half_w, -half_h, half_d]),
            "C6": np.array([half_w, -half_h, half_d]),
            "C7": np.array([half_w, half_h, half_d]),
            "C8": np.array([-half_w, half_h, half_d]),
        }
        
        # 8개 꼭지점의 글로벌 위치 계산 (회전 적용)
        global_corners = {}
        for name, pt in corners.items():
            gpt = R_final @ pt
            gpt_shifted = gpt - target_pt_global
            global_corners[name] = gpt_shifted
            print(f"    {name} (Local: {pt}) -> Global Shifted (Z-height): {gpt_shifted[2]:.6f} (Pt: {gpt_shifted})")

if __name__ == "__main__":
    verify()
