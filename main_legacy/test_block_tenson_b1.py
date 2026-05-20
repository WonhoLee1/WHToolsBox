import os
import numpy as np

# =====================================================================
# [1] 시스템 설정 및 파라미터
# =====================================================================
SYS_CONFIG = {
    "interface_mode": "contact", # 침투가 해결되었으므로 contact 모드 적극 권장
    "base_path": r'D:\PythonCodeStudy\mujoco-3.5.0-windows-x86_64\model'
}
os.makedirs(SYS_CONFIG["base_path"], exist_ok=True)

DIM = {
    "grid_N": 5, "grid_M": 4, "grid_L": 3,
    "cush_outer": [2.0, 1.4, 0.25],
    "drop_height": 0.5,
    "tv_ratio": 0.7,
    "thick_disp": 0.01,
    "thick_gap": 0.02, # 이 Gap이 침투를 막는 절대 공간이 됩니다.
    "thick_chas": 0.08
}

TV_DIM = [DIM["cush_outer"][0] * DIM["tv_ratio"], DIM["cush_outer"][1] * DIM["tv_ratio"]]
TV_TOTAL_THICK = DIM["thick_disp"] + DIM["thick_gap"] + DIM["thick_chas"]

WELD_STIFF = {
    "cushion": "0.006 1.0", "tv": "0.002 1.0", "bezel": "0.001 1.2"
}

# =====================================================================
# [2] ★ 핵심: 평면 분할을 통한 비균일 노드(Node) 좌표 생성
# =====================================================================
# 2-1. TV의 격자 노드 (N x M)
tv_nodes_x = np.linspace(-TV_DIM[0]/2, TV_DIM[0]/2, DIM["grid_N"] + 1)
tv_nodes_y = np.linspace(-TV_DIM[1]/2, TV_DIM[1]/2, DIM["grid_M"] + 1)

# 2-2. 쿠션의 격자 노드 (N+2 x M+2 x 3) - TV 경계면(Gap 포함)을 기준으로 분할
cush_nodes_x = [-DIM["cush_outer"][0]/2] + list(np.linspace(-TV_DIM[0]/2 - DIM["thick_gap"], TV_DIM[0]/2 + DIM["thick_gap"], DIM["grid_N"] + 1)) + [DIM["cush_outer"][0]/2]
cush_nodes_y = [-DIM["cush_outer"][1]/2] + list(np.linspace(-TV_DIM[1]/2 - DIM["thick_gap"], TV_DIM[1]/2 + DIM["thick_gap"], DIM["grid_M"] + 1)) + [DIM["cush_outer"][1]/2]
cush_nodes_z = [-DIM["cush_outer"][2]/2, -TV_TOTAL_THICK/2 - DIM["thick_gap"], TV_TOTAL_THICK/2 + DIM["thick_gap"], DIM["cush_outer"][2]/2]

def is_cavity(i, j, k):
    # i: 1~N, j: 1~M, k: 1 인덱스 구간이 정확히 TV가 들어갈 공동(Cavity)입니다.
    return (1 <= i <= DIM["grid_N"]) and (1 <= j <= DIM["grid_M"]) and (k == 1)

# ISTA 6A 회전 로직 (동일)
corner_vec = np.array([DIM["cush_outer"][0]/2, DIM["cush_outer"][1]/2, -DIM["cush_outer"][2]/2])
corner_dist = np.linalg.norm(corner_vec)
rot_axis = np.cross(corner_vec, [0, 0, -corner_dist])
rot_axis /= np.linalg.norm(rot_axis)
angle_rad = np.arccos(np.dot(corner_vec, [0, 0, -corner_dist]) / (corner_dist**2))
z_start = DIM["drop_height"] + corner_dist
ISTA_ROTATION = f"{rot_axis[0]:.4f} {rot_axis[1]:.4f} {rot_axis[2]:.4f} {np.degrees(angle_rad):.4f}"

def get_world_pose(lx, ly, lz):
    v = np.array([lx, ly, lz])
    v_rot = v * np.cos(angle_rad) + np.cross(rot_axis, v) * np.sin(angle_rad) + rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_rad))
    return v_rot + np.array([0, 0, z_start])

# =====================================================================
# [3] blocks.xml 생성
# =====================================================================
with open(os.path.join(SYS_CONFIG["base_path"], 'blocks.xml'), "w", encoding="utf-8") as fb:
    fb.write("<mujoco>\n")
    
    # [3-1] TV 블록 생성 (Gap을 제외한 순수 크기)
    for i in range(DIM["grid_N"]):
        for j in range(DIM["grid_M"]):
            cx = (tv_nodes_x[i] + tv_nodes_x[i+1]) / 2
            cy = (tv_nodes_y[j] + tv_nodes_y[j+1]) / 2
            dx = (tv_nodes_x[i+1] - tv_nodes_x[i]) / 2
            dy = (tv_nodes_y[j+1] - tv_nodes_y[j]) / 2
            
            # 디스플레이
            wx, wy, wz = get_world_pose(cx, cy, TV_TOTAL_THICK/2 - DIM["thick_disp"]/2)
            fb.write(f'  <body name="b_disp_{i}_{j}" pos="{wx:.4f} {wy:.4f} {wz:.4f}" axisangle="{ISTA_ROTATION}"><freejoint/>\n')
            fb.write(f'    <geom type="box" size="{dx:.4f} {dy:.4f} {DIM["thick_disp"]/2:.4f}" rgba="0.1 0.5 0.8 0.9" mass="0.15" contype="2" conaffinity="5"/>\n')
            fb.write(f'    <site name="s_disp_{i}_{j}_PX" pos="{dx:.4f} 0 0"/><site name="s_disp_{i}_{j}_NX" pos="{-dx:.4f} 0 0"/>\n')
            fb.write(f'    <site name="s_disp_{i}_{j}_PY" pos="0 {dy:.4f} 0"/><site name="s_disp_{i}_{j}_NY" pos="0 {-dy:.4f} 0"/>\n')
            fb.write(f'    <site name="s_disp_{i}_{j}_BZ" pos="0 0 {-DIM["thick_disp"]/2:.4f}"/>\n  </body>\n')
            
            # 섀시
            wx, wy, wz = get_world_pose(cx, cy, -TV_TOTAL_THICK/2 + DIM["thick_chas"]/2)
            fb.write(f'  <body name="b_chas_{i}_{j}" pos="{wx:.4f} {wy:.4f} {wz:.4f}" axisangle="{ISTA_ROTATION}"><freejoint/>\n')
            fb.write(f'    <geom type="box" size="{dx:.4f} {dy:.4f} {DIM["thick_chas"]/2:.4f}" rgba="0.3 0.3 0.3 0.9" mass="0.15" contype="2" conaffinity="5"/>\n')
            fb.write(f'    <site name="s_chas_{i}_{j}_PX" pos="{dx:.4f} 0 0"/><site name="s_chas_{i}_{j}_NX" pos="{-dx:.4f} 0 0"/>\n')
            fb.write(f'    <site name="s_chas_{i}_{j}_PY" pos="0 {dy:.4f} 0"/><site name="s_chas_{i}_{j}_NY" pos="0 {-dy:.4f} 0"/>\n')
            fb.write(f'    <site name="s_chas_{i}_{j}_BZ" pos="0 0 {DIM["thick_chas"]/2:.4f}"/>\n  </body>\n')

    # [3-2] 맞춤형 쿠션 블록 생성
    for i in range(DIM["grid_N"] + 2):
        for j in range(DIM["grid_M"] + 2):
            for k in range(3):
                if is_cavity(i, j, k): continue # TV 공간은 완벽하게 비움
                
                cx = (cush_nodes_x[i] + cush_nodes_x[i+1]) / 2
                cy = (cush_nodes_y[j] + cush_nodes_y[j+1]) / 2
                cz = (cush_nodes_z[k] + cush_nodes_z[k+1]) / 2
                
                dx = (cush_nodes_x[i+1] - cush_nodes_x[i]) / 2
                dy = (cush_nodes_y[j+1] - cush_nodes_y[j]) / 2
                dz = (cush_nodes_z[k+1] - cush_nodes_z[k]) / 2
                
                wx, wy, wz = get_world_pose(cx, cy, cz)
                fb.write(f'  <body name="b_cush_{i}_{j}_{k}" pos="{wx:.4f} {wy:.4f} {wz:.4f}" axisangle="{ISTA_ROTATION}"><freejoint/>\n')
                fb.write(f'    <geom type="box" size="{dx:.4f} {dy:.4f} {dz:.4f}" rgba="0.8 0.8 0.8 0.4" mass="0.03" contype="4" conaffinity="3"/>\n')
                # 가변 크기에 맞춘 Weld 사이트
                fb.write(f'    <site name="s_cush_{i}_{j}_{k}_PX" pos="{dx:.4f} 0 0"/><site name="s_cush_{i}_{j}_{k}_NX" pos="{-dx:.4f} 0 0"/>\n')
                fb.write(f'    <site name="s_cush_{i}_{j}_{k}_PY" pos="0 {dy:.4f} 0"/><site name="s_cush_{i}_{j}_{k}_NY" pos="0 {-dy:.4f} 0"/>\n')
                fb.write(f'    <site name="s_cush_{i}_{j}_{k}_PZ" pos="0 0 {dz:.4f}"/><site name="s_cush_{i}_{j}_{k}_NZ" pos="0 0 {-dz:.4f}"/>\n  </body>\n')
    fb.write("</mujoco>\n")

# =====================================================================
# [4] main.xml 생성 (동적 Weld 연결)
# =====================================================================
with open(os.path.join(SYS_CONFIG["base_path"], 'main.xml'), "w", encoding="utf-8") as fm:
    fm.write(f"""<mujoco model="tv_package_perfect_mesh">
    <option integrator="implicitfast" timestep="0.001" ><flag contact="enable"/></option>
    <default>
        <joint armature="0.05" damping="1.0"/>
        <geom friction="0.8" solref="0.02 1.0" solimp="0.9 0.95 0.001"/>
    </default>
    <worldbody>
        <light pos="0 0 5" dir="0 0 -1"/><geom name="floor" type="plane" size="5 5 0.1" friction="0.8" contype="1" conaffinity="1"/>
        <include file="blocks.xml"/>
    </worldbody>
    <equality>
""")
    # 쿠션 격자 조립 (동적 인덱싱)
    for i in range(DIM["grid_N"] + 2):
        for j in range(DIM["grid_M"] + 2):
            for k in range(3):
                if is_cavity(i, j, k): continue
                if i < DIM["grid_N"]+1 and not is_cavity(i+1, j, k): fm.write(f'        <weld site1="s_cush_{i}_{j}_{k}_PX" site2="s_cush_{i+1}_{j}_{k}_NX" solref="{WELD_STIFF["cushion"]}"/>\n')
                if j < DIM["grid_M"]+1 and not is_cavity(i, j+1, k): fm.write(f'        <weld site1="s_cush_{i}_{j}_{k}_PY" site2="s_cush_{i}_{j+1}_{k}_NY" solref="{WELD_STIFF["cushion"]}"/>\n')
                if k < 2 and not is_cavity(i, j, k+1): fm.write(f'        <weld site1="s_cush_{i}_{j}_{k}_PZ" site2="s_cush_{i}_{j}_{k+1}_NZ" solref="{WELD_STIFF["cushion"]}"/>\n')
                    
    # TV 격자 조립
    for i in range(DIM["grid_N"]):
        for j in range(DIM["grid_M"]):
            if i < DIM["grid_N"]-1:
                fm.write(f'        <weld site1="s_disp_{i}_{j}_PX" site2="s_disp_{i+1}_{j}_NX" solref="{WELD_STIFF["tv"]}"/>\n')
                fm.write(f'        <weld site1="s_chas_{i}_{j}_PX" site2="s_chas_{i+1}_{j}_NX" solref="{WELD_STIFF["tv"]}"/>\n')
            if j < DIM["grid_M"]-1:
                fm.write(f'        <weld site1="s_disp_{i}_{j}_PY" site2="s_disp_{i}_{j+1}_NY" solref="{WELD_STIFF["tv"]}"/>\n')
                fm.write(f'        <weld site1="s_chas_{i}_{j}_PY" site2="s_chas_{i}_{j+1}_NY" solref="{WELD_STIFF["tv"]}"/>\n')
            if i==0 or i==DIM["grid_N"]-1 or j==0 or j==DIM["grid_M"]-1: 
                fm.write(f'        <weld site1="s_disp_{i}_{j}_BZ" site2="s_chas_{i}_{j}_BZ" solref="{WELD_STIFF["bezel"]}"/>\n')

    fm.write("    </equality>\n</mujoco>")
print(">>> [비균일 격자 생성 완료] 겹침(Overlap)이 완전히 제거된 맞춤형 쿠션과 TV가 생성되었습니다.")
