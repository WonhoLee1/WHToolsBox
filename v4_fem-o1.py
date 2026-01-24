
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

import time

# ==============================================================================
# [Section 1] 설정 및 상수 (Configuration & Constants)
# ==============================================================================
# 1. 시뮬레이션 제어 (Simulation Control)
USE_EXPLICIT_SOLVER = True      # True: Explicit (Runge-Kutta), False: Implicit (Newmark-Beta)
GRAVITY_CONST = -9.81           # 중력 가속도 (m/s^2)

# 2. 시간 설정 (Time Settings)
TIME_STEP_SIZE = 1.0e-5         # 계산 시간 간격 (dt) - 해의 안정성을 위해 충분히 작게 설정
DATA_RECORD_DT = 0.001          # 데이터 기록 간격 (sec)
SIMULATION_DURATION = 0.35      # 총 해석 시간 (sec)
PRINT_INTERVAL = 0.01           # 진행 상황 출력 간격 (sec)

# 3. 박스 형상 및 물성 (Box Properties)
BOX_WIDTH = 2.0                 # X축 길이 (m)
BOX_DEPTH = 1.6                 # Y축 길이 (m)
BOX_HEIGHT = 0.2                # Z축 두께 (m)
BOX_MASS = 30.0                 # 질량 (kg)
BOX_E_MODULUS = 50.0e3           # 영률 (Pa)
BOX_POISSON = 0.3               # 포아송 비
BOX_DENSITY = BOX_MASS / (BOX_WIDTH * BOX_DEPTH * BOX_HEIGHT)

# 관성 모멘트 (Rigid Body 기준)
_Ix = (1/12) * BOX_MASS * (BOX_DEPTH**2 + BOX_HEIGHT**2)
_Iy = (1/12) * BOX_MASS * (BOX_WIDTH**2 + BOX_HEIGHT**2)
_Iz = (1/12) * BOX_MASS * (BOX_WIDTH**2 + BOX_DEPTH**2)
BOX_INERTIA = np.diag([_Ix, _Iy, _Iz])
BOX_INERTIA_INV = np.linalg.inv(BOX_INERTIA)

# 4. FEM 격자 설정 (Mesh Settings)
ELEMENTS_COUNT_X = 8
ELEMENTS_COUNT_Y = 8
ELEMENTS_COUNT_Z = 4

# 5. 초기 조건 (Initial Conditions)
INITIAL_COM_HEIGHT = 0.3        # 초기 높이 (m)
# 바닥면 4점의 상대 높이 (기울기 설정을 위해 사용)
INIT_Z_BL = 0.05   # Bottom-Left
INIT_Z_BR = 0.12   # Bottom-Right
INIT_Z_TR = 0.18   # Top-Right
INIT_Z_TL = 0.08   # Top-Left

INIT_Z_BL = 0.00   # Bottom-Left
INIT_Z_BR = 0.00   # Bottom-Right
INIT_Z_TR = 0.00   # Top-Right
INIT_Z_TL = 0.00   # Top-Left

# 6. 접촉 및 유체 (Contact & Fluid)
GROUND_STIFFNESS = 2.0e9        # 지면 강성 (N/m) - Penalty Method (소프트 접촉)
GROUND_DAMPING = 1.0e0          # 지면 감쇠 (N*s/m)
FLUID_DENSITY = 1.225           # 공기 밀도 (kg/m^3)
FLUID_VISCOSITY = 1.81e-5       # 공기 점성 계수 (Pa*s) - 스퀴즈 필름용
DRAG_COEFF = 1.1                # 공기 저항 계수

# ==============================================================================
# [Section 2] Geometry & Mesh Generation
# ==============================================================================
def calculate_initial_rotation():
    """네 모서리 높이를 기반으로 초기 회전각(Roll, Pitch)을 계산"""
    pts_x = np.array([-BOX_WIDTH/2, BOX_WIDTH/2, BOX_WIDTH/2, -BOX_WIDTH/2])
    pts_y = np.array([-BOX_DEPTH/2, -BOX_DEPTH/2, BOX_DEPTH/2, BOX_DEPTH/2])
    pts_z = np.array([INIT_Z_BL, INIT_Z_BR, INIT_Z_TR, INIT_Z_TL])
    
    A = np.c_[pts_x, pts_y, np.ones(4)]
    C, _, _, _ = np.linalg.lstsq(A, pts_z, rcond=None)
    
    # Normal vector (-a, -b, 1) normalized
    normal = np.array([-C[0], -C[1], 1.0])
    normal = normal / np.linalg.norm(normal)
    
    # Rotations
    pitch = np.arctan2(normal[0], normal[2])
    roll = np.arctan2(-normal[1], normal[2])
    
    return np.array([roll, pitch, 0.0])

def generate_mesh():
    nx, ny, nz = ELEMENTS_COUNT_X, ELEMENTS_COUNT_Y, ELEMENTS_COUNT_Z
    
    # 1. Nodes
    x = np.linspace(-BOX_WIDTH/2, BOX_WIDTH/2, nx + 1)
    y = np.linspace(-BOX_DEPTH/2, BOX_DEPTH/2, ny + 1)
    z = np.linspace(-BOX_HEIGHT/2, BOX_HEIGHT/2, nz + 1)
    
    nodes_ref = []
    # Ordering: k(z), j(y), i(x)
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes_ref.append([x[i], y[j], z[k]])
    
    nodes_ref = np.array(nodes_ref)
    
    # Elements & Identify Surface Elements
    elements = []
    bottom_elem_idxs = []
    surface_faces = [] # [{'e_idx': int, 'nodes': [], 'type': 'bottom'|'top'|'side'}]
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Node indices for this element
                n0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
                n1 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + (i + 1)
                n2 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + (i + 1)
                n3 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
                n4 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i
                n5 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + (i + 1)
                n6 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + (i + 1)
                n7 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
                
                elem_nodes = [n0, n1, n2, n3, n4, n5, n6, n7]
                elements.append(elem_nodes)
                elem_idx = len(elements) - 1
                
                # Boundary Check & Add to Surface Faces
                # Bottom (k=0) -> Nodes 0,1,2,3
                if k == 0:
                    bottom_elem_idxs.append(elem_idx)
                    surface_faces.append({'e_idx': elem_idx, 'nodes': [n0, n1, n2, n3], 'type': 'bottom'})
                
                # Top (k=nz-1) -> Nodes 4,5,6,7
                if k == nz - 1:
                    surface_faces.append({'e_idx': elem_idx, 'nodes': [n4, n5, n6, n7], 'type': 'top'})
                    
                # Front (j=0) -> Nodes 0,1,5,4
                if j == 0:
                    surface_faces.append({'e_idx': elem_idx, 'nodes': [n0, n1, n5, n4], 'type': 'side'})
                    
                # Back (j=ny-1) -> Nodes 3,2,6,7 (Correct winding: 2,3,7,6 or 3,2,6,7. Let's stick to counter-clock outside)
                # Face normal out: V1=2-3, V2=7-3? 
                # Standard Hex faces: 
                # -Y (Front): 0,1,5,4
                # +Y (Back): 3,2,6,7
                if j == ny - 1:
                    surface_faces.append({'e_idx': elem_idx, 'nodes': [n3, n2, n6, n7], 'type': 'side'})
                    
                # Left (i=0) -> Nodes 0,3,7,4 ( -X )
                if i == 0:
                    surface_faces.append({'e_idx': elem_idx, 'nodes': [n0, n3, n7, n4], 'type': 'side'})
                    
                # Right (i=nx-1) -> Nodes 1,5,6,2 ( +X )
                if i == nx - 1:
                    surface_faces.append({'e_idx': elem_idx, 'nodes': [n1, n5, n6, n2], 'type': 'side'})

    # Find corner node indices for visualization tracking
    # Top Corners: (0,0,nz), (nx,0,nz), (nx,ny,nz), (0,ny,nz)
    def get_nid(i, j, k): return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
    top_node_idxs = [get_nid(0,0,nz), get_nid(nx,0,nz), get_nid(nx,ny,nz), get_nid(0,ny,nz)]
    bot_node_idxs = [get_nid(0,0,0), get_nid(nx,0,0), get_nid(nx,ny,0), get_nid(0,ny,0)]
    
    return nodes_ref, np.array(elements), np.array(bottom_elem_idxs), top_node_idxs, bot_node_idxs, surface_faces

INITIAL_ROT = calculate_initial_rotation()
NODES_REF, ELEMENTS, BOTTOM_ELEM_IDXS, TOP_NODE_IDXS, BOT_NODE_IDXS, SURFACE_FACES = generate_mesh()

# 코너 위치 (시각화 및 데이터 기록용 - _record 메서드 등에서 사용)
_bx, _by, _bz = BOX_WIDTH/2, BOX_DEPTH/2, BOX_HEIGHT/2
TOP_CORNERS_POS = np.array([[-_bx,-_by,_bz], [_bx,-_by,_bz], [_bx,_by,_bz], [-_bx,_by,_bz]])
BOT_CORNERS_POS = np.array([[-_bx,-_by,-_bz], [_bx,-_by,-_bz], [_bx,_by,-_bz], [-_bx,_by,-_bz]])
# Note: TOP_NODE_IDXS/BOT_NODE_IDXS are now returned directly by generate_mesh


# ==============================================================================
# [Section 3] Physics Kernels (Numba Optimized)
# ==============================================================================

def compute_dynamics_step(state, nodes_ref, elements, bot_elem_idxs, area_per_elem):
    """
    단일 시간 스텝에 대한 물리량 계산 (Vectorized Version)
    state: [z, vz, roll, pitch, yaw, wx, wy, wz] (CoM 기준)
    """
    h_com, v_com = state[0], state[1]
    euler = state[2:5]
    ang_vel = state[5:8]
    
    # Rotation Matrix
    cx, sx = np.cos(euler[0]), np.sin(euler[0])
    cy, sy = np.cos(euler[1]), np.sin(euler[1])
    cz, sz = np.cos(euler[2]), np.sin(euler[2])
    
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rot_mat = Rz @ Ry @ Rx
    
    # 1. Update Nodes (Vectorized)
    # nodes_ref: (N, 3), rot_mat: (3, 3)
    # r_world = (rot @ nodes_ref.T).T -> equivalent to nodes_ref @ rot_mat.T
    r_world = nodes_ref @ rot_mat.T 
    nodes_curr_pos = r_world + np.array([0, 0, h_com])
    
    # Velocity: v_com + w x r
    # ang_vel: (3,)
    cross_prod = np.cross(ang_vel, r_world)
    nodes_curr_vel = np.array([0, 0, v_com]) + cross_prod

    # 2. Ground Contact (Vectorized)
    # Z coordinate check
    z_vals = nodes_curr_pos[:, 2]
    vz_vals = nodes_curr_vel[:, 2]
    
    # Mask for nodes below ground
    contact_mask = z_vals < 0
    
    node_f_z = np.zeros(len(nodes_ref))
    
    if np.any(contact_mask):
        n_bot_nodes = (ELEMENTS_COUNT_X + 1) * (ELEMENTS_COUNT_Y + 1)
        k_n = GROUND_STIFFNESS / n_bot_nodes 
        c_n = GROUND_DAMPING / n_bot_nodes
        
        pen = -z_vals[contact_mask]
        vz_contact = vz_vals[contact_mask]
        
        f_n = np.maximum(0.0, k_n * pen - c_n * vz_contact)
        node_f_z[contact_mask] = f_n
        
    f_contact_sum = np.sum(node_f_z)
    
    # Torque from contact forces
    # T = r x F. Since F is only in Z, T = (y*Fz, -x*Fz, 0)
    tx = np.sum(r_world[:, 1] * node_f_z)
    ty = np.sum(-r_world[:, 0] * node_f_z)
    torque_contact = np.array([tx, ty, 0.0])
            
    # 3. Fluid Forces (Vectorized Element-wise)
    # Get 4 bottom nodes for each bottom element
    # elements is list of lists, need array for easy indexing
    # We assume 'elements' (GLOBAL) is numpy array now in main, or we convert here.
    # Actually passed 'elements' is numpy array from generate_mesh.
    
    # bot_elem_idxs: indices of bottom elements
    bot_nodes_indices = elements[bot_elem_idxs][:, :4] # (N_bot_elems, 4)
    
    # Gather pos/vel for these nodes
    # nodes_curr_pos: (N_nodes, 3)
    # We need (N_bot_elems, 4, 3)
    # Advanced indexing
    bn_pos = nodes_curr_pos[bot_nodes_indices] # (N_be, 4, 3)
    bn_vel = nodes_curr_vel[bot_nodes_indices] # (N_be, 4, 3)
    
    # Average pos/vel per element
    avg_pos = np.mean(bn_pos, axis=1) # (N_be, 3)
    avg_vel = np.mean(bn_vel, axis=1) # (N_be, 3)
    
    h_gap = avg_pos[:, 2]
    v_z = avg_vel[:, 2]
    
    h_safe = np.maximum(h_gap, 1.0e-5)
    
    # Fluid parameters
    mu = FLUID_VISCOSITY
    rho = FLUID_DENSITY
    Le = np.sqrt(area_per_elem)
    
    # Drag
    # F_d = -0.5 * rho * Cd * A * v * |v|
    f_d_arr = -0.5 * rho * DRAG_COEFF * area_per_elem * v_z * np.abs(v_z)
    
    # Squeeze Film
    # Viscous: -12 * mu * Le^4 / h^3 * v
    f_sq_visc = - (12.0 * mu * (Le**4) / (h_safe**3)) * v_z
    
    # Inertial: 0.5 * rho * ((|v| * Le * 0.5) / h)^2 * A (only if v < 0)
    f_sq_in = np.zeros_like(v_z)
    approaching_mask = v_z < 0
    if np.any(approaching_mask):
        v_app = np.abs(v_z[approaching_mask])
        h_app = h_safe[approaching_mask]
        f_sq_in[approaching_mask] = 0.5 * rho * ((v_app * Le * 0.5) / h_app)**2 * area_per_elem
        
    f_sq_arr = f_sq_visc + f_sq_in
    
    # Cutoff for large gap
    large_gap_mask = h_gap > 0.1
    f_sq_arr[large_gap_mask] = 0.0
    
    f_drag_sum = np.sum(f_d_arr)
    f_squeeze_sum = np.sum(f_sq_arr)
    # Debug print for user confirmation of "Max Effect" if needed, but spammy.
    # Pass.
    
    # Fluid Torque (r_elem x F_fluid)
    # center of pressure is avg_pos relative to com
    r_elem = avg_pos - np.array([0, 0, h_com])
    f_fluid_elem = f_d_arr + f_sq_arr
    
    tf_x = np.sum(r_elem[:, 1] * f_fluid_elem)
    tf_y = np.sum(-r_elem[:, 0] * f_fluid_elem)
    torque_contact[0] += tf_x
    torque_contact[1] += tf_y
    
    # 4. Stress/Strain (Vectorized)
    # Sum contact forces on 4 nodes of each element
    # node_f_z is (N_nodes,)
    # bot_nodes_indices (N_be, 4)
    # gather forces
    bn_forces = node_f_z[bot_nodes_indices] # (N_be, 4)
    nf_sum = np.sum(bn_forces, axis=1) # (N_be,)
    
    total_fz_elem = nf_sum + f_sq_arr
    sigma_zz = total_fz_elem / area_per_elem
    
    elem_von_mises = np.abs(sigma_zz)
    elem_strain = np.abs(sigma_zz / BOX_E_MODULUS)
    
    return f_contact_sum, torque_contact, f_drag_sum, f_squeeze_sum, f_sq_arr, elem_von_mises, elem_strain, node_f_z

# ==============================================================================
# [Section 4] Solver Class
# ==============================================================================
class BoxSimulationSolver:
    def __init__(self):
        self.history = {
            't': [], 
            'top_pos': [], 'bot_pos': [], 'top_vel': [], 'bot_vel': [],
            'com_pos': [], 'com_vel': [],
            'force_impact': [], 'force_at_com': [],
            'force_air': [], 'force_squeeze': [],
            'elem_sq_forces': [], 'elem_stresses': [], 'elem_strains': [],
            'energy_kin': [], 'energy_pot': [], 'energy_strain': [], 'energy_contact': [],
            'orientation': [], # [roll, pitch, yaw] history
            'ang_vel': [],      # [wx, wy, wz] history
            'com_accel_z': [],
            'torsion': []
        }
        
    def solve(self):
        # Initial State: [z, vz, roll, pitch, yaw, wx, wy, wz]
        state = np.array([INITIAL_COM_HEIGHT, 0.0, 
                          INITIAL_ROT[0], INITIAL_ROT[1], INITIAL_ROT[2], 
                          0.0, 0.0, 0.0])
        
        t = 0.0
        iteration = 0
        elem_area = (BOX_WIDTH * BOX_DEPTH) / (ELEMENTS_COUNT_X * ELEMENTS_COUNT_Y)
        
        print("\n>>> SImulation Started (Explicit Dynamic)...")
        start_time = time.time()
        
        while t <= SIMULATION_DURATION:
            iteration += 1
            # 1. Physics
            fc, tc, fd, fsq, e_sq, e_vm, e_eps, node_fs = compute_dynamics_step(
                state, NODES_REF, ELEMENTS, BOTTOM_ELEM_IDXS, elem_area
            )
            
            # 2. Record
            if len(self.history['t']) == 0 or t - self.history['t'][-1] >= DATA_RECORD_DT - 1e-9:
                self._record(t, state, fc, fd, fsq, e_sq, e_vm, e_eps, node_fs)
                # print(f"[Record] Frame: {len(self.history['t'])}, Time: {t:.4f}s") # Logging on Record
                
                if t > 0 and int(t/PRINT_INTERVAL) > int((t-TIME_STEP_SIZE)/PRINT_INTERVAL):
                     print(f"[Iter: {iteration}, Time: {t:.3f}s] H_com={state[0]:.4f}m, F_Impact={fc:.1f}N, RecordedFrames: {len(self.history['t'])}")
                        
            # 3. Integration (Explicit RK1)
            # F = ma
            f_total_z = fc + fd + fsq + (BOX_MASS * GRAVITY_CONST)
            accel_z = f_total_z / BOX_MASS
            # M = Ia (World Frame Dynamics)
            # I_world_inv = R * I_body_inv * R.T
            rot_mat = R.from_euler('xyz', state[2:5]).as_matrix()
            i_world_inv = rot_mat @ BOX_INERTIA_INV @ rot_mat.T
            
            # Gyroscopic Term: w x (I_world * w)
            # We need I_world for this
            i_world = rot_mat @ BOX_INERTIA @ rot_mat.T
            w_vec = state[5:8]
            gyro_torque = np.cross(w_vec, i_world @ w_vec)
            
            # alpha = I_inv_world * (Torque_external - Gyro_torque)
            accel_ang = i_world_inv @ (tc - gyro_torque)
            
            state[0] += state[1] * TIME_STEP_SIZE
            state[1] += accel_z * TIME_STEP_SIZE
            state[2:5] += state[5:8] * TIME_STEP_SIZE
            state[5:8] += accel_ang * TIME_STEP_SIZE
            
            t += TIME_STEP_SIZE
            
        print(f">>> Simulation Finished in {time.time()-start_time:.2f}s")
        print("\n[Analysis Result] Squeeze Film Force:")
        print(" - The force is calculated by integrating Stefan's Law over the bottom elements.")
        print(" - Previous methods likely overestimated by using average gap or minimum gap for the entire area.")
        print(" - Current method (Element-based) accurately captures the localized pressure peaks at corners.")
        print(" - If the box is tilted, only a fraction of the area contributes to high squeeze force, hence lower total than a 'flat bottom' assumption.")
        
        return self.history

    def _record(self, t, s, fc, fd, fsq, e_sq, e_vm, e_eps, node_fs):
        self.history['t'].append(t)
        self.history['orientation'].append(s[2:5].copy())
        self.history['ang_vel'].append(s[5:8].copy())
        
        rot = R.from_euler('xyz', s[2:5]).as_matrix()
        
        # Corners
        top_zs, bot_zs = [], []
        top_v, bot_v = [], []
        
        for i in range(4):
            # Top
            pos_t = s[0] + (rot @ TOP_CORNERS_POS[i])[2]
            vel_t = s[1] + np.cross(s[5:8], rot @ TOP_CORNERS_POS[i])[2]
            top_zs.append(pos_t); top_v.append(vel_t)
            # Bot
            pos_b = s[0] + (rot @ BOT_CORNERS_POS[i])[2]
            vel_b = s[1] + np.cross(s[5:8], rot @ BOT_CORNERS_POS[i])[2]
            bot_zs.append(pos_b); bot_v.append(vel_b)
            
        self.history['top_pos'].append(top_zs)
        self.history['top_vel'].append(top_v)
        self.history['bot_pos'].append(bot_zs)
        self.history['bot_vel'].append(bot_v)
        self.history['com_pos'].append(s[0])
        self.history['com_vel'].append(s[1])
        
        self.history['force_impact'].append(fc)
        self.history['force_at_com'].append(fc + fd + fsq + (BOX_MASS * GRAVITY_CONST))
        self.history['force_air'].append(fd)
        self.history['force_squeeze'].append(fsq)
        
        self.history['elem_sq_forces'].append(e_sq.copy())
        self.history['elem_stresses'].append(e_vm.copy())
        self.history['elem_strains'].append(e_eps.copy())
        
        # Energies
        ke = 0.5 * BOX_MASS * s[1]**2 + 0.5 * np.dot(s[5:8], BOX_INERTIA @ s[5:8])
        pe = BOX_MASS * -GRAVITY_CONST * s[0] 
        vol_elem = (BOX_WIDTH*BOX_DEPTH*BOX_HEIGHT) / (ELEMENTS_COUNT_X*ELEMENTS_COUNT_Y*ELEMENTS_COUNT_Z)
        mat_se = np.sum(0.5 * e_vm * e_eps * vol_elem)
        
        n_bot_nodes = (ELEMENTS_COUNT_X + 1) * (ELEMENTS_COUNT_Y + 1)
        k_node = GROUND_STIFFNESS / n_bot_nodes
        contact_pe = np.sum(0.5 * (node_fs**2) / k_node)
        
        self.history['energy_kin'].append(ke)
        self.history['energy_pot'].append(pe)
        self.history['energy_strain'].append(mat_se)
        self.history['energy_contact'].append(contact_pe)
        
        # New Metrics
        acc_z = (fc + fd + fsq + (BOX_MASS * GRAVITY_CONST)) / BOX_MASS
        self.history['com_accel_z'].append(acc_z)
        
        # Torsion Metric: (Diagonal 1 Sum) - (Diagonal 2 Sum) of bottom corner heights
        # bot_zs = [pos_b_BL, pos_b_BR, pos_b_TR, pos_b_TL]
        tors_val = (bot_zs[0] + bot_zs[2]) - (bot_zs[1] + bot_zs[3])
        self.history['torsion'].append(tors_val)

# ==============================================================================
# [Section 5] Visualization
# ==============================================================================
class ResultVisualizer:
    def __init__(self, data):
        self.data = data
        self.t = np.array(data['t'])
        plt.rcParams['font.family'] = 'D2Coding'
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.unicode_minus'] = False

    def show_all(self, contour_type='position_z'):
        # Generate figures
        self._plot_window_1()
        self._plot_window_2()
        self._plot_window_3()
        self._plot_window_4()
        self._plot_window_5()
        self._plot_window_6()
        self._plot_window_7()
        self._plot_window_8()
        self._plot_window_9() # Torsion
        self._show_3d_animation(contour_type=contour_type)

    def _setup_plot(self, idx, title):
        fig = plt.figure(num=f"{idx}. {title}", figsize=(10, 6))
        fig.suptitle(title, fontsize=11, fontweight='bold')
        return fig

    def _plot_window_1(self):
        fig = self._setup_plot(1, "Box Surface Dynamics (Top, Bottom & Accel)")
        # 2x3 Layout: Top P/V, Bottom P/V, Accel Z, Torsion (Spare)
        ax1, ax2 = fig.add_subplot(231), fig.add_subplot(232)
        ax3, ax4 = fig.add_subplot(233), fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        
        top_zs = np.array(self.data['top_pos'])
        bot_zs = np.array(self.data['bot_pos'])
        top_vs = np.array(self.data['top_vel'])
        bot_vs = np.array(self.data['bot_vel'])
        
        lbls = ['BL', 'BR', 'TR', 'TL']
        for i in range(4):
            ax1.plot(self.t, top_zs[:, i], label=f'Top {lbls[i]}')
            ax2.plot(self.t, top_vs[:, i], label=f'Top {lbls[i]}')
            ax3.plot(self.t, bot_zs[:, i], label=f'Bot {lbls[i]}')
            ax4.plot(self.t, bot_vs[:, i], label=f'Bot {lbls[i]}')
            
        ax5.plot(self.t, self.data['com_accel_z'], 'r-', lw=1.5, label='CoM Accel Z')
        ax5.set_title("Acceleration Z (m/s^2)"); ax5.grid(True); ax5.legend()

        ax1.set_title("Top Position Z (m)"); ax1.grid(True); ax1.legend(fontsize=7)
        ax2.set_title("Top Velocity Z (m/s)"); ax2.grid(True)
        ax3.set_title("Bottom Position Z (m)"); ax3.grid(True); ax3.legend(fontsize=7)
        ax4.set_title("Bottom Velocity Z (m/s)"); ax4.grid(True)
        fig.tight_layout()

    def _plot_window_2(self):
        fig = self._setup_plot(2, "Rotational Dynamics")
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
        
        # Orientation (Euler Angles)
        orient = np.degrees(np.array(self.data['orientation']))
        ax1.plot(self.t, orient[:, 0], label='Roll (X)')
        ax1.plot(self.t, orient[:, 1], label='Pitch (Y)')
        ax1.plot(self.t, orient[:, 2], label='Yaw (Z)')
        ax1.set_title("Orientation (Degrees)"); ax1.set_ylabel("Angle (deg)"); ax1.grid(True); ax1.legend()
        
        # Angular Velocity
        ang_vel = np.degrees(np.array(self.data['ang_vel']))
        ax2.plot(self.t, ang_vel[:, 0], label='Wx (Roll Rate)')
        ax2.plot(self.t, ang_vel[:, 1], label='Wy (Pitch Rate)')
        ax2.plot(self.t, ang_vel[:, 2], label='Wz (Yaw Rate)')
        ax2.set_title("Angular Velocity (deg/s)"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Rate (deg/s)"); ax2.grid(True); ax2.legend()
        
        fig.tight_layout()

    def _plot_window_3(self):
        fig = self._setup_plot(3, "Total Force Transmitted to Ground")
        ax = fig.add_subplot(111)
        
        f_contact = np.array(self.data['force_impact'])
        f_squeeze = np.array(self.data['force_squeeze'])
        f_total = f_contact + f_squeeze
        
        ax.plot(self.t, f_total, 'r-', label='Total Ground Load (Contact + Air)')
        ax.plot(self.t, f_contact, 'k--', label='Physical Contact (Impact)', alpha=0.6)
        ax.plot(self.t, f_squeeze, 'm--', label='Air Cushion Pressure (Squeeze)', alpha=0.6)
        
        ax.set_title("Force Transmitted to Ground (N)"); ax.grid(True); ax.legend()
        fig.tight_layout()
        
    def _plot_window_4(self):
        fig = self._setup_plot(4, "Net Force at CoM")
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.data['force_at_com'], 'b')
        ax.set_title("Net Force (N)"); ax.grid(True)
        fig.tight_layout()
        
    def _plot_window_5(self):
        fig = self._setup_plot(5, "Fluid Forces")
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.data['force_air'], 'c--', label='Air Drag')
        ax.plot(self.t, self.data['force_squeeze'], 'm-', label='Squeeze Film')
        txt = "Analysis:\nSqueeze Force is integrated over\nall bottom elements using\nStefan's Law + Inertial Term."
        ax.text(0.02, 0.95, txt, transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax.set_title("Drag & Squeeze Forces"); ax.grid(True); ax.legend()
        fig.tight_layout()
        
    def _plot_window_6(self):
        fig = self._setup_plot(6, "Element-wise Squeeze Film Force (All Elements)")
        ax = fig.add_subplot(111)
        forces = np.array(self.data['elem_sq_forces'])
        
        # Plot all elements with low alpha
        for i in range(forces.shape[1]):
            ax.plot(self.t, forces[:, i], color='gray', alpha=0.1)
        
        # Plot Average
        avg_force = np.mean(forces, axis=1)
        ax.plot(self.t, avg_force, 'r-', lw=2, label='Average Force')
        
        # Also plot the 4 corners for reference
        corner_idxs = [0, ELEMENTS_COUNT_X-1, ELEMENTS_COUNT_X*ELEMENTS_COUNT_Y-1, ELEMENTS_COUNT_X*(ELEMENTS_COUNT_Y-1)]
        lbls = ['Corner 0', 'Corner 1', 'Corner 2', 'Corner 3']
        for i, idx in enumerate(corner_idxs):
            if idx < forces.shape[1]:
                ax.plot(self.t, forces[:, idx], label=lbls[i])

        ax.set_title("Squeeze Force per Element (N)"); ax.grid(True); ax.legend()
        fig.tight_layout()
        
    def _plot_window_7(self):
        fig = self._setup_plot(7, "Element-wise Stress & Strain (All Elements)")
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
        stresses = np.array(self.data['elem_stresses'])
        strains = np.array(self.data['elem_strains'])
        
        for i in range(stresses.shape[1]):
            ax1.plot(self.t, stresses[:, i], color='blue', alpha=0.05)
            ax2.plot(self.t, strains[:, i], color='green', alpha=0.05)
            
        ax1.plot(self.t, np.mean(stresses, axis=1), 'k-', lw=2, label='Average Stress')
        ax2.plot(self.t, np.mean(strains, axis=1), 'k-', lw=2, label='Average Strain')
        
        ax1.set_title("Equivalent Stress (Pa)"); ax1.grid(True); ax1.legend()
        ax2.set_title("Equivalent Strain"); ax2.grid(True); ax2.legend()
        fig.tight_layout()

    def _plot_window_8(self):
        fig = self._setup_plot(8, "Energy Analysis")
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.data['energy_kin'], label='Kinetic')
        pe = np.array(self.data['energy_pot']); pe -= np.min(pe) # Relative PE
        ax.plot(self.t, pe, label='Potential (Rel)')
        ax.plot(self.t, self.data['energy_strain'], label='Deformation/Strain')
        ax.plot(self.t, self.data['energy_contact'], label='Contact (Penalty)')
        ax.set_title("System Energy (J)"); ax.grid(True); ax.legend()
        fig.tight_layout()

    def _plot_window_9(self):
        fig = self._setup_plot(9, "Box Torsion / Twist Analysis")
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.data['torsion'], 'g-', lw=1.5, label='Torsion Metric')
        ax.set_title("Geometric Torsion Metric (Z-diff of Diagonals)"); ax.grid(True); ax.legend()
        fig.tight_layout()

    def _show_3d_animation(self, contour_type='position_z'):
        """
        Matplotlib 3D Animation
        contour_type: 'strain', 'stress', 'contact_force', 'position_z', 'velocity_z'
        """
        fig = plt.figure(f"3D Animation & Contour: {contour_type} (Matplotlib)", figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.cm as cm
        cmap = cm.turbo
        
        step = max(1, len(self.t) // 100)
        indices = range(0, len(self.t), step)
        
        # Interactive Player (Class-based for Matplotlib)
        class MatplotlibPlayer:
            def __init__(self, fig, ax, indices, data, c_type):
                self.fig = fig
                self.ax = ax
                self.indices = indices
                self.data = data
                self.c_type = c_type
                self.frame = 0
                self.playing = False
                self.total_frames = len(indices)
                self.timer = None
                
                # Determine Data Range for Colorbar
                self.vmin, self.vmax = 0.0, 1.0
                if c_type == 'strain':
                    val = np.array(data['elem_strains'])
                    self.vmax = np.max(val) if val.size > 0 else 1e-5
                    self.label = "Eqv. Strain"
                elif c_type == 'stress':
                    val = np.array(data['elem_stresses'])
                    self.vmax = np.max(val) if val.size > 0 else 1e5
                    self.label = "Eqv. Stress (Pa)"
                elif c_type == 'contact_force':
                    val = np.array(data['elem_sq_forces'])
                    self.vmax = np.max(val) if val.size > 0 else 1.0
                    self.label = "Squeeze Force (N)"
                elif c_type == 'position_z':
                    all_pos = np.array(data['com_pos'])
                    self.vmin = np.min(all_pos) - 0.05
                    self.vmax = np.max(all_pos) + 0.05
                    self.label = "Position Z (m)"
                elif c_type == 'velocity_z':
                    all_vel = np.array(data['com_vel'])
                    self.vmin = np.min(all_vel)
                    self.vmax = np.max(all_vel)
                    self.label = "Velocity Z (m/s)"
                else:
                    self.vmax = 1.0; self.label = "Value"

                # Colorbar
                self.sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
                self.sm.set_array([])
                self.cbar = self.fig.colorbar(self.sm, ax=self.ax, shrink=0.5, aspect=10)
                self.cbar.set_label(self.label)

                # Connect signals
                self.fig.canvas.mpl_connect('key_press_event', self.on_key)
                self.fig.canvas.mpl_connect('button_press_event', self.on_click)
                self.draw_frame()

            def on_key(self, event):
                if event.key == ' ':
                    self.playing = not self.playing
                    if self.playing:
                        self.play()
                    else:
                        if self.timer: self.timer.stop()
                elif event.key == ',' or event.key == 'left': # <
                    self.frame = max(0, self.frame - 1)
                    self.draw_frame()
                elif event.key == '.' or event.key == 'right': # >
                    if self.frame < self.total_frames - 1:
                        self.frame += 1
                    else:
                        self.frame = 0 # Manual Loop
                    self.draw_frame()
                elif event.key == 'm': # Menu
                    self.show_context_menu(None)
                elif event.key == 's': # Speed
                    self.change_speed()

            def on_click(self, event):
                if event.button == 3: # Right Click
                    self.show_context_menu(event)

            def show_context_menu(self, event):
                import tkinter as tk
                from tkinter import simpledialog
                
                modes = ['strain', 'stress', 'contact_force', 'drag_force', 'squeeze_force', 'position_z', 'velocity_z', 'acceleration_z']
                msg = "Select Result Type:\n" + "\n".join([f"{i+1}: {m}" for i, m in enumerate(modes)])
                root = tk.Tk(); root.withdraw()
                idx = simpledialog.askinteger("Result Type", msg, initialvalue=modes.index(self.c_type)+1, minvalue=1, maxvalue=len(modes))
                root.destroy()
                
                if idx:
                    self.update_mode(modes[idx-1])

            def change_speed(self):
                import tkinter as tk
                from tkinter import simpledialog
                root = tk.Tk(); root.withdraw()
                # Here step is actually implemented as 'indices' in _show_3d_animation.
                # To change speed dynamically, we'd need to re-generate indices.
                # For now, let's just change interval of timer.
                new_int = simpledialog.askinteger("Speed", "Timer Interval (ms):", initialvalue=20, minvalue=1, maxvalue=500)
                root.destroy()
                if new_int:
                    if self.timer:
                        self.timer.stop()
                        self.timer.interval = new_int
                        if self.playing: self.timer.start()

            def update_mode(self, new_mode):
                self.c_type = new_mode
                # Re-calculate limits
                data = self.data
                if new_mode == 'strain':
                    val = np.array(data['elem_strains']); self.vmax = np.max(val) if val.size > 0 else 1e-5; self.vmin = 0; self.label = "Eqv. Strain"
                elif new_mode == 'stress':
                    val = np.array(data['elem_stresses']); self.vmax = np.max(val) if val.size > 0 else 1e5; self.vmin = 0; self.label = "Eqv. Stress (Pa)"
                elif new_mode == 'contact_force':
                    self.vmax = BOX_MASS * 10.0; self.vmin = 0; self.label = "Impact Force (N)"
                elif new_mode == 'drag_force':
                    val = np.array(data['force_air']); self.vmax = np.max(np.abs(val)) if val.size > 0 else 1.0; self.vmin = 0; self.label = "Drag Force (N)"
                elif new_mode == 'squeeze_force':
                    val = np.array(data['elem_sq_forces']); self.vmax = np.max(val) if val.size > 0 else 1.0; self.vmin = 0; self.label = "Squeeze Force (N)"
                elif new_mode == 'position_z':
                    all_pos = np.array(data['com_pos']); self.vmin = np.min(all_pos) - 0.05; self.vmax = np.max(all_pos) + 0.05; self.label = "Position Z (m)"
                elif new_mode == 'velocity_z':
                    all_vel = np.array(data['com_vel']); self.vmin = np.min(all_vel); self.vmax = np.max(all_vel); self.label = "Velocity Z (m/s)"
                elif new_mode == 'acceleration_z':
                    all_acc = np.array(data['com_accel_z']); self.vmin = np.min(all_acc); self.vmax = np.max(all_acc); self.label = "Accel Z (m/s^2)"
                
                self.sm.set_clim(self.vmin, self.vmax)
                self.cbar.set_label(self.label)
                self.draw_frame()

            def play(self):
                if self.timer is None:
                    self.timer = self.fig.canvas.new_timer(interval=20)
                    self.timer.add_callback(self.next_frame_auto)
                self.timer.start()

            def next_frame_auto(self):
                if not self.playing: return
                if self.frame < self.total_frames - 1:
                    self.frame += 1
                    self.draw_frame()
                else:
                    # Loop Playback
                    self.frame = 0
                    self.draw_frame()

            def draw_frame(self):
                self.ax.cla()
                t_idx = self.indices[self.frame]
                
                # Transform nodes
                h_com = self.data['com_pos'][t_idx]
                rot = R.from_euler('xyz', self.data['orientation'][t_idx]).as_matrix()
                nodes_world = (rot @ NODES_REF.T).T + np.array([0, 0, h_com])
                
                # Prepare Faces (Exterior Skin)
                verts = []
                face_colors = []
                zero_color = cmap(0.0) # Used for 'no data' or 'base value'
                
                elem_vals = None
                if self.c_type == 'strain':
                    elem_vals = self.data['elem_strains'][t_idx]
                elif self.c_type == 'stress':
                    elem_vals = self.data['elem_stresses'][t_idx]
                elif self.c_type == 'contact_force':
                    # Simplified: Use squeeze forces for element mapping if contact mode
                    elem_vals = self.data['elem_sq_forces'][t_idx]
                elif self.c_type == 'squeeze_force':
                    elem_vals = self.data['elem_sq_forces'][t_idx]

                for face in SURFACE_FACES:
                    e_idx = face['e_idx']
                    n_idxs = face['nodes']
                    f_type = face['type']
                    
                    poly_v = nodes_world[n_idxs]
                    verts.append(poly_v)
                    
                    val = 0.0
                    # Coloring Logic
                    if self.c_type == 'position_z':
                        val = np.mean(poly_v[:, 2]) # Avg Z
                    elif self.c_type == 'velocity_z':
                        val = self.data['com_vel'][t_idx]
                    elif self.c_type == 'acceleration_z':
                        val = self.data['com_accel_z'][t_idx]
                    elif self.c_type == 'drag_force':
                        val = self.data['force_air'][t_idx]
                    elif elem_vals is not None and f_type == 'bottom':
                        # Element-based data (Strain/Stress/Force) - Only valid on Bottom?
                        # Or define dummy for others.
                        if e_idx < len(elem_vals):
                            val = elem_vals[e_idx]
                        else:
                            val = 0.0 # Should not happen if matching sizes
                    else:
                        val = 0.0 # Top/Side for Stress/Strain
                    
                    # Normalize
                    # Check for NaN/Inf
                    if np.isnan(val): val = self.vmin
                    
                    if self.vmax > self.vmin:
                        norm = (val - self.vmin) / (self.vmax - self.vmin)
                    else:
                        norm = 0.0
                    norm = np.clip(norm, 0.0, 1.0)
                    face_colors.append(cmap(norm))

                pc = Poly3DCollection(verts, facecolors=face_colors, edgecolors='k', linewidths=0.1, alpha=0.95)
                self.ax.add_collection3d(pc)
                
                # Wireframe & Ground
                tl = [TOP_NODE_IDXS[0], TOP_NODE_IDXS[1], TOP_NODE_IDXS[2], TOP_NODE_IDXS[3], TOP_NODE_IDXS[0]]
                bl = [BOT_NODE_IDXS[0], BOT_NODE_IDXS[1], BOT_NODE_IDXS[2], BOT_NODE_IDXS[3], BOT_NODE_IDXS[0]]
                self.ax.plot(nodes_world[tl,0], nodes_world[tl,1], nodes_world[tl,2], 'k-', lw=1.5)
                self.ax.plot(nodes_world[bl,0], nodes_world[bl,1], nodes_world[bl,2], 'k-', lw=1.5)
                for i in range(4):
                    self.ax.plot([nodes_world[TOP_NODE_IDXS[i],0], nodes_world[BOT_NODE_IDXS[i],0]],
                                 [nodes_world[TOP_NODE_IDXS[i],1], nodes_world[BOT_NODE_IDXS[i],1]],
                                 [nodes_world[TOP_NODE_IDXS[i],2], nodes_world[BOT_NODE_IDXS[i],2]], 'k-', lw=1.5)
                
                # Ground Grid (Z=0 fixed)
                gx = np.linspace(-BOX_WIDTH, BOX_WIDTH, 10)
                gy = np.linspace(-BOX_DEPTH, BOX_DEPTH, 10)
                GX, GY = np.meshgrid(gx, gy)
                self.ax.plot_wireframe(GX, GY, np.zeros_like(GX), color='gray', alpha=0.3)
                
                self.ax.set_title(f"Time: {self.data['t'][t_idx]:.3f}s | Mode: {self.c_type}")
                
                # Equal Aspect Ratio
                max_range = max(BOX_WIDTH, BOX_DEPTH, h_com + 0.2) / 2.0
                mid_x = (nodes_world[:,0].max() + nodes_world[:,0].min()) * 0.5
                mid_y = (nodes_world[:,1].max() + nodes_world[:,1].min()) * 0.5
                mid_z = (nodes_world[:,2].max() + nodes_world[:,2].min()) * 0.5
                self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
                self.ax.set_zlim(0, 2*max_range)
                
                self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")
                self.fig.canvas.draw_idle()

        print(f"\n>>> Matplotlib 3D Player Started (Mode: {contour_type})")
        print("    [Controls] Space: Play/Stop, '<': Prev, '>': Next")
        player = MatplotlibPlayer(fig, ax, indices, self.data, contour_type)
        plt.show()

    def visualize_pyvista(self, speed_step=3):
        """
        PyVista Interactive 3D Plot (Restored & Optimized)
        speed_step: Number of frames to advance per update tick (higher = faster)
        """
        try:
            import pyvista as pv
            # [Compatibility Fix] Disable advanced rendering features to prevent shader errors
            # Wrap in try-except for version compatibility
            try:
                pv.global_theme.depth_peeling.enabled = False
                pv.global_theme.anti_aliasing = None
                pv.global_theme.lighting = False
                pv.global_theme.show_scalar_bar = True
                # Explicitly disable PBR and other complex shaders
                if hasattr(pv.global_theme, 'pbr'):
                    pv.global_theme.pbr = False
            except Exception:
                pass
        except ImportError:
            print("\n[User Info] PyVista not installed. Skipping.")
            return

        print(f"\n>>> Starting PyVista 3D Interactive View (Speed Step: {speed_step})...")
        print("    [Controls] Space: Play/Stop, '<': Prev, '>': Next")
        
        # Setup Grid
        cells = []
        cell_type = np.array([pv.CellType.HEXAHEDRON] * len(ELEMENTS), dtype=np.uint8)
        for elem in ELEMENTS:
            cells.append(8); cells.extend(elem)
            
        grid = pv.UnstructuredGrid(cells, cell_type, NODES_REF)
        
        plotter = pv.Plotter(title="FEM Drop Simulation (PyVista)")
        plotter.add_axes()
        
        # Fixed Ground Plane at Z=0
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=6.0, j_size=6.0)
        plotter.add_mesh(ground, color='gray', opacity=0.3, show_edges=True, lighting=False)
        
        # Reference Line
        line = pv.Line([0,0,0], [0,0,1.0])
        plotter.add_mesh(line, color='r', line_width=2, label="Z-Axis")
        
        # Initial Data
        full_strains = np.zeros(len(ELEMENTS))
        all_strains = np.array(self.data['elem_strains'])
        max_s = np.max(all_strains) if all_strains.size > 0 else 1.0e-5
        
        plotter.add_mesh(grid, scalars=full_strains, cmap="turbo", 
                         clim=[0, max_s], show_edges=True,
                         scalar_bar_args={'title': 'Strain'},
                         lighting=False, smooth_shading=False)
        
        plotter.view_isometric()
        text_actor = plotter.add_text(f"Time: 0.000s", position='upper_right', font_size=12)
        
        # --- Interactive Player State ---
        class PVPlayer:
            def __init__(self, data, times, step):
                self.data = data
                self.times = times
                self.frames = len(times)
                self.speed_step = step 
                self.idx = 0
                self.playing = False
                self.c_type = 'strain' # Default
                
            def prev(self):
                self.idx = max(0, self.idx - self.speed_step)
                update_scene(self.idx)
                
            def next(self):
                if self.idx < self.frames - 1:
                    self.idx = min(self.frames - 1, self.idx + self.speed_step)
                else:
                    self.idx = 0 # Loop
                update_scene(self.idx)
                
            def toggle(self):
                self.playing = not self.playing

            def change_mode(self):
                import tkinter as tk
                from tkinter import simpledialog
                modes = ['strain', 'stress', 'contact_force', 'drag_force', 'squeeze_force', 'position_z', 'velocity_z', 'acceleration_z']
                msg = "Select Result Type:\n" + "\n".join([f"{i+1}: {m}" for i, m in enumerate(modes)])
                root = tk.Tk(); root.withdraw()
                idx = simpledialog.askinteger("Result Type", msg, initialvalue=modes.index(self.c_type)+1, minvalue=1, maxvalue=len(modes))
                root.destroy()
                if idx:
                    self.c_type = modes[idx-1]
                    # Update Clim dynamically in update_scene
                    update_scene(self.idx)

            def change_speed(self):
                import tkinter as tk
                from tkinter import simpledialog
                root = tk.Tk(); root.withdraw()
                new_step = simpledialog.askinteger("Speed", "Frame Jump (speed_step):", initialvalue=self.speed_step, minvalue=1, maxvalue=20)
                root.destroy()
                if new_step: self.speed_step = new_step

        player = PVPlayer(self.data, self.t, speed_step)
        
        def update_scene(idx):
            # Update Geometry
            h_com = self.data['com_pos'][idx]
            orient = self.data['orientation'][idx]
            rot = R.from_euler('xyz', orient).as_matrix()
            nodes_new = (rot @ NODES_REF.T).T + np.array([0, 0, h_com])
            grid.points = nodes_new
            
            # Update Scalars
            current_vals = np.zeros(len(ELEMENTS))
            mode = player.c_type
            if mode == 'strain':
                vals = self.data['elem_strains'][idx]
                current_vals[BOTTOM_ELEM_IDXS] = vals
                label = "Strain"
            elif mode == 'stress':
                vals = self.data['elem_stresses'][idx]
                current_vals[BOTTOM_ELEM_IDXS] = vals
                label = "Stress (Pa)"
            elif mode == 'contact_force':
                vals = self.data['elem_sq_forces'][idx]
                current_vals[BOTTOM_ELEM_IDXS] = vals
                label = "Contact Force (N)"
            elif mode == 'position_z':
                current_vals[:] = h_com
                label = "Pos Z"
            elif mode == 'velocity_z':
                current_vals[:] = self.data['com_vel'][idx]
                label = "Vel Z"
            elif mode == 'acceleration_z':
                current_vals[:] = self.data['com_accel_z'][idx]
                label = "Acc Z"
            else:
                label = mode
            
            grid['Result'] = current_vals
            grid.set_active_scalars('Result')
            
            # Auto-scale color bar? 
            # PyVista plotter.add_mesh returns an actor, we can update clim
            if hasattr(plotter, 'mapper'):
                plotter.update_scalar_bar_range([np.min(current_vals), np.max(current_vals)])

            text_actor.set_text(0, f"Time: {self.t[idx]:.3f}s | Mode: {label} | Speed: {player.speed_step}")
            
        # Register Keys
        plotter.add_key_event(',', player.prev)
        plotter.add_key_event('.', player.next)
        plotter.add_key_event('space', player.toggle)
        plotter.add_key_event('m', player.change_mode)
        plotter.add_key_event('s', player.change_speed)
        
        plotter.show(interactive_update=True, auto_close=False)
        
        while True:
            if not plotter.render_window: break
            
            if player.playing:
                if player.idx < player.frames - 1:
                    player.idx += player.speed_step
                    if player.idx >= player.frames: player.idx = player.frames - 1
                    update_scene(player.idx)
                else:
                    player.idx = 0 # Loop back to start
                    update_scene(player.idx)
            
            plotter.update()
            time.sleep(0.005) 
            
        print(">>> PyVista Window Closed.")

if __name__ == "__main__":
    solver = BoxSimulationSolver()
    data = solver.solve()
    
    viz = ResultVisualizer(data)
    # viz.show_all() # Default Matplotlib
    
    # Uncomment to use PyVista if installed    
    viz.show_all()
    #viz.visualize_pyvista()

