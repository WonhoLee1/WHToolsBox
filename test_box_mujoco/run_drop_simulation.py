import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import time
from run_discrete_builder import create_model, get_default_config

plt.rcParams.update({'font.size': 8})


def get_body_kinematics(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return None
    pos = data.xpos[body_id].copy()
    mat = data.xmat[body_id].reshape(3, 3).copy()
    vel = data.cvel[body_id].copy() # [w_x, w_y, w_z, v_x, v_y, v_z]
    acc = data.cacc[body_id].copy() # [alpha_x, alpha_y, alpha_z, a_x, a_y, a_z]
    return pos, mat, vel, acc

def compute_corners(center_pos, center_mat, box_w, box_h, box_d):
    """지정된 중심점과 회전 행렬을 기반으로 8개 모서리 좌표 역계산"""
    corners_local = []
    for x in [-box_w/2, box_w/2]:
        for y in [-box_h/2, box_h/2]:
            for z in [-box_d/2, box_d/2]:
                corners_local.append(np.array([x, y, z]))
    
    corners_global = []
    for loc in corners_local:
        glob = center_pos + center_mat @ loc
        corners_global.append(glob)
    return np.array(corners_global)

def compute_corner_kinematics(center_pos, center_mat, center_vel, center_acc, box_w, box_h, box_d):
    # center_vel: [wx, wy, wz, vx, vy, vz]
    w = center_vel[0:3]
    v = center_vel[3:6]
    
    # center_acc: [alphax, alphay, alphaz, ax, ay, az]
    alpha = center_acc[0:3]
    a = center_acc[3:6]
    
    corners_local = []
    for x in [-box_w/2, box_w/2]:
        for y in [-box_h/2, box_h/2]:
            for z in [-box_d/2, box_d/2]:
                corners_local.append(np.array([x, y, z]))
                
    results = []
    for loc in corners_local:
        # global offset vector from center
        r = center_mat @ loc
        
        # velocity = v + w x r
        v_corner = v + np.cross(w, r)
        
        # acceleration = a + alpha x r + w x (w x r)
        a_corner = a + np.cross(alpha, r) + np.cross(w, np.cross(w, r))
        
        results.append({
            'pos': center_pos + r,
            'vel': v_corner,
            'acc': a_corner
        })
    return results

def run_simulation(config_or_path, sim_duration=0.5):
    if isinstance(config_or_path, str):
        # It's an XML path
        xml_path = config_or_path
        print(f"Loading MuJoCo model from XML: {xml_path}")
        model = mujoco.MjModel.from_xml_path(xml_path)
        config = get_default_config() # fallback for later config.get calls
    else:
        # It's a config dictionary
        config = config_or_path
        print("Generating discrete box model from config...")
        xml_str = create_model("temp_drop_sim.xml", config=config)
        model = mujoco.MjModel.from_xml_string(xml_str)
        # Override sim_duration if provided in config
        sim_duration = config.get('sim_duration', sim_duration)
    data = mujoco.MjData(model)
    
    # Identify bodies for structural metrics
    # Group by component name -> dict of (i,j,k) -> body_id
    components = {}
    nominal_local_pos = {}
    
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("b_"):
            parts = name.split("_")
            if len(parts) >= 5:
                comp_name = parts[1]
                idx_i, idx_j, idx_k = int(parts[2]), int(parts[3]), int(parts[4])
                
                if comp_name not in components:
                    components[comp_name] = {}
                components[comp_name][(idx_i, idx_j, idx_k)] = i
                nominal_local_pos[i] = model.body_pos[i].copy()

    # Time setup
    dt = model.opt.timestep
    duration = config.get("sim_duration", 1.0)
    steps = int(duration / dt)
    
    time_history = []
    z_hist = []
    pos_hist = []
    vel_hist = []
    acc_hist = []
    corner_pos_hist = []
    corner_vel_hist = []
    corner_acc_hist = []
    floor_impact_hist = []
    air_drag_hist = []
    air_viscous_hist = []
    air_squeeze_hist = []
    
    # Air resistance parameters from config
    rho      = config.get('air_density', 1.225)
    mu       = config.get('air_viscosity', 1.81e-5)
    Cd_blunt = config.get('air_cd_drag', 1.05)
    Cd_visc  = config.get('air_cd_viscous', 0.0)
    Csq      = config.get('air_coef_squeeze', 1.0)
    h_sq_max = config.get('air_squeeze_hmax', 0.20)
    h_sq_min = config.get('air_squeeze_hmin', 0.001)
    enable_air = config.get('enable_air_resistance', True)
    
    # Box face areas for drag & viscous
    W_box = config.get('box_w', 2.0)
    H_box = config.get('box_h', 1.4)
    D_box = config.get('box_d', 0.25)
    A_front = W_box * H_box   # ZY face (facing fall direction)
    A_side  = H_box * D_box   # XZ face
    A_top   = W_box * D_box   # XY face
    
    # Squeeze geometric factor: (L*W / 2*(L+W))^2
    L_sq, W_sq = W_box, H_box
    geo_factor_sq = ((L_sq * W_sq) / (2 * (L_sq + W_sq))) ** 2
    PHYSICS_COEF_SQ = 0.5 * rho * geo_factor_sq * Csq
    
    # Metric histories per component by row (j) and individual blocks
    metrics = {}
    for comp in components:
        metrics[comp] = {}
        metrics[comp]['all_blocks_angle'] = {}
        metrics[comp]['block_nominals'] = {}
        for block_idx in components[comp]:
            metrics[comp]['all_blocks_angle'][block_idx] = []
            metrics[comp]['block_nominals'][block_idx] = nominal_local_pos[components[comp][block_idx]]
            
        # Find unique j indices
        j_idx = set([k[1] for k in components[comp].keys()])
        for j in j_idx:
            metrics[comp][j] = {
                'bending': [], 'twist': [], 'energy': [],
                'loc_b': [], 'loc_t': [], 'loc_e': []
            }
            
    print(f"Starting simulation for {duration} seconds with {steps} steps...")
    
    max_g_force = 0.0
    k_spring_proxy = 1e4 # N/m proxy for strain energy calculation
    
    prev_vel_z = 0.0
    
    mujoco.mj_forward(model, data)
    
    for step in range(steps):
        try:
            mujoco.mj_step(model, data)
        except Exception as e:
            import sys
            sys.stderr.write(f"\n[Drop Sim Error] Simulation unstable or crashed (mujoco.mj_step): {e}\n")
            break
            
        time_history.append(data.time)
        
        # 1. Root Kinematics (BPackagingBox itself is just a massless container)
        pos, mat, vel, acc = get_body_kinematics(model, data, "BPackagingBox")
        z_hist.append(pos[2])
        pos_hist.append(pos)
        vel_hist.append(vel)
        acc_hist.append(acc)
        
        corners = compute_corner_kinematics(pos, mat, vel, acc, config["box_w"], config["box_h"], config["box_d"])
        corner_pos_hist.append([c['pos'] for c in corners])
        corner_vel_hist.append([c['vel'] for c in corners])
        corner_acc_hist.append([c['acc'] for c in corners])
        
        # Floor Impact (Sum of normal forces from worldbody contacts)
        floor_f = 0.0
        for i_con in range(data.ncon):
            contact = data.contact[i_con]
            body1 = model.geom_bodyid[contact.geom1]
            body2 = model.geom_bodyid[contact.geom2]
            if body1 == 0 or body2 == 0:
                forces = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, i_con, forces)
                floor_f += abs(forces[0])
        floor_impact_hist.append(floor_f)
        
        # Air Resistance: Drag, Viscous, Squeeze
        f_drag = 0.0
        f_viscous = 0.0
        f_squeeze = 0.0
        if enable_air:
            lin_vel_w = vel[3:6]  # linear velocity [vx, vy, vz] world frame
            v_z = lin_vel_w[2]    # vertical velocity (downward = negative)
            v_mag = np.linalg.norm(lin_vel_w)
            
            # Drag: F_d = 0.5 * rho * Cd * A * v_z^2  (resists motion)
            # Use frontal area relative to velocity direction
            v_vec_n = lin_vel_w / (v_mag + 1e-9)
            A_eff = abs(v_vec_n[0]) * A_side + abs(v_vec_n[1]) * A_side + abs(v_vec_n[2]) * A_front
            f_drag = 0.5 * rho * Cd_blunt * A_eff * v_mag**2
            
            # Viscous: F_v = mu * Cd_visc * A * v  (linear in v for low Re)
            A_total_surface = 2 * (W_box*H_box + H_box*D_box + W_box*D_box)
            f_viscous = mu * Cd_visc * A_total_surface * v_mag
            
            # Squeeze Film: grid integration over bottom face when near floor
            h_body = pos[2]
            if 0 < h_body < h_sq_max and v_z < 0:
                N_sq = 8
                dA_sq = (L_sq * W_sq) / (N_sq * N_sq)
                grid_steps = np.linspace(-0.5 + 0.5/N_sq, 0.5 - 0.5/N_sq, N_sq)
                for uu in grid_steps:
                    for vv in grid_steps:
                        h_pt = max(h_body, h_sq_min)
                        dF = PHYSICS_COEF_SQ * dA_sq * (abs(v_z) / h_pt)**2
                        dF = min(dF, 500.0)
                        f_squeeze += dF
            
            # Apply squeeze as upward external force on root body
            try:
                root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BPackagingBox")
                if root_id >= 0:
                    data.xfrc_applied[root_id][2] += f_squeeze
            except Exception:
                pass
        
        air_drag_hist.append(f_drag)
        air_viscous_hist.append(f_viscous)
        air_squeeze_hist.append(f_squeeze)
        
        # G-Force is now calculated purely from root kinematic differences after simulation
            
        # 2. Component Structural Metrics
        mat_inv = mat.T
        for comp, blocks in components.items():
            # Gather current local positions
            current_local = {}
            for (curr_i, curr_j, curr_k), b_id in blocks.items():
                g_pos = data.xpos[b_id]
                g_mat = data.xmat[b_id].reshape(3, 3)
                rel_pos = g_pos - pos
                l_pos = mat_inv @ rel_pos
                current_local[(curr_i, curr_j, curr_k)] = l_pos
                
                # Calculate angular deformation (angle of rotation relative to root)
                rot_rel = mat_inv @ g_mat
                tr = np.clip(np.trace(rot_rel), -1.0, 3.0)
                theta = np.arccos((tr - 1.0) / 2.0)
                block_angle_deg = np.degrees(theta)
                metrics[comp]['all_blocks_angle'][(curr_i, curr_j, curr_k)].append(block_angle_deg)
                
            # Process by row (j)
            j_indices = sorted(list(set([k[1] for k in blocks.keys()])))
            j_mid = j_indices[len(j_indices)//2] if len(j_indices) > 0 else 0
            
            # Mid row slope for twist reference
            mid_row_x = []
            mid_row_z = []
            for (curr_i, curr_j, curr_k), l_pos in current_local.items():
                if curr_j == j_mid:
                    mid_row_x.append(l_pos[0])
                    mid_row_z.append(l_pos[2])
            mid_slope = 0.0
            if len(mid_row_x) > 1:
                mid_slope = np.polyfit(mid_row_x, mid_row_z, 1)[0]
                
            for j in j_indices:
                row_x = []
                row_z = []
                energies = {}
                dz_vals = {}
                
                for (curr_i, curr_j, curr_k), l_pos in current_local.items():
                    if curr_j == j:
                        b_id = blocks[(curr_i, curr_j, curr_k)]
                        nom_pos = nominal_local_pos[b_id]
                        delta = l_pos - nom_pos
                        
                        # Energy proxy: 0.5 * k * dx^2
                        energy = 0.5 * k_spring_proxy * np.sum(delta**2)
                        energies[(curr_i, curr_k)] = energy
                        
                        row_x.append(l_pos[0])
                        row_z.append(l_pos[2])
                        dz_vals[(curr_i, curr_k)] = delta[2]
                
                # Bending: Max Z deflection angle approximation
                bending_deg = 0.0
                loc_b = "N/A"
                if len(dz_vals) > 1:
                    max_idx = max(dz_vals, key=dz_vals.get)
                    min_idx = min(dz_vals, key=dz_vals.get)
                    dz_diff = dz_vals[max_idx] - dz_vals[min_idx]
                    dx_diff = max(row_x) - min(row_x) if max(row_x) != min(row_x) else 1e-6
                    bending_deg = np.degrees(np.arctan(abs(dz_diff / dx_diff)))
                    loc_b = f"{max_idx[0]}_{j}_{max_idx[1]}"
                    
                # Twisting: relative slope diff to mid row
                twist_deg = 0.0
                loc_t = f"*_{j}_*"
                if len(row_x) > 1:
                    slope = np.polyfit(row_x, row_z, 1)[0]
                    twist_deg = np.degrees(np.arctan(abs(slope - mid_slope)))
                    
                # Energy Peak
                peak_energy = 0.0
                loc_e = "N/A"
                if energies:
                    max_e_idx = max(energies, key=energies.get)
                    peak_energy = energies[max_e_idx]
                    loc_e = f"{max_e_idx[0]}_{j}_{max_e_idx[1]}"
                    
                metrics[comp][j]['bending'].append(bending_deg)
                metrics[comp][j]['twist'].append(twist_deg)
                metrics[comp][j]['energy'].append(peak_energy)
                
                # Store locations only if it's a new max (to retrieve easily later, simplified below)
                metrics[comp][j]['loc_b'].append(loc_b)
                metrics[comp][j]['loc_t'].append(loc_t)
                metrics[comp][j]['loc_e'].append(loc_e)

    print("Simulation completed.")
    
    # Calculate global peak G over the differentiated root trajectory
    # Double differentiation of Z position for absolute shock magnitude
    z_array = np.array(z_hist)
    v_z = np.gradient(z_array, dt)
    a_z = np.gradient(v_z, dt)
    # The actual physics simulation tracks relative to earth freefall, so we pull max impact
    root_acc_history = np.abs(a_z) / 9.81
    max_g_force = float(np.max(root_acc_history))
    
    print("-" * 50)
    print(f"Peak Assembly G-Force: {max_g_force:.2f} G")
    print("-" * 50)
    
    # Generate Terminal Summary Report
    for comp, j_data in metrics.items():
        print("=" * 83)
        print(f"[최대 구조 변형 지표 로컬라이징 리포트] - Body: {comp}")
        print("-" * 83)
        print(f"{'Row Index':<9} | {'Max Bending (deg) / Loc':<23} | {'Max Twist (deg) / Loc':<21} | {'Peak Energy (J) / Loc':<21}")
        print("-" * 83)
        
        j_keys = [k for k in j_data.keys() if isinstance(k, int)]
        for j in sorted(j_keys):
            hist = j_data[j]
            max_b = max(hist['bending'])
            idx_b = hist['bending'].index(max_b)
            loc_b = hist['loc_b'][idx_b]
            
            max_t = max(hist['twist'])
            idx_t = hist['twist'].index(max_t)
            loc_t = hist['loc_t'][idx_t]
            
            max_e = max(hist['energy'])
            idx_e = hist['energy'].index(max_e)
            loc_e = hist['loc_e'][idx_e]
            
            str_b = f"{max_b:>6.2f} (@ {loc_b:<7})"
            str_t = f"{max_t:>6.2f} (@ {loc_t:<7})"
            str_e = f"{max_e:>6.2f} (@ {loc_e:<7})"
            print(f"y = {j:<5} | {str_b:<23} | {str_t:<21} | {str_e:<21}")
        print("=" * 83)
        print()
        
    # Plotting (only if requested)
    if config.get("plot_results", True):
        plt.figure(figsize=(10, 5))
        plt.plot(time_history, root_acc_history, label='Internal TV Accel (G)', color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('G-Force')
        plt.title('Internal TV (OpenCell/Chassis) Peak Impact G-Force')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('rds-impact_gforce.png')
        plt.close()
        
        # Floor Impact + Air Resistance combined plot (2 subplots)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        fig.suptitle('Floor Impact Force & Air Resistance Forces')
        ax1.plot(time_history, floor_impact_hist, label='Floor Normal Force (N)', color='red')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Floor Impact')
        ax1.grid(True)
        ax1.legend()
        ax2.plot(time_history, air_drag_hist,    label=f'Drag  (Cd={config.get("air_cd_drag",1.05):.2f})', color='blue')
        ax2.plot(time_history, air_viscous_hist, label=f'Viscous (Cd_v={config.get("air_cd_viscous",0.0):.2f})', color='green')
        ax2.plot(time_history, air_squeeze_hist, label=f'Squeeze Film (k={config.get("air_coef_squeeze",1.0):.1f})', color='orange')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force (N)')
        ax2.set_title('Air Resistance Components')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.savefig('rds-floor_impact.png')
        plt.close()
        
        # Motion All & Motion Z Setup
        pos_np = np.array(pos_hist) # (steps, 3)
        vel_np = np.array(vel_hist) # (steps, 6) -> [wx, wy, wz, vx, vy, vz]
        acc_np = np.array(acc_hist) # (steps, 6) -> [ax, ay, az, lin_ax, ...]
        
        c_pos_np = np.array(corner_pos_hist) # (steps, 8, 3)
        c_vel_np = np.array(corner_vel_hist) # (steps, 8, 3)
        c_acc_np = np.array(corner_acc_hist) # (steps, 8, 3)
        
        curves = {
            'COG': {'pos': pos_np, 'vel': vel_np[:, 3:6], 'acc': acc_np[:, 3:6]},
            'CORNER_L-B-B': {'pos': c_pos_np[:,0,:], 'vel': c_vel_np[:,0,:], 'acc': c_acc_np[:,0,:]},
            'CORNER_L-B-F': {'pos': c_pos_np[:,1,:], 'vel': c_vel_np[:,1,:], 'acc': c_acc_np[:,1,:]},
            'CORNER_L-T-B': {'pos': c_pos_np[:,2,:], 'vel': c_vel_np[:,2,:], 'acc': c_acc_np[:,2,:]},
            'CORNER_L-T-F': {'pos': c_pos_np[:,3,:], 'vel': c_vel_np[:,3,:], 'acc': c_acc_np[:,3,:]},
            'CORNER_R-B-B': {'pos': c_pos_np[:,4,:], 'vel': c_vel_np[:,4,:], 'acc': c_acc_np[:,4,:]},
            'CORNER_R-B-F': {'pos': c_pos_np[:,5,:], 'vel': c_vel_np[:,5,:], 'acc': c_acc_np[:,5,:]},
            'CORNER_R-T-B': {'pos': c_pos_np[:,6,:], 'vel': c_vel_np[:,6,:], 'acc': c_acc_np[:,6,:]},
            'CORNER_R-T-F': {'pos': c_pos_np[:,7,:], 'vel': c_vel_np[:,7,:], 'acc': c_acc_np[:,7,:]}
        }
        ordered_keys = ['COG', 'CORNER_L-T-F', 'CORNER_R-T-F', 'CORNER_L-T-B', 'CORNER_R-T-B', 'CORNER_L-B-F', 'CORNER_R-B-F', 'CORNER_L-B-B', 'CORNER_R-B-B']
        
        # rds-Motion_All.png
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('Kinematics: Position, Velocity, Acceleration vs Time', fontsize=16)
        labels_row = ['Position (m)', 'Velocity (m/s)', 'Acceleration (m/s^2)']
        labels_col = ['X Axis', 'Y Axis', 'Z Axis']
        for row in range(3):
            for col in range(3):
                ax = axs[row, col]
                for k in ordered_keys:
                    metric = 'pos' if row == 0 else ('vel' if row == 1 else 'acc')
                    ax.plot(time_history, curves[k][metric][:, col], label=k)
                if row == 2: ax.set_xlabel('Time (s)')
                if col == 0: ax.set_ylabel(labels_row[row])
                if row == 0: ax.set_title(labels_col[col])
                ax.grid(True)
        axs[0, 2].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig('rds-Motion_All.png')
        plt.close()
        
        # rds-Motion_Z.png
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Kinematics (Z Axis Only): Position, Velocity, Acceleration vs Time', fontsize=16)
        titles_z = ['Z Position (m)', 'Z Velocity (m/s)', 'Z Acceleration (m/s^2)']
        for i, metric in enumerate(['pos', 'vel', 'acc']):
            ax = axs[i]
            for k in ordered_keys:
                ax.plot(time_history, curves[k][metric][:, 2], label=k)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(titles_z[i])
            ax.grid(True)
        axs[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig('rds-Motion_Z.png')
        plt.close()

        # Plot metric per component max over time
        for comp, j_data in metrics.items():
            j_keys = [k for k in j_data.keys() if isinstance(k, int)]
            if len(j_keys) > 0:
                plt.figure(figsize=(10, 5))
                
                # Find maximums across all J rows dynamically
                all_b_hist = np.array([j_data[j]['bending'] for j in j_keys])
                all_t_hist = np.array([j_data[j]['twist'] for j in j_keys])
                
                max_b_time = np.max(all_b_hist, axis=0)
                max_t_time = np.max(all_t_hist, axis=0)
                
                # Find the location string of the absolute global peak to display in legend
                max_b_idx = np.argmax(max_b_time)
                row_idx_b = np.argmax(all_b_hist[:, max_b_idx])
                global_loc_b = j_data[j_keys[row_idx_b]]['loc_b'][max_b_idx]
                
                max_t_idx = np.argmax(max_t_time)
                row_idx_t = np.argmax(all_t_hist[:, max_t_idx])
                global_loc_t = j_data[j_keys[row_idx_t]]['loc_t'][max_t_idx]
                
                plt.plot(time_history, max_b_time, label=f'Max Bending (deg) [{global_loc_b}]')
                plt.plot(time_history, max_t_time, label=f'Max Twisting (deg) [{global_loc_t}]')
                plt.xlabel('Time (s)')
                plt.ylabel('Angle (deg)')
                plt.title(f'{comp} Structural Deformation (Max)')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'rds-{comp}_deformation.png')
                plt.close()
            
            # New: All Blocks Angle Plot
            all_blocks = j_data.get('all_blocks_angle', {})
            block_noms = j_data.get('block_nominals', {})
            if len(all_blocks) > 0:
                fig = plt.figure(figsize=(14, 6))
                
                # Create main plot for angle curves
                ax_main = fig.add_axes([0.05, 0.1, 0.65, 0.8])
                num_blocks = len(all_blocks)
                cols_legend = min(4, max(1, num_blocks // 10))
                
                for block_idx, a_hist in all_blocks.items():
                    legend_name = f"{block_idx[0]}-{block_idx[1]}-{block_idx[2]}"
                    ax_main.plot(time_history, a_hist, label=legend_name, linewidth=1.0)
                ax_main.set_xlabel('Time (s)')
                ax_main.set_ylabel('Def. Angle (deg)')
                ax_main.set_title(f'{comp} Block Angle Deformation (Relative to Root)')
                ax_main.grid(True)
                
                # Put Legend outside
                ax_main.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=cols_legend, fontsize=6)
                
                # Inset 3D scatter plot for index guide
                ax_inset = fig.add_axes([0.65, 0.05, 0.3, 0.3], projection='3d')
                xs, ys, zs = [], [], []
                labels = []
                for b_idx, nom in block_noms.items():
                    xs.append(nom[0])
                    ys.append(nom[1])
                    zs.append(nom[2])
                    labels.append(f"{b_idx[0]}-{b_idx[1]}-{b_idx[2]}")
                ax_inset.scatter(xs, ys, zs, color='blue', alpha=0.5, s=10)
                
                # Subsample labels if too many to avoid clutter
                step = max(1, len(labels) // 20)
                for i in range(0, len(labels), step):
                    ax_inset.text(xs[i], ys[i], zs[i], labels[i], size=5, zorder=1, color='k')
                
                ax_inset.set_title("i-j-k Reference")
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                ax_inset.set_zticks([])
                
                plt.savefig(f'rds-{comp}_deformation_all.png')
                plt.close()
            
    return time_history, z_hist, vel_hist, root_acc_history, corner_acc_hist, max_g_force, metrics

if __name__ == "__main__":
    # Get basic defaults, override specifically if needed
    
    cfg = get_default_config()
    # 예: 전면 하단 꼭짓점 낙하 자세로 설정
    cfg["drop_mode"] = "L-F-B" 
    cfg["include_paperbox"] = False # 로컬 테스트용 오버라이드
    cfg["drop_height"] = 0.5    
    cfg["plot_results"] = True
    cfg['sim_duration'] = 2.0
    run_simulation(cfg)
