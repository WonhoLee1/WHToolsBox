import os

path = r'c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_drop_simulation_v2.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update test_run_case_1
old_cfg = '    cfg["cush_yield_strain"] = 0.1 # 10% 변형 시 소성 변형 발생'
new_cfg = '    cfg["cush_yield_strain"] = 0.1 # 10% 변형 시 소성 변형 발생\n    cfg["plasticity_algorithm"] = 2 # 1: Pressure/Penetration, 2: Strain(Neighbor Distance)'

content = content.replace(old_cfg, new_cfg)

# 2. Insert v1 and Dispatcher before v2 definition
v2_start_marker = '    def apply_plastic_deformation_v2():'
v1_code = """    def apply_plastic_deformation_v1():
        if not enable_plasticity: return
        current_penetrations = {}
        geom_hits = {} # gid -> {'max_p', 'sum_f', 'local_n', 'parts'}
        for i in range(data.ncon):
            con = data.contact[i]
            g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            target_geom = -1
            if g1_name and "ground" in g1_name.lower(): target_geom = con.geom2
            elif g2_name and "ground" in g2_name.lower(): target_geom = con.geom1
            if target_geom != -1:
                t_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, target_geom)
                if t_name and "cushion" in t_name.lower(): 
                    parts = t_name.split('_')
                    if len(parts) >= 5:
                        gid = target_geom
                        if gid not in geom_hits:
                            geom_hits[gid] = {'max_p': 0.0, 'sum_f': 0.0, 'local_n': np.zeros(3), 'parts': parts, 'name': t_name}
                        force_vec = np.zeros(6)
                        mujoco.mj_contactForce(model, data, i, force_vec)
                        geom_hits[gid]['sum_f'] += abs(force_vec[0])
                        pen = -con.dist
                        if pen > geom_hits[gid]['max_p']:
                            geom_hits[gid]['max_p'] = pen
                            body_id = model.geom_bodyid[gid]
                            nw = con.frame[:3]
                            geom_hits[gid]['local_n'] = data.xmat[body_id].reshape(3,3).T @ nw
        for gid, hit in geom_hits.items():
            parts = hit['parts']
            try:
                c_i = int(parts[-3]); c_j = int(parts[-2])
                comp_name = "_".join(parts[1:-3]).lower()
                nx_max, ny_max, nz_max = comp_max_idxs.get(comp_name, (0,0,0))
                if (c_i == 0 or c_i == nx_max) and (c_j == 0 or c_j == ny_max):
                    sz = model.geom_size[gid]
                    areas = [sz[1]*sz[2], sz[0]*sz[2], sz[0]*sz[1]]
                    g_area = 4.0 * min(areas) 
                    local_n = hit['local_n']
                    ma = int(np.argmax(np.abs(local_n))) 
                    current_penetrations[gid] = hit['max_p']
                    pressure = hit['sum_f'] / g_area if g_area > 0 else 0
                    if pressure > yield_stress_pa:
                        if hit['max_p'] > 1e-6:
                            if gid not in geom_state_tracker:
                                geom_state_tracker[gid] = {'max_p': 0.0, 'major_axis': ma}
                                log_and_print(f"  [Plasticity] Corner Activated(v1): {hit['name']} (Pressure: {pressure/1e3:.1f}kPa, Axis: {ma})")
                            if hit['max_p'] > geom_state_tracker[gid]['max_p']:
                                geom_state_tracker[gid]['max_p'] = hit['max_p']
                                geom_state_tracker[gid]['major_axis'] = ma
            except: pass
        for gid, state in geom_state_tracker.items():
            curr_p = current_penetrations.get(gid, 0.0)
            if state['max_p'] > 0.0001 and curr_p < state['max_p']:
                delta_p = state['max_p'] - curr_p
                deformation = delta_p * plasticity_ratio
                if deformation > 1e-6:
                    body_id = model.geom_bodyid[gid]
                    b_pos = model.body_pos[body_id]
                    inward_dir = -np.sign(b_pos)
                    for i_ax in range(3):
                        if abs(inward_dir[i_ax]) < 0.1: inward_dir[i_ax] = -1.0
                    major_axis = state.get('major_axis', 2)
                    model.geom_size[gid][major_axis] = max(0.001, model.geom_size[gid][major_axis] - (deformation/2.0))
                    shift_vec = np.zeros(3)
                    shift_vec[major_axis] = inward_dir[major_axis] * (deformation/2.0)
                    model.geom_pos[gid] += shift_vec
                    total_shrink = original_geom_size[gid][major_axis] - model.geom_size[gid][major_axis]
                    color_scale = min(1.0, total_shrink / 0.005)
                    model.geom_rgba[gid][0:3] = [0.5*(1-color_scale), 0.2*(1-color_scale), 0.6+0.4*color_scale]
                    state['max_p'] = curr_p
                    t_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    log_and_print(f"  [Plasticity] {t_name} Deforming(v1): -{total_shrink*1000:.1f}mm")
            if curr_p > state['max_p']: state['max_p'] = curr_p

    def apply_plasticity():
        algo = config.get("plasticity_algorithm", 2)
        if algo == 1:
            apply_plastic_deformation_v1()
        else:
            apply_plastic_deformation_v2()

"""
content = content.replace(v2_start_marker, v1_code + v2_start_marker)

# 3. Update the loop call
old_call = '                apply_plastic_deformation_v2()  # 매 스텝마다 변형률(v2) 체킹'
new_call = '                apply_plasticity()  # 설정에 따른 소성 변형 알고리즘(v1/v2) 차등 적용'
content = content.replace(old_call, new_call)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Success")
