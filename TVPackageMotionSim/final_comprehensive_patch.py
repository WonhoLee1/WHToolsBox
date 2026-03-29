import os
import re

def final_patch():
    sim_path = r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_drop_simulation.py"
    builder_path = r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_discrete_builder\__init__.py"
    
    # 1. Update run_drop_simulation.py
    if os.path.exists(sim_path):
        with open(sim_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # [RENAME] edge -> corner
        content = content.replace('cush_edge_solref', 'cush_corner_solref')
        content = content.replace('cush_edge_solimp', 'cush_corner_solimp')
        
        # [HIGHLIGHT] Add yellow highlighting for corners
        highlight_code = """    # [VISUALIZATION] 코너(수직 엣지) 블록 색상 변경 (Yellow)
    corner_count = 0
    for comp_name, (nx_max, ny_max, nz_max) in comp_max_idxs.items():
        if "cushion" in comp_name:
            for gid in range(model.ngeom):
                t_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if t_name and t_name.lower().startswith(f"g_{comp_name}_"):
                    parts = t_name.split('_')
                    if len(parts) >= 5:
                        try:
                            c_i, c_j = int(parts[-3]), int(parts[-2])
                            if (c_i == 0 or c_i == nx_max) and (c_j == 0 or c_j == ny_max):
                                model.geom_rgba[gid] = [1.0, 1.0, 0.0, 1.0] # Yellow
                                corner_count += 1
                        except: pass
    
    if comp_max_idxs:
        log_and_print(f"  >> [Plasticity] Component Index Ranges: {comp_max_idxs}")
        log_and_print(f"  >> [Plasticity] {corner_count} corner geoms highlighted in Yellow.")
"""
        insertion_point = '    if comp_max_idxs:\n        log_and_print(f"  >> [Plasticity] Component Index Ranges detected: {comp_max_idxs}")'
        content = content.replace(insertion_point, highlight_code)

        # [PLASTICITY] Replace apply_plastic_deformation entire block
        # Start: def apply_plastic_deformation():
        # End: state['applied'] = False
        
        # We need a robust regex or finding start/end because indices might shift
        new_plasticity_fn = """    def apply_plastic_deformation():
        if not enable_plasticity: return
        current_penetrations = {}
        
        # 1. 런타임에 바닥(ground)과 쿠션 상호작용 검사 (하중 및 침투량 집계)
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

        # 1.5. 집계된 정보를 바탕으로 소성 변형 여부 판정
        for gid, hit in geom_hits.items():
            parts = hit['parts']
            try:
                c_i = int(parts[-3]); c_j = int(parts[-2])
                comp_name = "_".join(parts[1:-3]).lower()
                nx_max, ny_max, nz_max = comp_max_idxs.get(comp_name, (0,0,0))
                
                bx = (c_i == 0 or c_i == nx_max)
                by = (c_j == 0 or c_j == ny_max)
                
                if bx and by:
                    sz = model.geom_size[gid]
                    areas = [sz[1]*sz[2], sz[0]*sz[2], sz[0]*sz[1]]
                    g_area = 4.0 * min(areas) 
                    ma = 2 # Z-axis deformation preference
                    
                    current_penetrations[gid] = hit['max_p']
                    pressure = hit['sum_f'] / g_area if g_area > 0 else 0
                    if pressure > yield_stress_pa:
                        if hit['max_p'] > 1e-6:
                            if gid not in geom_state_tracker:
                                geom_state_tracker[gid] = {'max_p': 0.0, 'major_axis': ma}
                                log_and_print(f"  [Plasticity] Corner Activated: {hit['name']} (Pressure: {pressure/1e3:.1f}kPa)")
                            
                            if hit['max_p'] > geom_state_tracker[gid]['max_p']:
                                geom_state_tracker[gid]['max_p'] = hit['max_p']
                                geom_state_tracker[gid]['major_axis'] = ma
            except: pass

        # 2. 실시간 소성 변형 적용 (침투 감소 시 즉시 반영)
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
                    half_shrink = deformation / 2.0
                    shift_amount = deformation / 2.0
                    
                    model.geom_size[gid][major_axis] = max(0.001, model.geom_size[gid][major_axis] - half_shrink)
                    shift_vec = np.zeros(3)
                    shift_vec[major_axis] = inward_dir[major_axis] * shift_amount
                    model.geom_pos[gid] += shift_vec
                    
                    total_shrink = original_geom_size[gid][major_axis] - model.geom_size[gid][major_axis]
                    color_scale = min(1.0, total_shrink / 0.005) # 5mm sensitivity
                    
                    model.geom_rgba[gid][0] = 0.5 * (1.0 - color_scale)
                    model.geom_rgba[gid][1] = 0.2 * (1.0 - color_scale)
                    model.geom_rgba[gid][2] = 0.6 + 0.4 * color_scale 
                    model.geom_rgba[gid][3] = 1.0
                    
                    state['max_p'] = curr_p
                    t_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    log_and_print(f"  [Plasticity] {t_name} Deforming: -{total_shrink*1000:.1f}mm (Current Def: {deformation*1000:.2f}mm)")
            if curr_p > state['max_p']: state['max_p'] = curr_p
"""

        # Complex replacement of old apply_plastic_deformation
        # We find from '    def apply_plastic_deformation():' to '                state[\'applied\'] = False\n'
        start_pattern = r'    def apply_plastic_deformation\(\):'
        end_pattern = r'                state\[\'applied\'\] = False\n'
        
        match_start = re.search(start_pattern, content)
        if match_start:
            # Look for the end after the start
            match_end = re.search(end_pattern, content[match_start.start():])
            if match_end:
                total_end = match_start.start() + match_end.end()
                content = content[:match_start.start()] + new_plasticity_fn + content[total_end:]

        # [USER EDITS] Update test_run_case_1 values
        content = content.replace('cfg["cush_weld_solref_stiff"] = 0.02', 'cfg["cush_weld_solref_stiff"] = 0.008')
        content = content.replace('cfg["cush_weld_solref_damp"]  = 0.6', 'cfg["cush_weld_solref_damp"]  = 0.8')
        content = content.replace('cfg["cush_contact_solimp"]    = "0.1 0.95 0.002 0.5 2"', 'cfg["cush_contact_solimp"]    = "0.1 0.95 0.01 0.5 2"')
        content = content.replace('cfg["cush_edge_solimp"]       = "0.1 0.95 0.004 0.5 2"', 'cfg["cush_corner_solimp"]    = "0.1 0.95 0.01 0.5 2"')
        
        # [YIELD STRESS] Update default
        content = content.replace('yield_stress_pa = config.get("cush_yield_stress", 0.1) * 1e6', 'yield_stress_pa = config.get("cush_yield_stress", 0.01) * 1e6')

        with open(sim_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Updated run_drop_simulation.py (Full Patch)")

    # 2. Update run_discrete_builder/__init__.py
    if os.path.exists(builder_path):
        with open(builder_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # is_edge_block logic
        old_logic = "bx = (i == 0 or i == nx_max)\n        by = (j == 0 or j == ny_max)\n        bz = (k == 0 or k == nz_max)\n        \n        return bx or by or bz"
        new_logic = "bx = (i == 0 or i == nx_max)\n        by = (j == 0 or j == ny_max)\n        \n        return bx and by"
        content = content.replace(old_logic, new_logic)
        
        # Rename edge -> corner
        content = content.replace('edge_solref', 'corner_solref')
        content = content.replace('edge_solimp', 'corner_solimp')
        content = content.replace('cush_edge_solref', 'cush_corner_solref')
        content = content.replace('cush_edge_solimp', 'cush_corner_solimp')

        with open(builder_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Updated run_discrete_builder/__init__.py")

final_patch()
