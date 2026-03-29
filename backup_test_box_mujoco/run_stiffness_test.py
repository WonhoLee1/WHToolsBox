import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from run_discrete_builder import get_default_config, get_single_body_instance

def generate_stiffness_test_xml(body_name, test_type="BENDING", cfg=None):
    cfg = get_default_config(cfg)
    body = get_single_body_instance(body_name, cfg)
    
    xml_str = []
    xml_str.append('<mujoco model="stiffness_test">')
    if test_type == "TWIST":
        # TWIST의 경우 이산화된 직사각형 블록들이 회전하면서 모서리가 비현실적으로 파고드는 
        # 기하학적 형상 충돌(Self-Collision Penetration)이 무한대의 저항 토크를 낳는 것을 방지
        xml_str.append('  <option integrator="implicitfast" timestep="0.001" gravity="0 0 0"><flag contact="disable"/></option>')
    else:
        xml_str.append('  <option integrator="implicitfast" timestep="0.001" gravity="0 0 0"><flag contact="enable"/></option>')
    xml_str.append('  <default>')
    # Need less armature to prevent artificial stiffness, keeping it realistic but stable
    xml_str.append('    <joint armature="0.01" damping="5.0"/>')
    xml_str.append('    <geom friction="0.8" solref="0.02 1.0" solimp="0.9 0.95 0.001"/>')
    xml_str.append('  </default>')
    xml_str.append('  <worldbody>')
    xml_str.append('    <light pos="0 0 5" dir="0 0 -1"/>')
    # No floor, we test in free space with clamps
    
    # Render discrete body
    bodies_xml = body.get_worldbody_xml_strings(indent_level=2)
    for line in bodies_xml:
        xml_str.append(line)
        
    # --- Test Rig Setup ---
    # Include block half-extents (dx, dy, dz) to find the true outer surfaces
    min_x = min(blk.cx - blk.dx for blk in body.blocks.values())
    max_x = max(blk.cx + blk.dx for blk in body.blocks.values())
    min_y = min(blk.cy - blk.dy for blk in body.blocks.values())
    max_y = max(blk.cy + blk.dy for blk in body.blocks.values())
    min_z = min(blk.cz - blk.dz for blk in body.blocks.values())
    max_z = max(blk.cz + blk.dz for blk in body.blocks.values())
    
    # Identify clamp and ram nodes based on test type
    weld_eqs = []
    
    cy_mid = (min_y + max_y) / 2
    cz_mid = (min_z + max_z) / 2
    
    # 튼튼한 Clamp 연결을 위해 피검사 대상 자체의 최상급 물성 또는 강력한 강성을 부여 (Tape 등 약한 물성 배제)
    clamp_solref = body.material_props.get("solref", "0.001 1.0")
    clamp_solimp = body.material_props.get("solimp", "0.95 0.99 0.001 0.5 2")
    
    # 만약 테스트 보디가 굉장히 부드럽다면(Cushion 등) 억지로라도 고정력을 확보하기 위한 보정
    try:
        if float(clamp_solref.split()[0]) > 0.05:
            clamp_solref = "0.01 1.0"
    except: pass
    
    if test_type == "BENDING":
        # Clamps at both ends (-x, +x), Ram in the middle
        xml_str.append(f'    <body name="clamp_L" pos="{min_x - 0.051} {cy_mid} {cz_mid}"><geom type="box" size="0.05 {body.height/2} {body.depth/2}" rgba="1 0 0 0.5" contype="0" conaffinity="0"/></body>')
        xml_str.append(f'    <body name="clamp_R" pos="{max_x + 0.051} {cy_mid} {cz_mid}"><geom type="box" size="0.05 {body.height/2} {body.depth/2}" rgba="1 0 0 0.5" contype="0" conaffinity="0"/></body>')
        
        xml_str.append(f'    <body name="ram" pos="0 {cy_mid} {max_z + 0.051}">')
        xml_str.append('      <joint name="ram_z" type="slide" axis="0 0 1"/>')
        xml_str.append(f'      <geom type="cylinder" size="0.05 {body.height/2}" euler="90 0 0" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>')
        xml_str.append('    </body>')
        
        # Welds
        for blk in body.blocks.values():
            blk_name = f"b_{body.name.lower()}_{blk.idx[0]}_{blk.idx[1]}_{blk.idx[2]}"
            if abs(blk.cx - min_x) < blk.dx * 2:
                weld_eqs.append(f'    <weld body1="clamp_L" body2="{blk_name}" solref="{clamp_solref}" solimp="{clamp_solimp}"/>')
            elif abs(blk.cx - max_x) < blk.dx * 2:
                weld_eqs.append(f'    <weld body1="clamp_R" body2="{blk_name}" solref="{clamp_solref}" solimp="{clamp_solimp}"/>')
            elif abs(blk.cx) < blk.dx * 3: # middle area
                weld_eqs.append(f'    <weld body1="ram" body2="{blk_name}" solref="{clamp_solref}" solimp="{clamp_solimp}"/>')

    elif test_type == "TWIST":
        # Clamp at -x, Twist at +x
        xml_str.append(f'    <body name="clamp_L" pos="{min_x - 0.051} {cy_mid} {cz_mid}"><geom type="box" size="0.05 {body.height/2} {body.depth/2}" rgba="1 0 0 0.5" contype="0" conaffinity="0"/></body>')
        xml_str.append(f'    <body name="ram" pos="{max_x + 0.051} {cy_mid} {cz_mid}">')
        xml_str.append('      <joint name="ram_twist" type="hinge" axis="1 0 0"/>') # twist around x axis
        xml_str.append(f'      <geom type="box" size="0.05 {body.height/2} {body.depth/2}" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>')
        xml_str.append('    </body>')
        
        for blk in body.blocks.values():
            blk_name = f"b_{body.name.lower()}_{blk.idx[0]}_{blk.idx[1]}_{blk.idx[2]}"
            if abs(blk.cx - min_x) < blk.dx * 2:
                weld_eqs.append(f'    <weld body1="clamp_L" body2="{blk_name}" solref="{clamp_solref}" solimp="{clamp_solimp}"/>')
            elif abs(blk.cx - max_x) < blk.dx * 2:
                weld_eqs.append(f'    <weld body1="ram" body2="{blk_name}" solref="{clamp_solref}" solimp="{clamp_solimp}"/>')
                
    elif test_type == "COMPRESSION":
        # Clamp at -z, Compress at +z
        # Bottom clamp is physically welded
        xml_str.append(f'    <body name="clamp_B" pos="0 0 {min_z - 0.011}"><geom type="box" size="{body.width/2} {body.height/2} 0.01" rgba="1 0 0 0.5" contype="0" conaffinity="0"/></body>')
        
        # Ram (Top Press) welded to top.
        xml_str.append(f'    <body name="ram" pos="0 0 {max_z + 0.011}">')
        xml_str.append('      <joint name="ram_z" type="slide" axis="0 0 1"/>')
        xml_str.append(f'      <geom type="box" size="{body.width/2} {body.height/2} 0.01" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>')
        xml_str.append('    </body>')
        
        # [수정] 외부 프레스와 블록을 묶는 Weld가 튜닝 중인 블록 본연의 물성보다 약하면(Tape) 
        # 블록이 찌그러지기 전에 프레스 연결부가 먼저 늘어나서 하중이 전달되지 않습니다. 
        # 따라서 테스트 타겟 대상의 물성을 그대로 가져오거나 극단 강성을 주어야 합니다.
        solref = body.material_props.get("solref", "0.0001 1.0")
        solimp = body.material_props.get("solimp", "0.99 0.999 0.001 0.5 2")
        
        for blk in body.blocks.values():
            blk_name = f"b_{body.name.lower()}_{blk.idx[0]}_{blk.idx[1]}_{blk.idx[2]}"
            if abs(blk.cz - min_z) < blk.dz * 2:
                weld_eqs.append(f'    <weld body1="clamp_B" body2="{blk_name}" solref="{solref}" solimp="{solimp}"/>')
            elif abs(blk.cz - max_z) < blk.dz * 2:
                weld_eqs.append(f'    <weld body1="ram" body2="{blk_name}" solref="{solref}" solimp="{solimp}"/>')
                
    xml_str.append('  </worldbody>')
    
    xml_str.append('  <actuator>')
    if test_type == "BENDING" or test_type == "COMPRESSION":
        xml_str.append('    <position name="act_ram" joint="ram_z" kp="200000000" kv="100000"/>')
    elif test_type == "TWIST":
        # TWIST는 무거운 Ram 블록이 제자리 회전을 해야 하므로 너무 강한 kv(감쇠)를 주면 
        # 극초반 정지 마찰벽(관성 저항력)이 수십 Nm 단위로 치솟는 기이한 동역학 튀는 현상이 발생합니다.
        # 회전 관성에 맞도록 위치/감쇠 게인을 대폭 낮추어 순수 대상체(Box)의 강성만 측정되게 세팅합니다.
        xml_str.append('    <position name="act_ram" joint="ram_twist" kp="2000000" kv="50000"/>')
    xml_str.append('  </actuator>')
    
    xml_str.append('  <equality>')
    for line in body.get_weld_xml_strings():
        xml_str.append(line)
    for w in weld_eqs:
        xml_str.append(w)
    xml_str.append('  </equality>')
    
    xml_str.append('</mujoco>')
    
    export_path = f"stiffness_{body_name}_{test_type}.xml"
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_str))
    
    return export_path

def run_stiffness_test(xml_path, test_type="BENDING", target_disp=0.1, duration=3.0, plot_results=True):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    n_steps = int(duration / m.opt.timestep)
    
    time_history = []
    disp_history = []
    force_history = []
    
    act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_ram")
    
    # Find the real joint/body id for displacement tracking
    # For ram tracking (since ram connects to the tested body)
    ram_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ram")
    initial_ram_pos = np.copy(d.xpos[ram_body_id])
    
    if test_type == "TWIST":
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ram_twist")
    else:
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ram_z")
        
    mujoco.mj_forward(m, d)

    print(f"Running Stiffness Test: {xml_path}")
    for step in range(n_steps):
        t = d.time
        # 부드러운 S자 곡선 궤적 유도: 시작 속도=0, 종료 속도=0
        # 이를 통해 t=0 일 때의 급격한 가속도(Inertial Kick, 튀는 현상) 물리적으로 완벽히 제거
        ratio = 0.5 * (1.0 - np.cos(np.pi * (t / duration)))
        
        current_target = target_disp * ratio
        if test_type == "TWIST":
            d.ctrl[act_id] = np.radians(current_target)
        else:
            d.ctrl[act_id] = current_target
        
        try:
            mujoco.mj_step(m, d)
        except Exception as e:
            import sys
            sys.stderr.write(f"\n[Stiffness Test Error] Simulation unstable or crashed (mujoco.mj_step): {e}\n")
            break  # 조기 종료하여 그동안 모인 쓰레기 데이터를 평가부로 넘김 (평가부에서 높은 Loss 부과)
        
        if step % 10 == 0:
            time_history.append(t)
            
            # Record actual physical displacement of the ram body (not just actuator ideal qpos)
            qpos = d.qpos[m.jnt_qposadr[joint_id]]
            current_qpos = qpos[0] if isinstance(qpos, np.ndarray) else qpos
            
            if test_type == "TWIST":
                actual_disp = np.degrees(current_qpos)
            elif test_type == "BENDING" or test_type == "COMPRESSION":
                actual_disp = current_qpos
                
            act_force = float(d.actuator_force[act_id][0] if isinstance(d.actuator_force[act_id], np.ndarray) else d.actuator_force[act_id])
            
            disp_history.append(actual_disp)
            force_history.append(act_force)
            
    # Smoothing force data for cleaner output using simple moving avg
    force_arr = np.array(force_history)
    disp_arr = np.array(disp_history)
    
    # (0,0) 원점 지나는 순수 강성 곡선을 위해 첫 스텝 기준 0점 보정 안전 장치
    if len(force_arr) > 0:
        force_arr = force_arr - force_arr[0]
        disp_arr = disp_arr - disp_arr[0]
        
    # COMPRESSION 직관적 그래프(양수 반전) 처리
    if test_type == "COMPRESSION":
        force_arr = np.abs(force_arr)
        disp_arr = np.abs(disp_arr)
        
    window = max(1, len(force_arr) // 50)
    force_smooth = np.convolve(force_arr, np.ones(window)/window, mode='valid')
    
    # 스무딩(Moving Average)으로 인해 발생한 위상 지연 보상 (윈도우 절반만큼 시프트)
    offset = window // 2
    disp_smooth = disp_arr[offset : offset + len(force_smooth)]
    
    # 튀는 현상을 안정화하며, 차트 작도를 위해 맨 앞에 (0,0) 좌표 강제 삽입
    force_smooth = np.insert(force_smooth, 0, 0.0)
    disp_smooth = np.insert(disp_smooth, 0, 0.0)
    
    if plot_results:
        plt.figure(figsize=(10, 5))
        plt.plot(disp_smooth, force_smooth, linewidth=2, color='tab:blue')
        plt.title(f"Stiffness Curve: {os.path.basename(xml_path)}")
        if test_type == "TWIST":
            plt.xlabel("Rotation Angle [degrees]")
            plt.ylabel("Actuator Torque [Nm]")
        else:
            plt.xlabel("Displacement [m]")
            plt.ylabel("Actuator Force [N]")
        plt.grid(True)
        
        out_png = xml_path.replace(".xml", ".png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(f"Saved stiffness plot to {out_png}")
        plt.close()
        
    stiffness = 0.0
    # Optional: basic stiffness estimate (k = max_F / max_disp)
    if len(disp_smooth) > 0 and abs(disp_smooth[-1]) > 1e-4:
        stiffness = force_smooth[-1] / disp_smooth[-1]
        if plot_results:
            unit = "Nm/deg" if test_type == "TWIST" else "N/m"
            print(f" -> Approximate Peak Stiffness (k): {stiffness:,.1f} {unit}")
            
    return np.array(time_history)[:len(force_smooth)], disp_smooth, force_smooth, stiffness

if __name__ == "__main__":
    targets = [
        ("BChassis", "BENDING", +0.05), # 5cm bending
        ("BChassis", "TWIST", 30.0), # 30 degrees twist
        ("BOpenCell", "BENDING", +0.05), # 5cm bending
        ("BOpenCell", "TWIST", 30.0),   # 30 degrees twist
        ("BCushion", "BENDING", +0.05), # 5cm bending
        ("BCushion", "TWIST", 30.0),   # 30 degrees twist
        ("BCushion", "COMPRESSION", -0.05) # 5cm compression
    ]
    
    for body_name, test_type, disp in targets:
        xml_file = generate_stiffness_test_xml(body_name, test_type)
        print(f"\\nStarted {test_type} on {body_name}...")
        run_stiffness_test(xml_file, test_type, target_disp=disp)
        
    print("\\nAll stiffness tests complete.")
    
