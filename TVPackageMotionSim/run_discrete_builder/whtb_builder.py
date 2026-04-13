import os
import io
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable, Union
from .whtb_utils import get_local_pose, parse_drop_target
from .whtb_config import get_default_config
from .whtb_base import BaseDiscreteBody
from .whtb_models import (
    BPaperBox, BCushion, BOpenCellCohesive, BOpenCell, BChassis, BAuxBoxMass, BUnitBlock
)

def get_single_body_instance(body_name: str, config: Optional[Dict[str, Any]] = None) -> BaseDiscreteBody:
    """
    특정 단일 부품(개단품)의 기하학적 형상과 격자 정보를 추출하기 위한 유틸리티 함수입니다.
    부품별 강성 평가나 개별 시각화 검증 시 사용됩니다.
    
    Args:
        body_name (str): 생성할 클래스 명칭 (예: 'BPaperBox', 'BCushion' 등).
        config (dict, optional): 사용자 정의 설정값. 미지정 시 기본 설정을 사용합니다.
        
    Returns:
        BaseDiscreteBody: 격자(Geometry)가 생성된 해당 부품 객체.
    """
    config = get_default_config(config)
    
    # 1. 외곽 치수 계산
    box_w = config["box_w"]; box_h = config["box_h"]; box_d = config["box_d"]
    box_thick = config["box_thick"]
    
    # 2. 완충재 치수 (박스 내부를 꽉 채우는 기준)
    cush_gap = config["cush_gap"]
    cush_w, cush_h, cush_d = box_w - 2 * box_thick, box_h - 2 * box_thick, box_d - 2 * box_thick
    
    # 3. 내부 제품(Assy) 치수 및 배치 위치 계산
    assy_w = config.get("assy_w", cush_w - 0.3); assy_h = config.get("assy_h", cush_h - 0.3)
    opencell_d = config["opencell_d"]; opencellcoh_d = config["opencellcoh_d"]; chassis_d = config["chassis_d"]
    assy_d = opencell_d + opencellcoh_d + chassis_d
    
    occ_ithick = config["occ_ithick"]
    
    # Z축 적층 위치 (Assy 중심 기준 상대 좌표)
    oc_z   = assy_d/2 - opencell_d/2
    occ_z  = oc_z - opencell_d/2 - opencellcoh_d/2
    chas_z = occ_z - opencellcoh_d/2 - chassis_d/2
    
    # 필수 절단선 (가운데 구멍/패턴 유지를 위해 격자가 반드시 지나가야 할 지점)
    occ_cut_x = [-assy_w/2 + occ_ithick, assy_w/2 - occ_ithick]
    occ_cut_y = [-assy_h/2 + occ_ithick, assy_h/2 - occ_ithick]
    
    # 4. 부품별 인스턴스화 및 지오메트리 빌드
    comp = config.get("components", {})
    
    if body_name == "BPaperBox":
        p = comp.get("paper", {})
        b = BPaperBox("BPaperBox", box_w, box_h, box_d, 
                      p.get("mass", config.get("mass_paper", 4.0)), 
                      p.get("div", config.get("box_div", [5, 5, 2])), 
                      box_thick, config["mat_paper"], 
                      p.get("use_weld", config.get("box_use_weld", False)),
                      inertia=p.get("inertia", config.get("inertia_paper")))
        b.build_geometry(local_offset=[0,0,0], 
                         required_cuts_x=[-box_w/2+box_thick, box_w/2-box_thick],
                         required_cuts_y=[-box_h/2+box_thick, box_h/2-box_thick],
                         required_cuts_z=[-box_d/2+box_thick, box_d/2-box_thick])
        # [4. CONTACT & PAIR PARAMETERS] : 명시적 접촉 쌍 설정 (A1/A2 통합 점검)
        # (create_model 내부에서 전역적으로 처리되므로 여기서는 생략 가능하지만 구조 유지를 위해 둠)
        return b
        
    elif body_name == "BCushion":
        p = comp.get("cushion", {})
        assy_bbox = [-assy_w/2, assy_w/2, -assy_h/2, assy_h/2, -assy_d/2, assy_d/2]
        BCushion_cutter = { "center": [0, 0, 0, cush_w*0.5, cush_h*0.5, cush_d*2] }
        b = BCushion("BCushion", cush_w, cush_h, cush_d, 
                     p.get("mass", config.get("mass_cushion", 2.0)), 
                     p.get("div", config.get("cush_div", [5, 5, 3])), 
                     config["mat_cush"], assy_bbox, cush_gap, BCushion_cutter, 
                     p.get("use_weld", config.get("cush_use_weld", True)),
                     inertia=p.get("inertia", config.get("inertia_cushion")))
        
        # 완충재는 제품 안착 공간과 커터 위치를 모두 절단선으로 포함
        req_cuts_cush_x = [-assy_w/2 - cush_gap, assy_w/2 + cush_gap]
        req_cuts_cush_y = [-assy_h/2 - cush_gap, assy_h/2 + cush_gap]
        req_cuts_cush_z = [-assy_d/2 - cush_gap, assy_d/2 + cush_gap]
        for cut_vals in BCushion_cutter.values():
            ctx, cty, ctz, cw, ch, cd = cut_vals
            req_cuts_cush_x.extend([ctx - cw/2, ctx + cw/2])
            req_cuts_cush_y.extend([cty - ch/2, cty + ch/2])
            req_cuts_cush_z.extend([ctz - cd/2, ctz + cd/2])
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=req_cuts_cush_x, required_cuts_y=req_cuts_cush_y, required_cuts_z=req_cuts_cush_z)
        return b
        
    elif body_name == "BOpenCell":
        p = comp.get("opencell", {})
        b = BOpenCell("BOpenCell", assy_w, assy_h, opencell_d, 
                      p.get("mass", config.get("mass_oc", 5.0)), 
                      p.get("div", config.get("opencell_div", [5, 5, 1])), 
                      config["mat_cell"], 
                      p.get("use_weld", config.get("opencell_use_weld", True)),
                      inertia=p.get("inertia", config.get("inertia_oc")))
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BOpenCellCohesive":
        p = comp.get("opencellcoh", {})
        b = BOpenCellCohesive("BOpenCellCohesive", assy_w, assy_h, opencellcoh_d, 
                              p.get("mass", config.get("mass_occ", 0.1)), 
                              p.get("div", config.get("opencellcoh_div", [5, 5, 1])), 
                              occ_ithick, config["mat_tape"], 
                              p.get("use_weld", config.get("opencellcoh_use_weld", True)),
                              inertia=p.get("inertia", config.get("inertia_occ")))
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BChassis":
        p = comp.get("chassis", {})
        b = BChassis("BChassis", assy_w, assy_h, chassis_d, 
                     p.get("mass", config.get("mass_chassis", 10.0)), 
                     p.get("div", config.get("chassis_div", [5, 5, 1])), 
                     config["mat_tv"], 
                     p.get("use_weld", config.get("chassis_use_weld", True)),
                     inertia=p.get("inertia", config.get("inertia_chassis")))
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BUnitBlock":
        size = config.get("unit_size", [1.0, 1.0, 1.0])
        div = config.get("unit_div", [5, 5, 5])
        mass = config.get("mass_cushion", 1.0)
        b = BUnitBlock("BUnitBlock", size[0], size[1], size[2], mass, div, config["mat_cush"])
        b.build_geometry(local_offset=[0,0,0])
        return b
        
    else: 
        raise ValueError(f"Unknown discrete body type: {body_name}")

def create_model(export_path: str, config: Optional[Dict[str, Any]] = None, logger: Callable = print) -> Tuple[str, float, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    설계 파라미터(Config)를 읽어 전체 포장 시스템을 조립하고 MuJoCo XML 파일을 생성합니다.
    
    Algorithm:
        1. Config 동기화: 부품 치수 및 물성 정보를 일관되게 업데이트함.
        2. 충돌 마스크 설정: 컨테이너와 내용물 간의 계층적 충돌 제어를 위해 Bitmask 적용.
        3. 부품 계층 구조화: PaperBox -> Cushion -> Assy(OpenCell/Tape/Chassis) 순으로 트리 구성.
        4. 격자 생성(Build Geometry): 각 부품을 미세 블록으로 쪼개고 공유 절단선을 맞춤.
        5. 관적 계산 및 XML 작성: 어셈블리의 전체 관성을 산출하고 MuJoCo 문법으로 내보냄.
    """
    config = get_default_config(config)
    
    # [WHTOOLS] 0. 내부 컴포넌트 관성 측정 및 Auto-Balancing 최적화 수행
    from run_discrete_builder.whtb_physics import analyze_and_balance_components
    config = analyze_and_balance_components(config, verbose=True)
    
    drop_mode = config["drop_mode"]; drop_height = config["drop_height"]
    include_paperbox = config["include_paperbox"]; include_cushion = config["include_cushion"]

    # 기본 치수 파라미터
    box_w, box_h, box_d = config["box_w"], config["box_h"], config["box_d"]
    box_thick = config["box_thick"]
    cush_gap = config["cush_gap"]
    cush_w, cush_h, cush_d = box_w - 2 * box_thick, box_h - 2 * box_thick, box_d - 2 * box_thick
    assy_w = config.get("assy_w", cush_w - 0.3); assy_h = config.get("assy_h", cush_h - 0.3)
    opencell_d, opencellcoh_d, chassis_d = config["opencell_d"], config["opencellcoh_d"], config["chassis_d"]
    assy_d = opencell_d + opencellcoh_d + chassis_d
    occ_ithick = config["occ_ithick"]
    
    mat_paper, mat_cush, mat_tape = config["mat_paper"], config["mat_cush"], config["mat_tape"]
    mat_cell, mat_tv = config["mat_cell"], config["mat_tv"]

    # [V5.6] 명시적 접촉 Pair 시스템으로 전환 (A1: 모든 마찰/접촉 계수는 pair에서 관리)
    # 모든 geom의 자동 충돌을 비활성화 (contype=0, conaffinity=0)
    for mat in [mat_paper, mat_cush, mat_cell, mat_tape, mat_tv]:
        mat["contype"] = "0"
        mat["conaffinity"] = "0"

    logger("\n" + "="*105)
    logger(" [Contact Configuration: Explicit Pair System Active]")
    logger(" - All bitmask collisions (contype/affinity) are disabled.")
    logger(" - Contacts will be generated as explicit <pair> elements.")
    logger("="*105 + "\n")

    # 2. 바디 객체 생성 (Dict 기반 설정 우선 적용)
    comp = config.get("components", {})
    root_container = BaseDiscreteBody("PackagingBox", 0,0,0, 0, [1,1,1], {})
    
    if include_paperbox: 
        p = comp.get("paper", {})
        b_paper = BPaperBox("BPaperBox", box_w, box_h, box_d, 
                            p.get("mass", config.get("mass_paper", 4.0)), 
                            p.get("div", config.get("box_div", [5, 5, 2])), 
                            box_thick, mat_paper, 
                            p.get("use_weld", config.get("box_use_weld", False)))
    else: b_paper = None
    
    assy_group = BaseDiscreteBody("AssySet", 0,0,0, 0, [1,1,1], {})
    # [WHTOOLS] 어셈블리 세트 전체를 하나로 거동하게 하기 위해 자유도(Joint) 부여 및 강체화
    assy_group.use_body_joints = True
    assy_group.use_internal_weld = False
    oc_z = assy_d/2 - opencell_d/2; occ_z = oc_z - opencell_d/2 - opencellcoh_d/2; chas_z = occ_z - opencellcoh_d/2 - chassis_d/2
    
    p_oc = comp.get("opencell", {})
    b_opencell = BOpenCell("BOpenCell", assy_w, assy_h, opencell_d, 
                           p_oc.get("mass", config.get("mass_oc", 5.0)), 
                           p_oc.get("div", config.get("opencell_div", [5, 5, 1])), 
                           mat_cell, 
                           p_oc.get("use_weld", config.get("opencell_use_weld", True)))
    
    p_occ = comp.get("opencellcoh", {})
    b_occ = BOpenCellCohesive("BOpenCellCohesive", assy_w, assy_h, opencellcoh_d, 
                              p_occ.get("mass", config.get("mass_occ", 0.1)), 
                              p_occ.get("div", config.get("opencellcoh_div", [5, 5, 1])), 
                              occ_ithick, mat_tape, 
                              p_occ.get("use_weld", config.get("opencellcoh_use_weld", True)))
    
    p_chas = comp.get("chassis", {})
    b_chassis = BChassis("BChassis", assy_w, assy_h, chassis_d, 
                         p_chas.get("mass", config.get("mass_chassis", 10.0)), 
                         p_chas.get("div", config.get("chassis_div", [5, 5, 1])), 
                         mat_tv, 
                         p_chas.get("use_weld", config.get("chassis_use_weld", True)))

    # [WHTOOLS] 컴포넌트가 강체 모드(use_weld=False)인 경우 개별 조인트를 제거하여 AssySet에 고정
    for b_comp in [b_opencell, b_occ, b_chassis]:
        if not b_comp.use_internal_weld:
            b_comp.use_body_joints = False

    b_cushion = None
    if include_cushion:
        p_cush = comp.get("cushion", {})
        assy_bbox = [-assy_w/2, assy_w/2, -assy_h/2, assy_h/2, -assy_d/2, assy_d/2]
        BCushion_cutter = { "center": [0, 0, 0, cush_w*0.5, cush_h*0.5, cush_d*2] }
        b_cushion = BCushion("BCushion", cush_w, cush_h, cush_d, 
                             p_cush.get("mass", config.get("mass_cushion", 2.0)), 
                             p_cush.get("div", config.get("cush_div", [5, 5, 3])), 
                             mat_cush, assy_bbox, cush_gap, BCushion_cutter, 
                             p_cush.get("use_weld", config.get("cush_use_weld", True)))
    
    # 3. 트리 구조 조립
    assy_group.add_child(b_opencell); assy_group.add_child(b_occ); assy_group.add_child(b_chassis)
    if include_paperbox: root_container.add_child(b_paper)
    if include_cushion: root_container.add_child(b_cushion)
    root_container.add_child(assy_group)

    # 4. 각 부품별 격자 빌드 (Geom Build) 및 지점 매칭
    if include_paperbox: 
        b_paper.build_geometry(required_cuts_x=[-box_w/2+box_thick, box_w/2-box_thick], 
                               required_cuts_y=[-box_h/2+box_thick, box_h/2-box_thick], 
                               required_cuts_z=[-box_d/2+box_thick, box_d/2-box_thick])
    
    if include_cushion:
        req_cuts_cush_x = [-assy_w/2 - cush_gap, assy_w/2 + cush_gap]
        req_cuts_cush_y = [-assy_h/2 - cush_gap, assy_h/2 + cush_gap]
        req_cuts_cush_z = [-assy_d/2 - cush_gap, assy_d/2 + cush_gap]
        for cut_vals in BCushion_cutter.values():
            ctx, cty, ctz, cw, ch, cd = cut_vals
            req_cuts_cush_x.extend([ctx - cw/2, ctx + cw/2])
            req_cuts_cush_y.extend([cty - ch/2, cty + ch/2])
            req_cuts_cush_z.extend([ctz - cd/2, ctz + cd/2])
        b_cushion.build_geometry(required_cuts_x=req_cuts_cush_x, required_cuts_y=req_cuts_cush_y, required_cuts_z=req_cuts_cush_z)
    
    occ_cut_x = [-assy_w/2 + occ_ithick, assy_w/2 - occ_ithick]; occ_cut_y = [-assy_h/2 + occ_ithick, assy_h/2 - occ_ithick]
    b_opencell.build_geometry(local_offset=[0, 0, oc_z], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
    b_occ.build_geometry(local_offset=[0, 0, occ_z], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
    b_chassis.build_geometry(local_offset=[0, 0, chas_z], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)

    # 보조 질량 추가
    aux_mass_objects = []
    component_aux = config.get("component_aux", {})
    for aux_name, aux_config in component_aux.items():
        b_aux_mass = BAuxBoxMass(name=aux_name, width=aux_config.get("size")[0], height=aux_config.get("size")[1], depth=aux_config.get("size")[2], mass=aux_config.get("mass"))
        b_aux_mass.build_geometry(local_offset=[aux_config.get("pos")[0], aux_config.get("pos")[1], aux_config.get("pos")[2] + chas_z])
        b_chassis.add_child(b_aux_mass); aux_mass_objects.append(b_aux_mass)

    # 5. 부품 간 인터페이스 용접
    inter_weld_xml = []
    for (i,j,k_occ), blk_occ in b_occ.blocks.items():
        if (i,j,0) in b_opencell.blocks: 
            inter_weld_xml.append(f'        <weld class="weld_bopencellcohesive" site1="s_BOpenCellCohesive_{i}_{j}_{0}_PZ" site2="s_BOpenCell_{i}_{j}_{0}_NZ"/>')
        if (i,j,0) in b_chassis.blocks: 
            inter_weld_xml.append(f'        <weld class="weld_bopencellcohesive" site1="s_BOpenCellCohesive_{i}_{j}_{0}_NZ" site2="s_BChassis_{i}_{j}_{0}_PZ"/>')

    for b_aux_mass in aux_mass_objects:
        blk_aux = b_aux_mass.blocks[(0, 0, 0)]; min_dist_sq = float('inf'); nearest_chassis_key = None
        for block_key, blk_chassis in b_chassis.blocks.items():
            dist_sq = (blk_chassis.cx - blk_aux.cx)**2 + (blk_chassis.cy - blk_aux.cy)**2 + (blk_chassis.cz - blk_aux.cz)**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq; nearest_chassis_key = block_key
        if nearest_chassis_key:
            ci, cj, ck = nearest_chassis_key
            # [V5.4.4 FIX] 부품의 use_internal_weld 설정에 따라 용접 대상 바디 명칭을 동적으로 결정함
            # weld=False 인 경우 부품 전체 바디(self.name)를 참조하고, True인 경우 개별 블록 바디를 참조함.
            body1_name = f"b_{b_aux_mass.name.lower()}_0_0_0" if b_aux_mass.use_internal_weld else b_aux_mass.name
            body2_name = f"b_{b_chassis.name.lower()}_{ci}_{cj}_{ck}" if b_chassis.use_internal_weld else b_chassis.name
            
            # [V5.11.2] 인터페이스 용접 전용 클래스 적용 (솔레프/솔임프 속성 제거)
            inter_weld_xml.append(f'        <weld class="weld_bauxboxmass" body1="{body1_name}" body2="{body2_name}"/>')

    # 6. 낙하 자세 제어
    drop_direction = config.get("drop_direction", "front")
    target_pt = parse_drop_target(drop_mode, drop_direction, box_w, box_h, box_d); target_dist = np.linalg.norm(target_pt)
    rot_axis = np.cross(target_pt, [0, 0, -target_dist])
    
    if np.linalg.norm(rot_axis) < 1e-6:
        rot_axis = np.array([1.0, 0.0, 0.0])
        angle_rad = 0.0 if target_pt[2] < 0 else np.pi
    else:
        rot_axis /= np.linalg.norm(rot_axis)
        dot_val = np.clip(np.dot(target_pt, [0, 0, -target_dist]) / (target_dist**2), -1, 1)
        angle_rad = np.arccos(dot_val)

    wx, wy, wz = get_local_pose([0,0,0], drop_height, rot_axis, angle_rad, target_dist)
    rot_str = f"{rot_axis[0]:.4f} {rot_axis[1]:.4f} {rot_axis[2]:.4f} {np.degrees(angle_rad):.4f}"
    
    # 7. XML 파일 작성
    xml_str_io = io.StringIO()
    xml_str_io.write('<mujoco model="discrete_custom_box">\n  <size memory="512M"/>\n')
    xml_str_io.write(f'  <option integrator="{config["sim_integrator"]}" timestep="{config["sim_timestep"]}" iterations="{config["sim_iterations"]}" noslip_iterations="{config["sim_noslip_iterations"]}" tolerance="{config["sim_tolerance"]}" impratio="{config["sim_impratio"]}" gravity="{config["sim_gravity"][0]} {config["sim_gravity"][1]} {config["sim_gravity"][2]}" density="{config.get("air_density", 1.225)}" viscosity="{config.get("air_viscosity", 1.81e-5)}">\n    <flag contact="enable"/>\n  </option>\n')
    xml_str_io.write('  <visual>\n    <quality shadowsize="0"/>\n    <global offwidth="0" offheight="0"/>\n')
    xml_str_io.write(f'    <headlight ambient="{config.get("light_head_ambient")}" diffuse="{config.get("light_head_diffuse")}" specular="0.07 0.07 0.07"/>\n    <map znear="0.01"/>\n  </visual>\n')
    xml_str_io.write('  <asset>\n    <texture type="skybox" builtin="gradient" rgb1="0.1 0.1 0.1" rgb2="0.3 0.3 0.3" width="1024" height="1024"/>\n')
    xml_str_io.write('    <texture type="2d" name="ground_tex" builtin="checker" mark="edge" rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75" markrgb="0.8 0.8 0.8" width="300" height="300"/>\n')
    xml_str_io.write('    <material name="ground_mat" texture="ground_tex" texrepeat="5 5" texuniform="true" reflectance="0.0"/>\n')
    xml_str_io.write('  </asset>\n')
    xml_str_io.write('  <default>\n    <joint armature="0.05" damping="1.0"/>\n    <geom/>\n')
    
    from .whtb_config import get_friction_standard
    
    # [A1/A2] 파트별 기본 클래스 생성: 시각화(Group/RGBA) 및 용접(Weld) 속성 통합
    for mat_name, mat_props in [("bpaperbox", mat_paper), ("bcushion", mat_cush), ("bopencell", mat_cell), ("bopencellcohesive", mat_tape), ("bchassis", mat_tv), ("bauxboxmass", mat_tv.copy())]:
        m_low = mat_name.lower()
        g_grp = 1 if "cushion" in m_low else 2 if any(x in m_low for x in ["chassis", "opencell", "adhesive", "tape", "cohesive"]) else 3 if "aux" in m_low else 0
        c_rgba = mat_props.get("rgba", "0.8 0.8 0.8 1")
        
        # [V5.11.0] CLEAN: contact_ 접두사 제거, 불필요한 contype/conaffinity/condim 삭제
        # [V5.11.2] RE-ENABLE: 자동 충돌 방지용 contype=0 conaffinity=0 명시적 추가
        xml_str_io.write(f'    <default class="{mat_name}">\n      <geom group="{g_grp}" rgba="{c_rgba}" contype="0" conaffinity="0"/>\n    </default>\n')
        if mat_name == "bcushion":
            xml_str_io.write(f'    <default class="{mat_name}_edge">\n      <geom group="{g_grp}" rgba="{c_rgba}" contype="0" conaffinity="0"/>\n    </default>\n')
            
        # [WHTOOLS] 용접(Weld) 전용 클래스 추가 (Stiffness 통합 제어)
        w_name = mat_name.replace("b", "", 1) if mat_name.startswith("b") else mat_name
        if "paperbox" in w_name: w_name = "paper"
        if "opencellcohesive" in w_name: w_name = "opencell" # Tape properties match cell/chassis logic
        
        w_prop = config["welds"].get(w_name, {"solref": [0.02, 1.0], "solimp": [0.1, 0.95, 0.005, 0.5, 2]})
        sr_w = " ".join(map(str, w_prop["solref"]))
        si_w = " ".join(map(str, w_prop["solimp"]))
        # [V5.11.1] MuJoCo: <weld>가 <default> 직접 자식 불가. <equality> 속성으로 정의하여 상속 유도.
        xml_str_io.write(f'    <default class="weld_{mat_name}">\n      <equality solref="{sr_w}" solimp="{si_w}"/>\n    </default>\n')
        
        # [WHTOOLS] Cushion Corner 전용 용접 클래스 추가
        if mat_name == "bcushion":
            w_prop_c = config["welds"].get("cushion_corner", w_prop)
            sr_wc = " ".join(map(str, w_prop_c["solref"]))
            si_wc = " ".join(map(str, w_prop_c["solimp"]))
            xml_str_io.write(f'    <default class="weld_{mat_name}_corner">\n      <equality solref="{sr_wc}" solimp="{si_wc}"/>\n    </default>\n')

    # [WHTOOLS] Ground 전용 클래스 정의 (물리 파라미터 통합)
    gr_f = f"{config.get('ground_friction')} {config.get('ground_friction')}"
    gr_sr = config.get("ground_solref")
    gr_si = config.get("ground_solimp")
    xml_str_io.write(f'    <default class="ground">\n      <geom friction="{gr_f}" solref="{gr_sr}" solimp="{gr_si}" condim="3" contype="0" conaffinity="0" group="0" material="ground_mat"/>\n    </default>\n')
    
    # --- Contact Pair Database Generation ---
    # 수집 엔진: 모든 바디의 모든 geom을 리스트업
    # [V5.10.0] 위치(pos) 및 크기(size) 데이터 추가
    all_geoms = [("ground", "ground", -1, (0, 0, 0), (2.5, 2.5, 0.1))] # (name, type, instance_id, pos, size)
    
    def _collect_metadata(body):
        t_map = {"bpaperbox":"paper", "bcushion":"cushion", "bopencell":"opencell", "bopencellcohesive":"paper", "bchassis":"chassis"}
        m_type = t_map.get(body.__class__.__name__.lower(), "part")
        
        for idx, blk in body.blocks.items():
            g_name = f"g_{body.name.lower()}_{idx[0]}_{idx[1]}_{idx[2]}"
            cur_type = m_type
            if hasattr(body, 'is_corner_block') and body.is_corner_block(*idx):
                g_name += "_edge"
                cur_type += "_edge"
            # 위치 및 크기 정보를 메타데이터에 포함
            all_geoms.append((g_name, cur_type, id(body), (blk.cx, blk.cy, blk.cz), (blk.dx, blk.dy, blk.dz)))
            
        for child in body.children: _collect_metadata(child)
    
    _collect_metadata(root_container)
    
    # Pair 생성을 위한 Default 클래스 및 Pair 구문 조립
    pair_classes_xml = ""
    contact_pairs_xml = ""
    defined_classes = set()
    
    for (t1, t2), p in config["contacts"].items():
        cls_name = f"cls_{t1}_{t2}"
        f_str = " ".join(map(str, get_friction_standard(p["friction"], 5)))
        sr_str = " ".join(map(str, p["solref"]))
        si_str = " ".join(map(str, p["solimp"]))
        
        pair_classes_xml += f'    <default class="{cls_name}">\n      <pair friction="{f_str}" solref="{sr_str}" solimp="{si_str}"/>\n    </default>\n'
        
        # 실제 조합 검색 및 Pair 추가
        for i in range(len(all_geoms)):
            for j in range(i+1, len(all_geoms)):
                gn1, gt1, gi1, p1, s1 = all_geoms[i]
                gn2, gt2, gi2, p2, s2 = all_geoms[j]
                
                # Rule 1: Exclusion (Same instance, unless one is floor)
                if gi1 == gi2 and gi1 != -1: continue
                
                # Rule 2: Type Match
                if (gt1 == t1 and gt2 == t2) or (gt1 == t2 and gt2 == t1):
                    # [V5.10.1] Distance-Based Pair Filter (Skip far away blocks)
                    # Ground 접촉은 필터를 적용하지 않음 (사용자 요청)
                    if gt1 != "ground" and gt2 != "ground":
                        dist = np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
                        max_dim_i = max(s1) * 2.0; max_dim_j = max(s2) * 2.0
                        if dist > 1.5 * max(max_dim_i, max_dim_j):
                            continue # 멀리 떨어져 있으면 Pair 생성을 건너뜀

                    contact_pairs_xml += f'    <pair class="{cls_name}" geom1="{gn1}" geom2="{gn2}"/>\n'

    xml_str_io.write(pair_classes_xml)
    xml_str_io.write('  </default>\n')
    xml_str_io.write(f'  <worldbody>\n    <camera name="iso_view" pos="4.5 -4.5 3.5" xyaxes="1 1 0 -0.4 0.4 1" mode="fixed"/>\n    <light pos="0 0 6" dir="0 0 -1" directional="false" diffuse="{config.get("light_main_diffuse")}" ambient="{config.get("light_main_ambient")}" castshadow="false"/>\n    <light pos="3 3 5" dir="-1 -1 -1" directional="false" diffuse="{config.get("light_sub_diffuse")}" castshadow="false"/>\n')
    xml_str_io.write(f'    <geom name="ground" type="plane" size="2.5 2.5 0.1" class="ground"/>\n')
    xml_str_io.write(f'    <body name="BPackagingBox" pos="{wx:.5f} {wy:.5f} {wz:.5f}" axisangle="{rot_str}">\n      <freejoint/>\n      <geom type="box" size="0.001 0.001 0.001" mass="0.000021" rgba="0 0 0 0"/>\n')
    for line in root_container.get_worldbody_xml_strings(indent_level=3): xml_str_io.write(line + "\n")
    xml_str_io.write('    </body>\n  </worldbody>\n')
    xml_str_io.write(f'  <contact>\n{contact_pairs_xml}  </contact>\n')
    xml_str_io.write('  <equality>\n')
    for line in root_container.get_weld_xml_strings() + inter_weld_xml: xml_str_io.write(line + "\n")
    xml_str_io.write('  </equality>\n</mujoco>\n')
    
    with open(export_path, "w", encoding="utf-8") as f: 
        f.write(xml_str_io.getvalue())
        
    total_mass, cog, moi, individual_details = root_container.calculate_inertia()
    logger(f"[Assembly Inertia Report] Mode: {drop_mode}\n - Total Mass: {total_mass:.4f} kg\n - Global CoG: {cog}\n - Global MoI: {moi}")
    
    return xml_str_io.getvalue(), total_mass, cog, moi, individual_details

if __name__ == "__main__":
    out_dir = r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test_shapes_check.xml")
    cfg = get_default_config({"drop_mode": "PARCEL", "drop_direction": "front-bottom-left", "include_paperbox": False})
    create_model(out_file, config=cfg)
    print(f"Generation Complete to {out_file}")
