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
    oc_d = config["oc_d"]; occ_d = config["occ_d"]; chas_d = config["chas_d"]
    assy_d = oc_d + occ_d + chas_d
    
    occ_ithick = config["occ_ithick"]
    
    # Z축 적층 위치 (Assy 중심 기준 상대 좌표)
    oc_z   = assy_d/2 - oc_d/2
    occ_z  = oc_z - oc_d/2 - occ_d/2
    chas_z = occ_z - occ_d/2 - chas_d/2
    
    # 필수 절단선 (가운데 구멍/패턴 유지를 위해 격자가 반드시 지나가야 할 지점)
    occ_cut_x = [-assy_w/2 + occ_ithick, assy_w/2 - occ_ithick]
    occ_cut_y = [-assy_h/2 + occ_ithick, assy_h/2 - occ_ithick]
    
    # 4. 부품별 인스턴스화 및 지오메트리 빌드
    if body_name == "BPaperBox":
        b = BPaperBox("BPaperBox", box_w, box_h, box_d, config["mass_paper"], config["box_div"], box_thick, config["mat_paper"], config["box_use_weld"])
        b.build_geometry(local_offset=[0,0,0], 
                         required_cuts_x=[-box_w/2+box_thick, box_w/2-box_thick],
                         required_cuts_y=[-box_h/2+box_thick, box_h/2-box_thick],
                         required_cuts_z=[-box_d/2+box_thick, box_d/2-box_thick])
        return b
        
    elif body_name == "BCushion":
        assy_bbox = [-assy_w/2, assy_w/2, -assy_h/2, assy_h/2, -assy_d/2, assy_d/2]
        BCushion_cutter = { "center": [0, 0, 0, cush_w*0.5, cush_h*0.5, cush_d*2] }
        b = BCushion("BCushion", cush_w, cush_h, cush_d, config["mass_cushion"], config["cush_div"], config["mat_cush"], assy_bbox, cush_gap, BCushion_cutter, config["cush_use_weld"])
        
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
        b = BOpenCell("BOpenCell", assy_w, assy_h, oc_d, config["mass_oc"], config["oc_div"], config["mat_cell"], config["oc_use_weld"])
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BOpenCellCohesive":
        b = BOpenCellCohesive("BOpenCellCohesive", assy_w, assy_h, occ_d, config["mass_occ"], config["occ_div"], occ_ithick, config["mat_tape"], config["occ_use_weld"])
        b.build_geometry(local_offset=[0,0,0], required_cuts_x=occ_cut_x, required_cuts_y=occ_cut_y)
        return b
        
    elif body_name == "BChassis":
        b = BChassis("BChassis", assy_w, assy_h, chas_d, config["mass_chassis"], config["chassis_div"], config["mat_tv"], config["chassis_use_weld"])
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
    drop_mode = config["drop_mode"]; drop_height = config["drop_height"]
    include_paperbox = config["include_paperbox"]; include_cushion = config["include_cushion"]

    # 기본 치수 파라미터
    box_w, box_h, box_d = config["box_w"], config["box_h"], config["box_d"]
    box_thick = config["box_thick"]
    cush_gap = config["cush_gap"]
    cush_w, cush_h, cush_d = box_w - 2 * box_thick, box_h - 2 * box_thick, box_d - 2 * box_thick
    assy_w = config.get("assy_w", cush_w - 0.3); assy_h = config.get("assy_h", cush_h - 0.3)
    oc_d, occ_d, chas_d = config["oc_d"], config["occ_d"], config["chas_d"]
    assy_d = oc_d + occ_d + chas_d
    occ_ithick = config["occ_ithick"]
    
    mat_paper, mat_cush, mat_tape = config["mat_paper"], config["mat_cush"], config["mat_tape"]
    mat_cell, mat_tv = config["mat_cell"], config["mat_tv"]

    # 1. 충돌 비트마스크(Collision Bitmask) 설정 로직
    bit_paper = 1; bit_cushion = 2; bit_oc = 4; bit_occ = 8; bit_chassis = 16
    all_bits = bit_paper | bit_cushion | bit_oc | bit_occ | bit_chassis
    
    mat_paper["contype"] = f"{bit_paper}"; mat_paper["conaffinity"] = f"{all_bits ^ bit_paper}"
    mat_cush["contype"]  = f"{bit_cushion}"; mat_cush["conaffinity"]  = f"{all_bits ^ bit_cushion}"
    mat_cell["contype"]  = f"{bit_oc}"; mat_cell["conaffinity"]  = f"{bit_paper | bit_cushion}"
    mat_tape["contype"]  = f"{bit_occ}"; mat_tape["conaffinity"]  = f"{bit_paper | bit_cushion}"
    mat_tv["contype"]    = f"{bit_chassis}"; mat_tv["conaffinity"]    = f"{bit_paper | bit_cushion}"

    # 충돌 마스크 요약 테이블 출력을 위한 헬퍼 (비트 분해 표시용)
    def _fmt_mask(val: str) -> str:
        v = int(val); parts = []
        for name, bit in [("Box", 1), ("Cush", 2), ("Cell", 4), ("Tape", 8), ("Chas", 16)]:
            if v & bit: parts.append(str(bit))
        return f"{v:<3} (" + "+".join(parts) + ")" if parts else f"{v:<3}"

    h_sep = "-" * 105
    logger("\n" + "="*105)
    logger(" [Collision Mask Configuration Table]")
    logger(h_sep)
    logger(f" | {'Body Name':<20} | {'conType':<12} | {'conAffinity (Bit Decomposition)':<35} | {'Notes':<20} |")
    logger(h_sep)
    logger(f" | {'BPaperBox':<20} | {f'{bit_paper} (2^0)':<12} | {_fmt_mask(mat_paper['conaffinity']):<35} | {'(Box Group)':<20} |")
    logger(f" | {'BCushion':<20} | {f'{bit_cushion} (2^1)':<12} | {_fmt_mask(mat_cush['conaffinity']):<35} | {'(Cushion Group)':<20} |")
    logger(f" | {'BOpenCell':<20} | {f'{bit_oc} (2^2)':<12} | {_fmt_mask(mat_cell['conaffinity']):<35} | {'(Internal Group)':<20} |")
    logger(f" | {'BCohesive(Tape)':<20} | {f'{bit_occ} (2^3)':<12} | {_fmt_mask(mat_tape['conaffinity']):<35} | {'(Internal Group)':<20} |")
    logger(f" | {'BChassis':<20} | {f'{bit_chassis} (2^4)':<12} | {_fmt_mask(mat_tv['conaffinity']):<35} | {'(Internal Group)':<20} |")
    logger("="*105 + "\n")

    # 2. 바디 객체 성성
    root_container = BaseDiscreteBody("PackagingBox", 0,0,0, 0, [1,1,1], {})
    
    if include_paperbox: 
        b_paper = BPaperBox("BPaperBox", box_w, box_h, box_d, config["mass_paper"], config["box_div"], box_thick, mat_paper, config["box_use_weld"])
    else: b_paper = None
    
    assy_group = BaseDiscreteBody("AssySet", 0,0,0, 0, [1,1,1], {})
    oc_z = assy_d/2 - oc_d/2; occ_z = oc_z - oc_d/2 - occ_d/2; chas_z = occ_z - occ_d/2 - chas_d/2
    b_opencell = BOpenCell("BOpenCell", assy_w, assy_h, oc_d, config["mass_oc"], config["oc_div"], mat_cell, config["oc_use_weld"])
    b_occ = BOpenCellCohesive("BOpenCellCohesive", assy_w, assy_h, occ_d, config["mass_occ"], config["occ_div"], occ_ithick, mat_tape, config["occ_use_weld"])
    b_chassis = BChassis("BChassis", assy_w, assy_h, chas_d, config["mass_chassis"], config["chassis_div"], mat_tv, config["chassis_use_weld"])

    b_cushion = None
    if include_cushion:
        assy_bbox = [-assy_w/2, assy_w/2, -assy_h/2, assy_h/2, -assy_d/2, assy_d/2]
        BCushion_cutter = { "center": [0, 0, 0, cush_w*0.5, cush_h*0.5, cush_d*2] }
        b_cushion = BCushion("BCushion", cush_w, cush_h, cush_d, config["mass_cushion"], config["cush_div"], mat_cush, assy_bbox, cush_gap, BCushion_cutter, config["cush_use_weld"])
    
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
    chassis_aux_configs = config.get("chassis_aux_masses", [])
    for i, aux_item_config in enumerate(chassis_aux_configs):
        b_aux_mass = BAuxBoxMass(name=aux_item_config.get("name", f"AuxMass_{i}"), width=aux_item_config.get("size")[0], height=aux_item_config.get("size")[1], depth=aux_item_config.get("size")[2], mass=aux_item_config.get("mass"))
        b_aux_mass.build_geometry(local_offset=[aux_item_config.get("pos")[0], aux_item_config.get("pos")[1], aux_item_config.get("pos")[2] + chas_z])
        b_chassis.add_child(b_aux_mass); aux_mass_objects.append(b_aux_mass)

    # 5. 부품 간 인터페이스 용접
    inter_weld_xml = []
    tape_solref = mat_tape.get("weld_solref", "0.02 1.0")
    tape_solimp = mat_tape.get("weld_solimp", "0.1 0.95 0.005 0.5 2")
    for (i,j,k_occ), blk_occ in b_occ.blocks.items():
        if (i,j,0) in b_opencell.blocks: 
            inter_weld_xml.append(f'        <weld site1="s_BOpenCellCohesive_{i}_{j}_{0}_PZ" site2="s_BOpenCell_{i}_{j}_{0}_NZ" solref="{tape_solref}" solimp="{tape_solimp}"/>')
        if (i,j,0) in b_chassis.blocks: 
            inter_weld_xml.append(f'        <weld site1="s_BOpenCellCohesive_{i}_{j}_{0}_NZ" site2="s_BChassis_{i}_{j}_{0}_PZ" solref="{tape_solref}" solimp="{tape_solimp}"/>')

    for b_aux_mass in aux_mass_objects:
        blk_aux = b_aux_mass.blocks[(0, 0, 0)]; min_dist_sq = float('inf'); nearest_chassis_key = None
        for block_key, blk_chassis in b_chassis.blocks.items():
            dist_sq = (blk_chassis.cx - blk_aux.cx)**2 + (blk_chassis.cy - blk_aux.cy)**2 + (blk_chassis.cz - blk_aux.cz)**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq; nearest_chassis_key = block_key
        if nearest_chassis_key:
            ci, cj, ck = nearest_chassis_key
            inter_weld_xml.append(f'        <weld body1="b_{b_aux_mass.name.lower()}_0_0_0" body2="b_{b_chassis.name.lower()}_{ci}_{cj}_{ck}" solref="0.002 1.0" solimp="0.9 0.999 0.001"/>')

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
    xml_str_io.write('  <asset>\n    <texture type="2d" name="ground_tex" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>\n    <material name="ground_mat" texture="ground_tex" texrepeat="5 5" texuniform="true" reflectance="0.0"/>\n  </asset>\n')
    xml_str_io.write('  <default>\n    <joint armature="0.05" damping="1.0"/>\n    <geom friction="0.8" solref="0.02 1.0" solimp="0.1 0.95 0.005"/>\n')
    
    for mat_name, mat_props in [("bpaperbox", mat_paper), ("bcushion", mat_cush), ("bopencell", mat_cell), ("bopencellcohesive", mat_tape), ("bchassis", mat_tv), ("bauxboxmass", mat_tv.copy())]:
        m_low = mat_name.lower()
        g_grp = 1 if "cushion" in m_low else 2 if any(x in m_low for x in ["chassis", "opencell", "adhesive", "tape", "cohesive"]) else 3 if "aux" in m_low else 0
        c_typ = mat_props.get("contype", "1"); c_aff = mat_props.get("conaffinity", "1"); c_fric = mat_props.get("friction", "0.8"); c_rgba = mat_props.get("rgba", "0.8 0.8 0.8 1")
        if "auxboxmass" in m_low: c_typ, c_aff, g_grp, c_rgba = "0", "0", 3, "1 0 0 0.4"
        
        xml_str_io.write(f'    <default class="contact_{mat_name}">\n      <geom solref="{mat_props.get("solref", "0.02 1.0")}" solimp="{mat_props.get("solimp", "0.1 0.95 0.005 0.5 2")}" contype="{c_typ}" conaffinity="{c_aff}" group="{g_grp}" friction="{c_fric}" rgba="{c_rgba}"/>\n    </default>\n')
        if "corner_solref" in mat_props: 
            xml_str_io.write(f'    <default class="contact_{mat_name}_edge">\n      <geom solref="{mat_props["corner_solref"]}" solimp="{mat_props.get("corner_solimp", "0.1 0.95 0.005 0.5 2")}" contype="{c_typ}" conaffinity="{c_aff}" group="{g_grp}" friction="{c_fric}" rgba="{c_rgba}"/>\n    </default>\n')
    
    xml_str_io.write('  </default>\n')
    xml_str_io.write(f'  <worldbody>\n    <camera name="iso_view" pos="4.5 -4.5 3.5" xyaxes="1 1 0 -0.4 0.4 1" mode="fixed"/>\n    <light pos="0 0 6" dir="0 0 -1" directional="false" diffuse="{config.get("light_main_diffuse")}" ambient="{config.get("light_main_ambient")}" castshadow="false"/>\n    <light pos="3 3 5" dir="-1 -1 -1" directional="false" diffuse="{config.get("light_sub_diffuse")}" castshadow="false"/>\n')
    xml_str_io.write(f'    <geom name="ground" type="plane" size="10 10 0.1" friction="{config.get("ground_friction")}" contype="1" conaffinity="1" group="0" material="ground_mat" solref="{config.get("ground_solref")}" solimp="{config.get("ground_solimp")}"/>\n')
    xml_str_io.write(f'    <body name="BPackagingBox" pos="{wx:.5f} {wy:.5f} {wz:.5f}" axisangle="{rot_str}">\n      <freejoint/>\n      <geom type="box" size="0.001 0.001 0.001" mass="0.000021" rgba="0 0 0 0" contype="0" conaffinity="0" friction="0.8"/>\n')
    for line in root_container.get_worldbody_xml_strings(indent_level=3): xml_str_io.write(line + "\n")
    xml_str_io.write('    </body>\n  </worldbody>\n  <equality>\n')
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
