# -*- coding: utf-8 -*-
"""
[WHTOOLS] Assembly Physics Analysis & Auto-Balancing Module
컴포넌트별 관성 정보를 정밀하게 측정하고, 목표 설계치(Mass, CoG, MoI)를 달성하기 위한 
보정 질량(Aux Masses)의 최적 배치를 자율적으로 결정합니다.
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich import box

# 임포트 시 순환 참조 방지를 위해 내부에서 수행하거나 최소화함
from .whtb_base import BaseDiscreteBody
from .whtb_models import (
    BPaperBox, BCushion, BOpenCellCohesive, BOpenCell, BChassis, BAuxBoxMass
)
from run_drop_simulator.whts_utils import calculate_required_aux_masses

def analyze_and_balance_components(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    [WHTOOLS] 전체 조립체의 물리적 특성을 분석하고 수치적 밸런싱을 수행합니다.
    """
    console = Console()
    
    # 1. 순수 기저 모델(Pure Base)의 관성 데이터 확보 (보조 질량 제외)
    temp_cfg = config.copy()
    temp_cfg["component_aux"] = {}
    temp_cfg["chassis_aux_masses"] = []
    
    base_info = _get_assembly_inertia_base(temp_cfg)
    m_base, c_base, i_base, _ = base_info
    
    # 2. 목표치(Target) 및 보정 파라미터 추출
    balance_cfg = config.get("components_balance", {})
    t_mass = balance_cfg.get("target_mass", config.get("target_mass", m_base))
    t_cog  = np.array(balance_cfg.get("target_cog", config.get("target_cog", c_base)))
    t_moi  = np.array(balance_cfg.get("target_inertia", config.get("target_moi", i_base)))

    # 3. 보정 질량(Aux Masses) 계산 (whts_utils의 정밀 엔진 사용)
    # 현재 config 기반으로 필요한 보조 질량 산출
    aux_masses_data = calculate_required_aux_masses(temp_cfg, t_mass, t_cog, t_moi, base_mci=(m_base, c_base, i_base))

    # 4. Config 업데이트 (Builder가 읽을 수 있도록 등록)
    config["component_aux"] = {}
    config["chassis_aux_masses"] = []
    
    for aux in aux_masses_data:
        config["component_aux"][aux["name"]] = {"pos": aux["pos"], "mass": aux["mass"], "size": aux["size"]}
        config["chassis_aux_masses"].append({"name": aux["name"], "pos": aux["pos"], "mass": aux["mass"], "size": aux["size"]})

    # 5. 결과 검증 및 상세 데이터 확보 (최종 상태 측정)
    final_info = _get_assembly_inertia_base(config)
    m_final, c_final, i_final, details = final_info

    # 6. 프리미엄 리포트 출력
    if verbose:
        _print_physics_report(console, details, m_base, c_base, i_base, t_mass, t_cog, t_moi, m_final, c_final, i_final)

    return config

def _get_assembly_inertia_base(config: Dict[str, Any]) -> Tuple[float, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    MuJoCo XML 생성 없이 순수하게 Geometry 정보를 기반으로 현재 조립체의 관성을 계산합니다.
    """
    # [WHTOOLS] 1. 기본 치수 및 파라미터 로드
    bw, bh, bd = config["box_w"], config["box_h"], config["box_d"]
    bt = config["box_thick"]
    c_gap = config["cush_gap"]
    cw, ch, cd = bw - 2*bt, bh - 2*bt, bd - 2*bt
    aw, ah = config.get("assy_w", cw-0.3), config.get("assy_h", ch-0.3)
    oc_d, occ_d, ch_d = config["opencell_d"], config["opencellcoh_d"], config["chassis_d"]
    ad = oc_d + occ_d + ch_d
    
    # 2. 루트 컨테이너 및 컴포넌트 생성 (Inertia 계산용)
    root = BaseDiscreteBody("PackagingSystem", 0,0,0, 0, [1,1,1], {})
    
    # [A] 종이 박스
    comp = config.get("components", {})
    if config.get("include_paperbox", True):
        p = comp.get("paper", {})
        b_p = BPaperBox("BPaperBox", bw, bh, bd, p.get("mass", 4.0), p.get("div", [1,1,1]), bt, {})
        b_p.build_geometry()
        root.add_child(b_p)
        
    # [B] 완충재
    if config.get("include_cushion", True):
        p = comp.get("cushion", {})
        a_bbox = [-aw/2, aw/2, -ah/2, ah/2, -ad/2, ad/2]
        cutter = {"center": [0,0,0, cw*0.5, ch*0.5, cd*2]}
        b_c = BCushion("BCushion", cw, ch, cd, p.get("mass", 2.0), p.get("div", [1,1,1]), {}, a_bbox, c_gap, cutter)
        b_c.build_geometry()
        root.add_child(b_c)
        
    # [C] 내용물 어셈블리 (Assy)
    oc_z = ad/2 - oc_d/2; occ_z = oc_z - oc_d/2 - occ_d/2; chas_z = occ_z - occ_d/2 - ch_d/2
    
    p_oc = comp.get("opencell", {}); b_oc = BOpenCell("BOpenCell", aw, ah, oc_d, p_oc.get("mass", 5.0), [1,1,1], {})
    b_oc.build_geometry(local_offset=[0,0,oc_z]); root.add_child(b_oc)
    
    p_occ = comp.get("opencellcoh", {}); b_occ = BOpenCellCohesive("BOpenCellCohesive", aw, ah, occ_d, p_occ.get("mass", 0.1), [1,1,1], config["occ_ithick"], {})
    b_occ.build_geometry(local_offset=[0,0,occ_z]); root.add_child(b_occ)
    
    p_ch = comp.get("chassis", {}); b_ch = BChassis("BChassis", aw, ah, ch_d, p_ch.get("mass", 10.0), [1,1,1], {})
    b_ch.build_geometry(local_offset=[0,0,chas_z]); root.add_child(b_ch)
    
    # [D] 보조 질량 (기존 등록분)
    for aux_name, aux_cfg in config.get("component_aux", {}).items():
        b_aux = BAuxBoxMass(aux_name, aux_cfg["size"][0], aux_cfg["size"][1], aux_cfg["size"][2], aux_cfg["mass"])
        b_aux.build_geometry(local_offset=[aux_cfg["pos"][0], aux_cfg["pos"][1], aux_cfg["pos"][2] + (chas_z if 'Inertia' in aux_name else 0)])
        root.add_child(b_aux)

    # 3. 전체 관성 합산
    return root.calculate_inertia()

def _print_physics_report(console, details, m0, c0, i0, tm, tc, ti, mf, cf, ifi):
    """Rich를 활용한 고해상도 물리 분석 리포팅"""
    console.print("\n" + "━"*105, style="dim")
    console.print(" 📦 [bold white][WHTOOLS] Assembly Physics Analysis - Component Detail[/bold white] ".center(105), style="on blue")
    
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", border_style="dim", width=105)
    table.add_column("Component", style="dim", width=15)
    table.add_column("⚖️ Mass", justify="right", width=10)
    table.add_column("🎯 CoG (x,y,z)", justify="center", width=25)
    table.add_column("🌀 Inertia (Ixx, Iyy, Izz)", justify="center", width=28)
    
    for d in details:
        name = d["name"].replace("B", "")
        # 관성이 0인 경우 (AutoBalance 등) 사용자 오해 방지를 위해 별도 표시
        moi_str = f"({d['moi'][0]:.3f}, {d['moi'][1]:.3f}, {d['moi'][2]:.3f})"
        if np.linalg.norm(d['moi']) < 1e-6:
            moi_str = "[dim](PointContribution)[/dim]"
            
        table.add_row(
            name, 
            f"{d['mass']:.3f}", 
            f"({d['cog'][0]:.3f}, {d['cog'][1]:.3f}, {d['cog'][2]:.3f})", 
            moi_str
        )
    console.print(table)
    
    console.print(" " + "📊 [bold white][WHTOOLS] Physics Balancing: Initial vs Target vs Final[/bold white] ".center(103), style="on magenta")
    res_table = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold yellow", width=105)
    res_table.add_column("Metric", width=15)
    res_table.add_column("🏗️ Initial (Base)", justify="right", width=20)
    res_table.add_column("🎯 Target (Req)", justify="right", width=20)
    res_table.add_column("🏁 Final (Balanced)", justify="right", width=20)
    res_table.add_column("Status", justify="center", width=10)
    
    def get_status(v1, v2, tol=1e-2):
        if abs(v1-v2) < tol: return "[green]✅ OK[/green]"
        return f"[yellow]⚠️ {'LIMIT' if v1 < v2 else 'SHIFT'}[/yellow]"

    res_table.add_row("Total Mass", f"{m0:.3f}", f"{tm:.3f}", f"{mf:.3f}", get_status(mf, tm))
    res_table.add_row("CoG (x, y, z)", f"({c0[0]:.2f}, {c0[1]:.2f}, {c0[2]:.2f})", "-", f"({cf[0]:.2f}, {cf[1]:.2f}, {cf[2]:.2f})", "-")
    res_table.add_row("MoI (xx,yy,zz)", f"({i0[0]:.2f}, {i0[1]:.2f}, {i0[2]:.2f})", f"({ti[0]:.2f}, {ti[1]:.2f}, {ti[2]:.2f})", f"({ifi[0]:.2f}, {ifi[1]:.2f}, {ifi[2]:.2f})", get_status(ifi[2], ti[2], 1.0))
    
    console.print(res_table)
    console.print("━"*105 + "\n", style="dim")
