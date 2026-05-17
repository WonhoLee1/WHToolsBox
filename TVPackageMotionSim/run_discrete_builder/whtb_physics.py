# -*- coding: utf-8 -*-
"""
[WHTOOLS] Assembly Physics Analysis & Auto-Balancing Module
컴포넌트별 관성 정보를 정밀하게 측정하고, 목표 설계치(Mass, CoG, MoI)를 달성하기 위한 
보정 질량(Aux Masses)의 최적 배치를 자율적으로 결정합니다.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from rich.console import Console
from rich.table import Table
from rich import box

from .whtb_base import BaseDiscreteBody
from .whtb_models import (
    BPaperBox, BCushion, BOpenCellCohesive, BOpenCell, BChassis, BAuxBoxMass
)

def analyze_and_balance_components(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    [WHTOOLS] 전체 조립체의 물리적 특성을 분석합니다.
    inertia_correction이 config에 있으면 delta-inertia 방식으로 보정합니다.
    (물리적 aux mass 배치는 더 이상 수행하지 않습니다.)
    """
    console = Console()

    base_info = _get_assembly_inertia_base(config)
    m_base, c_base, i_base, details = base_info

    balance_cfg = config.get("components_balance", {})
    t_mass = balance_cfg.get("target_mass", config.get("target_mass", m_base))
    t_cog  = np.array(balance_cfg.get("target_cog", config.get("target_cog", c_base)))
    t_moi  = np.array(balance_cfg.get("target_inertia", config.get("target_moi", i_base)))

    # inertia_correction이 없으면 components_balance 타겟으로부터 자동 계산
    ic = config.get("inertia_correction")
    has_targets = bool(config.get("components_balance"))
    if ic is None and has_targets:
        m_delta = t_mass - m_base
        if abs(m_delta) < 1e-9:
            pos_delta = t_cog.copy()
        else:
            pos_delta = (t_cog * t_mass - c_base * m_base) / m_delta
        d = t_cog - c_base
        i_base_at_tcog = np.zeros(6)
        i_base_at_tcog[:3] = i_base[:3] + m_base * np.array([d[1]**2+d[2]**2, d[0]**2+d[2]**2, d[0]**2+d[1]**2])
        i_base_at_tcog[3:6] = i_base[3:6] - m_base * np.array([d[0]*d[1], d[0]*d[2], d[1]*d[2]])
        i_delta = t_moi - i_base_at_tcog
        i_delta = _clamp_inertia_triangle(i_delta, label="I_delta")
        ic = {
            "m_delta": float(m_delta),
            "pos_delta": [float(v) for v in pos_delta],
            "I_delta": [float(v) for v in i_delta],
        }
        config["inertia_correction"] = ic

    if ic and abs(ic.get("m_delta", 0.0)) > 1e-9:
        m_final = m_base + ic["m_delta"]
        pos_d = np.array(ic["pos_delta"])
        c_final = (m_base * c_base + ic["m_delta"] * pos_d) / m_final
        d = c_final - c_base
        i_base_moved = np.zeros(6)
        i_base_moved[:3] = i_base[:3] + m_base * np.array([d[1]**2+d[2]**2, d[0]**2+d[2]**2, d[0]**2+d[1]**2])
        i_base_moved[3:6] = i_base[3:6] - m_base * np.array([d[0]*d[1], d[0]*d[2], d[1]*d[2]])
        dp = pos_d - c_final
        m_d = ic["m_delta"]
        i_delta_at_cfinal = np.array(ic["I_delta"])
        i_delta_at_cfinal[0] += m_d * (dp[1]**2 + dp[2]**2)
        i_delta_at_cfinal[1] += m_d * (dp[0]**2 + dp[2]**2)
        i_delta_at_cfinal[2] += m_d * (dp[0]**2 + dp[1]**2)
        i_delta_at_cfinal[3] -= m_d * dp[0] * dp[1]
        i_delta_at_cfinal[4] -= m_d * dp[0] * dp[2]
        i_delta_at_cfinal[5] -= m_d * dp[1] * dp[2]
        i_final = i_base_moved + i_delta_at_cfinal
    else:
        m_final, c_final, i_final = m_base, c_base, i_base

    if verbose:
        _print_physics_report(console, details, m_base, c_base, i_base, t_mass, t_cog, t_moi, m_final, c_final, i_final, ic=ic)

    return config

def _clamp_inertia_triangle(i6: np.ndarray, label: str = "") -> np.ndarray:
    """
    MuJoCo 삼각 부등식 A + B >= C 를 만족하도록 대각 성분을 최소한으로 올립니다.
    off-diagonal(product) 성분은 그대로 유지합니다.
    """
    i = i6.copy()
    ixx, iyy, izz = i[0], i[1], i[2]
    eps = 1e-9
    # 세 쌍 모두 검사 후 부족분 균등 분배
    violations = []
    if ixx + iyy < izz - eps:
        violations.append(f"Ixx+Iyy({ixx:.4f}+{iyy:.4f}={ixx+iyy:.4f}) < Izz({izz:.4f})")
        deficit = izz - (ixx + iyy)
        ixx += deficit / 2; iyy += deficit / 2
    if ixx + izz < iyy - eps:
        violations.append(f"Ixx+Izz({ixx:.4f}+{izz:.4f}={ixx+izz:.4f}) < Iyy({iyy:.4f})")
        deficit = iyy - (ixx + izz)
        ixx += deficit / 2; izz += deficit / 2
    if iyy + izz < ixx - eps:
        violations.append(f"Iyy+Izz({iyy:.4f}+{izz:.4f}={iyy+izz:.4f}) < Ixx({ixx:.4f})")
        deficit = ixx - (iyy + izz)
        iyy += deficit / 2; izz += deficit / 2
    if violations:
        tag = f"[{label}] " if label else ""
        print(f"  ⚠  {tag}Inertia triangle violation — diagonal clamped to satisfy A+B≥C:")
        for v in violations:
            print(f"     {v}")
        print(f"     → clamped to Ixx={ixx:.6f}  Iyy={iyy:.6f}  Izz={izz:.6f}")
    i[0], i[1], i[2] = ixx, iyy, izz
    return i

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

def _print_physics_report(console, details, m0, c0, i0, tm, tc, ti, mf, cf, ifi, ic=None):
    """Rich를 활용한 고해상도 물리 분석 리포팅"""
    W = 112
    console.print("\n" + "━"*W, style="dim")
    console.print(" 📦 [bold white][WHTOOLS] Assembly Physics Analysis — Component Detail[/bold white] ".center(W), style="on blue")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", border_style="dim", width=W)
    table.add_column("Component", style="dim", width=16)
    table.add_column("⚖️ Mass (kg)", justify="right", width=12)
    table.add_column("🎯 CoG (x, y, z)  m", justify="center", width=28)
    table.add_column("🌀 MoI  Diag (xx,yy,zz)  |  Prod (xy,xz,yz)  kg·m²", justify="center", width=50)

    for d in details:
        name = d["name"].replace("B", "", 1)
        if np.linalg.norm(d['moi']) < 1e-6:
            moi_str = "[dim](point mass)[/dim]"
        elif len(d['moi']) >= 6:
            moi_str = (f"({d['moi'][0]:.4f}, {d['moi'][1]:.4f}, {d['moi'][2]:.4f})"
                       f"  |  ({d['moi'][3]:.4f}, {d['moi'][4]:.4f}, {d['moi'][5]:.4f})")
        else:
            moi_str = f"({d['moi'][0]:.4f}, {d['moi'][1]:.4f}, {d['moi'][2]:.4f})"
        table.add_row(name, f"{d['mass']:.4f}",
                      f"({d['cog'][0]:.4f}, {d['cog'][1]:.4f}, {d['cog'][2]:.4f})", moi_str)
    console.print(table)

    # ── Inertia Correction block ──────────────────────────────────────────────
    if ic and abs(ic.get("m_delta", 0.0)) > 1e-9:
        console.print(" ⚡ [bold white][WHTOOLS] Inertia Correction (Delta-Inertia Virtual Body)[/bold white] ".center(W), style="on dark_green")
        ic_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold green", width=W)
        ic_table.add_column("Property", width=18)
        ic_table.add_column("Value", justify="left", width=70)
        pd = ic["pos_delta"]; Id = ic["I_delta"]
        ic_table.add_row("m_delta (kg)",   f"[bold]{ic['m_delta']:+.6f}[/bold]")
        ic_table.add_row("pos_delta (m)",  f"({pd[0]:.5f}, {pd[1]:.5f}, {pd[2]:.5f})")
        ic_table.add_row("I_delta diag",   f"Ixx={Id[0]:+.6f}  Iyy={Id[1]:+.6f}  Izz={Id[2]:+.6f}")
        ic_table.add_row("I_delta prod",   f"Ixy={Id[3]:+.6f}  Ixz={Id[4]:+.6f}  Iyz={Id[5]:+.6f}")
        neg_diag = [["Ixx","Iyy","Izz"][k] for k in range(3) if Id[k] < 0]
        feas = "[green]✅ Physically valid[/green]" if not neg_diag else f"[yellow]⚠️  Negative diagonal: {', '.join(neg_diag)} — MuJoCo may reject[/yellow]"
        ic_table.add_row("Feasibility", feas)
        console.print(ic_table)

    # ── Summary: Base → Target → Final ───────────────────────────────────────
    console.print(" 📊 [bold white][WHTOOLS] Physics Summary: Base → Target → Final (after correction)[/bold white] ".center(W), style="on magenta")
    res_table = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold yellow", width=W)
    res_table.add_column("Metric",             width=18)
    res_table.add_column("🏗️ Base",            justify="right", width=26)
    res_table.add_column("🎯 Target",           justify="right", width=26)
    res_table.add_column("🏁 Final",            justify="right", width=26)
    res_table.add_column("Status",              justify="center", width=12)

    # Mass
    mass_rel_err = abs(mf - tm) / (tm + 1e-9)
    mass_status = "[green]✅ Exact[/green]" if mass_rel_err < 1e-6 else (
        "[green]✅ OK[/green]" if mass_rel_err < 0.005 else f"[yellow]⚠️ {mass_rel_err*100:.2f}%[/yellow]")
    res_table.add_row("Total Mass (kg)", f"{m0:.4f}", f"{tm:.4f}", f"{mf:.4f}", mass_status)

    # CoG
    cog_err_mm = np.linalg.norm(np.array(cf) - np.array(tc)) * 1000
    cog_status = "[green]✅ Exact[/green]" if cog_err_mm < 0.01 else (
        "[green]✅ OK[/green]" if cog_err_mm < 10.0 else f"[yellow]⚠️ {cog_err_mm:.2f} mm[/yellow]")
    res_table.add_row("CoG (x,y,z) m",
                      f"({c0[0]:.4f}, {c0[1]:.4f}, {c0[2]:.4f})",
                      f"({tc[0]:.4f}, {tc[1]:.4f}, {tc[2]:.4f})",
                      f"({cf[0]:.4f}, {cf[1]:.4f}, {cf[2]:.4f})", cog_status)

    # MoI Diagonal
    moi_scale = max(float(np.abs(np.array(ti)[:3]).mean()), 0.1)
    moi_rel = float(np.linalg.norm((np.array(ifi)[:3] - np.array(ti)[:3]) / moi_scale)) / np.sqrt(3)
    moi_status = "[green]✅ Exact[/green]" if moi_rel < 1e-6 else (
        "[green]✅ OK[/green]" if moi_rel < 0.05 else f"[yellow]⚠️ {moi_rel*100:.1f}%[/yellow]")
    i0d = f"({i0[0]:.4f}, {i0[1]:.4f}, {i0[2]:.4f})"
    tid = f"({ti[0]:.4f}, {ti[1]:.4f}, {ti[2]:.4f})"
    ifd = f"({ifi[0]:.4f}, {ifi[1]:.4f}, {ifi[2]:.4f})"
    res_table.add_row("MoI Diag  kg·m²", i0d, tid, ifd, moi_status)

    # MoI Product
    i0p = i0[3:6] if len(i0) >= 6 else [0,0,0]
    tip = ti[3:6] if len(ti) >= 6 else [0,0,0]
    ifp = ifi[3:6] if len(ifi) >= 6 else [0,0,0]
    prod_err = float(np.linalg.norm(np.array(ifp) - np.array(tip)))
    prod_status = "[green]✅ Exact[/green]" if prod_err < 1e-6 else (
        "[green]✅ OK[/green]" if prod_err < 0.01 else f"[yellow]⚠️ Δ={prod_err:.4f}[/yellow]")
    res_table.add_row("MoI Prod  kg·m²",
                      f"({i0p[0]:.4f}, {i0p[1]:.4f}, {i0p[2]:.4f})",
                      f"({tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f})",
                      f"({ifp[0]:.4f}, {ifp[1]:.4f}, {ifp[2]:.4f})", prod_status)

    console.print(res_table)
    console.print("━"*W + "\n", style="dim")
