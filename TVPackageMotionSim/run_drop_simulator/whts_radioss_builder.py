# -*- coding: utf-8 -*-
"""
OpenRadioss model builder for WHToolsBox TV package drop simulation.

Units  : mm / kg / ms  (Radioss crash standard)
Parts  :
  1 - Box        (paper box outer shell)
  2 - CushFront  (EPS foam, front face)
  3 - CushBack   (EPS foam, rear face)
  4 - OpenCell   (open-cell foam)
  5 - Chassis    (TV chassis shell)
  6 - Ground     (floor slab, 10 mm thick, fixed)

Transform modes:
  'parts'  - package parts get /TRANSFORM ROT+TRA, ground stays at Z=0
  'ground' - ground gets inverse transform, parts at origin
"""

import math
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False


EXEC_DIR    = Path(r"D:\OpenRadioss_win64\OpenRadioss\exec")
STARTER_EXE = EXEC_DIR / "starter_win64.exe"
ENGINE_EXE  = EXEC_DIR / "engine_win64.exe"

MM  = 1000.0    # metres -> mm
G   = 9806.0    # gravity mm/s²  (tonne / mm / s unit system)

SEP = "#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|"


# ── formatting helpers ────────────────────────────────────────────────────────

def _f20(v: float) -> str:
    return f"{float(v):>20g}"


def _i10(v: int) -> str:
    return f"{int(v):>10d}"


# ── mesh data container ───────────────────────────────────────────────────────

class _MeshData:
    def __init__(self, name: str, part_id: int):
        self.name     = name
        self.part_id  = part_id
        self.nodes: Dict[int, Tuple[float, float, float]] = {}
        self.quads: Dict[int, Tuple[int, ...]] = {}   # SHELL (quad4)
        self.hexas: Dict[int, Tuple[int, ...]] = {}   # BRICK (hex8)
        self.tetras: Dict[int, Tuple[int, ...]] = {}  # TETRA (tetra4)
        self.is_shell  = False
        self.thickness = 1.0   # mm, only for shells
        self.top_face_segs: List[Tuple[int, ...]] = []  # (seg_id, n1, n2, n3, n4) for top face
        self.ref_node_id: int = 0


# ── main builder ─────────────────────────────────────────────────────────────

class RadiossModelBuilder:
    """Generate OpenRadioss FEM model from WHToolsBox simulation config."""

    def __init__(self,
                 config: Dict[str, Any],
                 output_dir: Path,
                 R_mat: Optional[np.ndarray] = None,
                 t_vec: Optional[np.ndarray] = None,
                 v_vec: Optional[np.ndarray] = None,
                 omega_vec: Optional[np.ndarray] = None,
                 transform_mode: str = 'parts',
                 drop_height_m: float = 0.5,
                 model_name: str = "TVDrop"):
        self.cfg      = config
        self.out      = Path(output_dir)
        self.R        = R_mat.copy() if R_mat is not None else np.eye(3)
        # t_vec in metres → convert to mm
        self.t        = (t_vec.copy() * 1000.0) if t_vec is not None else np.zeros(3)
        self.v_vec    = v_vec.copy() if v_vec is not None else np.zeros(3)
        self.omega_vec = omega_vec.copy() if omega_vec is not None else np.zeros(3)
        self.mode     = transform_mode
        self.h        = float(drop_height_m)
        self.name     = model_name

        self._nid_base = 0   # running global node ID base
        self._eid_base = 0   # running global element ID base
        self._seg_id_base = 0  # running surface segment ID base
        self._parts: List[_MeshData] = []

    # ── public ───────────────────────────────────────────────────────────────

    def build(self) -> Path:
        """Mesh all parts, write .inc files and _0000.rad / _0001.rad."""
        if not HAS_GMSH:
            raise ImportError("gmsh is required — install with: pip install gmsh")

        self.out.mkdir(parents=True, exist_ok=True)
        self._parts = self._create_all_parts()

        for p in self._parts:
            self._write_inc(p)

        starter = self._write_starter()
        self._write_engine()
        self._write_lsdyna()
        return starter

    def run(self, nt: int = 4, np_cores: int = 1, callback=None) -> None:
        """Build model files then launch starter + engine via RunOpenRadioss."""
        self.build()
        import sys as _sys
        import os
        import subprocess
        from .whts_utils import get_external_tool_path
        
        # runopenradioss.py is located in the openradioss_gui directory
        openradioss_gui_dir = get_external_tool_path('openradioss_gui_dir')
        if openradioss_gui_dir and os.path.exists(openradioss_gui_dir):
            _gui_dir = str(openradioss_gui_dir)
        else:
            _gui_dir = str(Path(r"D:\OpenRadioss_win64\OpenRadioss\openradioss_gui"))

        starter_file = str(self.out.resolve() / f"{self.name}_0000.rad")
        command = [
            starter_file,   # [0] input file
            str(nt),        # [1] OpenMP threads
            str(np_cores),  # [2] MPI processes
            'dp',           # [3] precision (double)
            'no',           # [4] anim_to_vtk
            'yes',          # [5] th_to_csv
            'no',           # [6] starter_only
            'no',           # [7] anim_to_d2plot
            'yes',          # [8] anim_to_vtkhdf
            '',             # [9] mpi_path
        ]
        
        openradioss_dir = get_external_tool_path('openradioss_dir')
        if not (openradioss_dir and os.path.exists(openradioss_dir)):
            openradioss_dir = _gui_dir

        # We must run this in a separate process because RunOpenRadioss uses signal.signal()
        # which throws ValueError if executed inside a QThread (not main thread).
        # We write a tiny wrapper script and execute it via python.
        script = f"""
import sys
sys.path.insert(0, {_gui_dir!r})
import runopenradioss
runner = runopenradioss.RunOpenRadioss({command!r}, debug=1)
runner.openradioss_path = {openradioss_dir!r}
runner.batch_run()
"""
        
        try:
            # Popen with stdout/stderr pipe so we can stream the output
            proc = subprocess.Popen([_sys.executable, "-c", script], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.STDOUT, 
                                    text=True,
                                    encoding='utf-8',
                                    errors='replace')
            
            for line in iter(proc.stdout.readline, ''):
                line = line.strip()
                if line:
                    if callback:
                        callback(f"[Radioss] {line}")
                    else:
                        print(f"[Radioss] {line}")
                    
            proc.stdout.close()
            ret_code = proc.wait()
            if ret_code != 0:
                print(f"[Radioss] Engine failed with exit code {ret_code}")
                
        except Exception:
            import traceback
            print(traceback.format_exc())

    # ── part geometry setup ───────────────────────────────────────────────────

    def _create_all_parts(self) -> List[_MeshData]:
        mm  = 1000.0
        cfg = self.cfg

        bw = cfg.get('box_w',    1.841) * mm
        bh = cfg.get('box_h',    1.103) * mm
        bd = cfg.get('box_d',    0.170) * mm
        bt = cfg.get('box_thick', 0.008) * mm

        aw     = cfg.get('assy_w',     1.670) * mm
        ah     = cfg.get('assy_h',     0.960) * mm
        chas_d = cfg.get('chassis_d',  0.035) * mm
        oc_d   = cfg.get('opencell_d', 0.012) * mm

        cush_d = max((bd - chas_d - oc_d) / 2.0, 5.0)

        gw = bw * 2.5
        gh = bh * 2.5
        gt = 10.0   # ground thickness mm

        # Z layout (bottom → top, ground top = Z=0)
        z_ground_bot  = -gt
        
        # Align with MuJoCo coordinate system where assembly center is Z=0
        assy_d = oc_d + chas_d
        oc_z   = assy_d/2 - oc_d/2
        chas_z = -assy_d/2 + chas_d/2

        parts = []
        # Move geometric Z center of Box and Cushion to 0.0 to match MuJoCo pivot
        parts.append(self._mesh_shell_closed_box(
            1, "Box",       bw + bt, bh + bt, bd + bt, bt,
            cx=0, cy=0, cz=0.0,                elem_size=30.0))
        
        cush_elem_size = cfg.get("components", {}).get("cushion", {}).get("mesh_elem_size", 20.0)
        parts.append(self._mesh_hollow_solid(
            2, "Cushion",
            outer_dx=bw, outer_dy=bh, outer_dz=bd,
            inner_dx=aw, inner_dy=ah, inner_dz=chas_d + oc_d,
            cx=0, cy=0, cz=0.0, elem_size=cush_elem_size))
            
        parts.append(self._mesh_solid(
            4, "OpenCell",  aw, ah, oc_d,
            cx=0, cy=0, cz=oc_z,   elem_size=15.0))
        parts.append(self._mesh_solid(
            5, "Chassis",   aw, ah, chas_d,
            cx=0, cy=0, cz=chas_z, elem_size=30.0))
        ground = self._mesh_solid(
            6, "Ground",    gw, gh, gt,
            cx=0, cy=0, cz=z_ground_bot +gt/2,     elem_size=60.0)
        self._extract_top_face_segs(ground, z_top=0.0)
        ground.ref_node_id = min(ground.nodes.keys())
        parts.append(ground)
        return parts

    # ── gmsh meshing ──────────────────────────────────────────────────────────

    def _mesh_hollow_solid(self, part_id: int, name: str,
                           outer_dx: float, outer_dy: float, outer_dz: float,
                           inner_dx: float, inner_dy: float, inner_dz: float,
                           cx: float, cy: float, cz: float,
                           elem_size: float) -> _MeshData:
        md = _MeshData(name, part_id)
        md.is_shell = False

        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)

        xo, yo, zo = cx - outer_dx/2, cy - outer_dy/2, cz - outer_dz/2
        outer_tag = gmsh.model.occ.addBox(xo, yo, zo, outer_dx, outer_dy, outer_dz)

        xi, yi, zi = cx - inner_dx/2, cy - inner_dy/2, cz - inner_dz/2
        inner_tag = gmsh.model.occ.addBox(xi, yi, zi, inner_dx, inner_dy, inner_dz)

        gmsh.model.occ.cut([(3, outer_tag)], [(3, inner_tag)])
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", elem_size * 0.8)
        gmsh.option.setNumber("Mesh.MeshSizeMax", elem_size * 1.2)
        gmsh.model.mesh.generate(3)

        self._extract_nodes_tetras(md)
        gmsh.finalize()
        return md

    def _mesh_solid(self, part_id: int, name: str,
                    dx: float, dy: float, dz: float,
                    cx: float, cy: float, cz: float,
                    elem_size: float) -> _MeshData:
        md = _MeshData(name, part_id)
        md.is_shell = False

        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)

        x0, y0, z0 = cx - dx/2, cy - dy/2, cz - dz/2
        tag = gmsh.model.occ.addBox(x0, y0, z0, dx, dy, dz)
        gmsh.model.occ.synchronize()

        # Structured hex mesh via transfinite
        nx = max(2, round(dx / elem_size) + 1)
        ny = max(2, round(dy / elem_size) + 1)
        nz = max(2, round(dz / elem_size) + 1)
        curves = gmsh.model.getBoundary([(3, tag)], oriented=False, recursive=True)
        bnd_surfs = gmsh.model.getBoundary([(3, tag)], oriented=False)
        for dim, ctag in curves:
            lo = gmsh.model.getBoundingBox(dim, ctag)
            length = math.sqrt(sum((lo[i+3]-lo[i])**2 for i in range(3)))
            divs = max(2, round(length / elem_size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(ctag, divs)
        for _, stag in bnd_surfs:
            gmsh.model.mesh.setTransfiniteSurface(abs(stag))
        gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.recombine()

        self._extract_nodes_hexas(md)
        gmsh.finalize()
        return md

    def _mesh_shell_plate(self, part_id: int, name: str,
                          dx: float, dy: float, dz: float,
                          cx: float, cy: float, cz: float,
                          elem_size: float) -> _MeshData:
        """Mid-plane shell for a thin plate (thickness = dz)."""
        md = _MeshData(name, part_id)
        md.is_shell   = True
        md.thickness  = dz

        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)

        x0, y0 = cx - dx/2, cy - dy/2
        tag = gmsh.model.occ.addRectangle(x0, y0, cz, dx, dy)
        gmsh.model.occ.synchronize()

        curves = gmsh.model.getBoundary([(2, tag)], oriented=False, recursive=False)
        for _, ctag in curves:
            lo = gmsh.model.getBoundingBox(1, ctag)
            length = math.sqrt(sum((lo[i+3]-lo[i])**2 for i in range(3)))
            divs = max(2, round(length / elem_size) + 1)
            gmsh.model.mesh.setTransfiniteCurve(ctag, divs)
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()

        self._extract_nodes_quads(md)
        gmsh.finalize()
        return md

    def _mesh_shell_closed_box(self, part_id: int, name: str,
                                dx: float, dy: float, dz: float,
                                thickness: float,
                                cx: float, cy: float, cz: float,
                                elem_size: float) -> _MeshData:
        """6-face closed shell box (paper box outer shell)."""
        md = _MeshData(name, part_id)
        md.is_shell   = True
        md.thickness  = thickness

        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)

        x0, y0, z0 = cx - dx/2, cy - dy/2, cz - dz/2
        gmsh.model.occ.addBox(x0, y0, z0, dx, dy, dz)
        gmsh.model.occ.synchronize()

        surfs = gmsh.model.getEntities(2)
        for _, stag in surfs:
            curves = gmsh.model.getBoundary([(2, stag)], oriented=False, recursive=False)
            for _, ctag in curves:
                lo = gmsh.model.getBoundingBox(1, ctag)
                length = math.sqrt(sum((lo[i+3]-lo[i])**2 for i in range(3)))
                divs = max(2, round(length / elem_size) + 1)
                try:
                    gmsh.model.mesh.setTransfiniteCurve(ctag, divs)
                except Exception:
                    pass
            try:
                gmsh.model.mesh.setTransfiniteSurface(stag)
            except Exception:
                pass

        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()

        self._extract_nodes_quads(md)
        gmsh.finalize()
        return md

    # ── gmsh extraction helpers ───────────────────────────────────────────────

    def _extract_nodes_tetras(self, md: _MeshData) -> None:
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        for tag, x, y, z in zip(node_tags, coords[::3], coords[1::3], coords[2::3]):
            md.nodes[int(tag)] = (float(x), float(y), float(z))

        etypes, _, enodes_list = gmsh.model.mesh.getElements(dim=3)
        eid = 0
        for etype, enodes in zip(etypes, enodes_list):
            if etype != 4:   # tetra4 only
                continue
            for i in range(0, len(enodes), 4):
                eid += 1
                md.tetras[eid] = tuple(int(n) for n in enodes[i:i+4])

    def _extract_nodes_hexas(self, md: _MeshData) -> None:
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        for tag, x, y, z in zip(node_tags, coords[::3], coords[1::3], coords[2::3]):
            md.nodes[int(tag)] = (float(x), float(y), float(z))

        etypes, _, enodes_list = gmsh.model.mesh.getElements(dim=3)
        eid = 0
        for etype, enodes in zip(etypes, enodes_list):
            if etype != 5:   # hex8 only
                continue
            for i in range(0, len(enodes), 8):
                eid += 1
                md.hexas[eid] = tuple(int(n) for n in enodes[i:i+8])

    def _extract_top_face_segs(self, md: _MeshData, z_top: float, tol: float = 0.5) -> None:
        """Populate md.top_face_segs with quad faces of hex elements at z≈z_top."""
        HEX_FACES = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
        seg_id = self._seg_id_base
        for eid, nids in md.hexas.items():
            for fi in HEX_FACES:
                face = [nids[i] for i in fi]
                if all(abs(md.nodes[n][2] - z_top) < tol for n in face):
                    seg_id += 1
                    md.top_face_segs.append((seg_id, *face))
        self._seg_id_base = seg_id

    def _extract_nodes_quads(self, md: _MeshData) -> None:
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        for tag, x, y, z in zip(node_tags, coords[::3], coords[1::3], coords[2::3]):
            md.nodes[int(tag)] = (float(x), float(y), float(z))

        etypes, _, enodes_list = gmsh.model.mesh.getElements(dim=2)
        eid = 0
        for etype, enodes in zip(etypes, enodes_list):
            if etype != 3:   # quad4 only
                continue
            for i in range(0, len(enodes), 4):
                eid += 1
                md.quads[eid] = tuple(int(n) for n in enodes[i:i+4])

    # ── file writers ──────────────────────────────────────────────────────────

    def _append_part_inline(self, md: _MeshData, L: list) -> None:
        """Append part mesh data (PART/NODE/SHELL|BRICK) directly into a line list."""
        L += [
            SEP,
            f"# Part {md.part_id}: {md.name}",
            f"/PART/{md.part_id}",
            md.name,
            f"{_i10(md.part_id)}{_i10(md.part_id)}{_i10(0)}",
            SEP,
            "/NODE",
        ]
        for nid, (x, y, z) in sorted(md.nodes.items()):
            L.append(f"{_i10(nid)}{_f20(x)}{_f20(y)}{_f20(z)}")
        L.append(SEP)
        if md.is_shell and md.quads:
            L.append(f"/SHELL/{md.part_id}")
            for eid, ns in sorted(md.quads.items()):
                L.append(_i10(eid) + "".join(_i10(n) for n in ns))
        elif (not md.is_shell) and md.hexas:
            L.append(f"/BRICK/{md.part_id}")
            for eid, ns in sorted(md.hexas.items()):
                L.append(_i10(eid) + "".join(_i10(n) for n in ns))
        elif (not md.is_shell) and md.tetras:
            L.append(f"/TETRA4/{md.part_id}")
            for eid, ns in sorted(md.tetras.items()):
                L.append(_i10(eid) + "".join(_i10(n) for n in ns))
        L.append(SEP)

    def _write_inc(self, md: _MeshData) -> None:
        """Write part data to a standalone .inc file (reference copy only)."""
        path = self.out / f"{self.name}_part{md.part_id}_{md.name}.inc"
        L: list = []
        self._append_part_inline(md, L)
        path.write_text("\n".join(L), encoding="utf-8")
        print(f"[Radioss] inc → {path.name}  "
              f"(nodes={len(md.nodes)}, "
              f"{'shells' if md.is_shell else 'bricks'}={len(md.quads) or len(md.hexas)})")

    def _write_starter(self) -> Path:
        path = self.out / f"{self.name}_0000.rad"
        cfg  = self.cfg
        mm   = 1000.0

        # Dimensions in mm (recompute for material densities)
        bw     = cfg.get('box_w',     1.841) * mm
        bh     = cfg.get('box_h',     1.103) * mm
        bd     = cfg.get('box_d',     0.170) * mm
        bt     = cfg.get('box_thick', 0.008) * mm
        aw     = cfg.get('assy_w',    1.670) * mm
        ah     = cfg.get('assy_h',    0.960) * mm
        chas_d = cfg.get('chassis_d', 0.035) * mm
        oc_d   = cfg.get('opencell_d',0.012) * mm
        cush_d = max((bd - chas_d - oc_d) / 2.0, 5.0)
        gt     = 10.0

        def _vol(w, h, d):
            return w * h * d   # mm³  (unit system: tonne / mm / s)

        # Densities in tonne/mm³  (mass_* config values are in kg → ÷1000 → tonne)
        sa_box   = 2 * (bw*bh + bh*bd + bw*bd)  # mm²
        vol_box  = sa_box * bt                    # mm³
        rho_box  = cfg.get('mass_paper', 4.0) / 1000.0 / vol_box if vol_box > 0 else 1e-9

        vol_cush = _vol(aw, ah, cush_d * 2)       # mm³ (front + back combined)
        rho_cush = cfg.get('mass_cushion', 2.0) / 1000.0 / vol_cush if vol_cush > 0 else 3e-11

        vol_oc   = _vol(aw, ah, oc_d)             # mm³
        rho_oc   = cfg.get('mass_oc', 5.0) / 1000.0 / vol_oc if vol_oc > 0 else 2e-11

        vol_ch   = _vol(aw, ah, chas_d)           # mm³
        rho_ch   = cfg.get('mass_chassis', 10.0) / 1000.0 / vol_ch if vol_ch > 0 else 2.7e-9

        rho_gnd  = 7.8e-9   # steel tonne/mm³

        # Initial velocity (drop): v0 = sqrt(2*g*h)  [mm/s]
        h_mm = self.h * mm
        v0   = math.sqrt(max(2.0 * G * h_mm, 0.0))   # mm/s

        L = []

        def sec(title: str):
            L.extend([SEP, f"#- {title}:", SEP])

        # ── Header ──
        # /BEGIN needs: title, version+icheck, WORK unit line, INPUT unit line (both identical)
        unit_line = f"{'Mg':>20}{'mm':>20}{'s':>20}"
        L += [
            "#RADIOSS STARTER",
            f"# Generated by WHToolsBox RadiossModelBuilder",
            SEP,
            "/BEGIN",
            f"{self.name:<80}",
            "      2022         0",
            unit_line,
            unit_line,
        ]

        # Title is embedded in /BEGIN block — no separate /TITLE card needed

        # ── Materials ──
        sec("MATERIALS")

        E_box = cfg.get('E_paper', 3000.0)
        E_cush = cfg.get('E_cushion', 0.5)
        E_oc = cfg.get('E_oc', 0.08)
        E_ch = cfg.get('E_chassis', 10000.0)

        # MAT 1: Cardboard (box) — elastic
        # rho in tonne/mm³
        L += [
            "/MAT/ELAST/1", "Cardboard_Box",
            "#        Init. dens.",
            f"{_f20(rho_box)}",
            "#                    E                  Nu",
            f"{_f20(E_box)}{_f20(0.30)}",
        ]
        # MAT 2: EPS foam cushion — foam_plas (LAW33)
        for mid, rho in [(2, rho_cush)]:
            L += [
                f"/MAT/FOAM_PLAS/{mid}", f"EPS_Foam_Cushion_{mid}",
                "#        Init. dens.          Ref. dens.",
                f"{_f20(rho)}{_f20(0)}",
                "#                  E                  Ka                  If",
                f"{_f20(E_cush)}{_f20(0)}{_f20(0)}",
                "#                 P0                 Phi             Gamma_0",
                f"{_f20(1e-4)}{_f20(0.1)}{_f20(0)}",
                "#                  C1                  C2                  C3                  C4                  C5",
                f"{_f20(-1.77e-4)}{_f20(6.77e-4)}{_f20(-3.34)}{_f20(0)}{_f20(0)}",
                "#          fct_IDd           fct_IDs            Fscale_d            Fscale_s",
                f"{_i10(0)}{_i10(0)}{_f20(1.0)}{_f20(1.0)}",
            ]
        # MAT 4: Open-cell foam (LAW33, softer)
        L += [
            "/MAT/FOAM_PLAS/4", "OpenCell_Foam",
            "#        Init. dens.          Ref. dens.",
            f"{_f20(rho_oc)}{_f20(0)}",
            "#                  E                  Ka                  If",
            f"{_f20(E_oc)}{_f20(0)}{_f20(0)}",
            "#                 P0                 Phi             Gamma_0",
            f"{_f20(1e-5)}{_f20(0.15)}{_f20(0)}",
            "#                  C1                  C2                  C3                  C4                  C5",
            f"{_f20(-1e-4)}{_f20(5e-4)}{_f20(-2.5)}{_f20(0)}{_f20(0)}",
            "#          fct_IDd           fct_IDs            Fscale_d            Fscale_s",
            f"{_i10(0)}{_i10(0)}{_f20(1.0)}{_f20(1.0)}",
        ]
        # MAT 5: Chassis — aluminum elastic
        L += [
            "/MAT/ELAST/5", "Chassis_Aluminum",
            "#        Init. dens.",
            f"{_f20(rho_ch)}",
            "#                    E                  Nu",
            f"{_f20(E_ch)}{_f20(0.33)}",
        ]
        # MAT 6: Ground — steel elastic (effectively rigid via BCS; all nodes fixed)
        L += [
            "/MAT/ELAST/6", "Ground_Steel",
            "#        Init. dens.",
            f"{_f20(rho_gnd)}",
            "#                    E                  Nu",
            f"{_f20(210000.0)}{_f20(0.30)}",
        ]

        # ── Properties ──
        sec("PROPERTIES")
        # PROP 1: Box shell — SHELL type 24
        L += [
            "/PROP/SHELL/1", "Box_Shell",
            "#   Ishell    Ismstr     Ish3n    Idrill",
            f"{_i10(24)}{_i10(0)}{_i10(0)}{_i10(0)}",
            "#                 hm                  hf                  hr                  dm                  dn",
            f"{_f20(0)}{_f20(0)}{_f20(0)}{_f20(0)}{_f20(0)}",
            "#        N                         Thick",
            f"{_i10(1)}{' '*10}{_f20(bt)}",
        ]
        # PROP 2: Cushion solid — SOLID Isolid=14
        for pid in [2]:
            L += [
                f"/PROP/SOLID/{pid}", f"Cushion_Solid_{pid}",
                "#   Isolid    Ismstr               Icpre  Itetra10     Inpts   Itetra4    Iframe",
                f"{_i10(14)}{_i10(-1)}{_i10(-1)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(-1)}",
                "#                q_a                 q_b                   h",
                f"{_f20(0)}{_f20(0)}{_f20(0)}",
                "#             dt_min", f"{_f20(0)}",
            ]
        # PROP 4: OpenCell solid-shell (Isolid=12 HEPH, 1 layer hex through thickness)
        L += [
            "/PROP/SOLID/4", "OpenCell_SolidShell",
            "#   Isolid    Ismstr               Icpre  Itetra10     Inpts   Itetra4    Iframe",
            f"{_i10(17)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}",
            "#                q_a                 q_b                   h",
            f"{_f20(0)}{_f20(0)}{_f20(0)}",
            "#             dt_min", f"{_f20(0)}",
        ]
        # PROP 5: Chassis solid-shell (Isolid=12 HEPH, 1 layer hex through thickness)
        L += [
            "/PROP/SOLID/5", "Chassis_SolidShell",
            "#   Isolid    Ismstr               Icpre  Itetra10     Inpts   Itetra4    Iframe",
            f"{_i10(17)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}",
            "#                q_a                 q_b                   h",
            f"{_f20(0)}{_f20(0)}{_f20(0)}",
            "#             dt_min", f"{_f20(0)}",
        ]
        # PROP 6: Ground solid
        L += [
            "/PROP/SOLID/6", "Ground_Solid",
            "#   Isolid    Ismstr               Icpre  Itetra10     Inpts   Itetra4    Iframe",
            f"{_i10(14)}{_i10(-1)}{_i10(-1)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(-1)}",
            "#                q_a                 q_b                   h",
            f"{_f20(0)}{_f20(0)}{_f20(0)}",
            "#             dt_min", f"{_f20(0)}",
        ]

        # ── Include mesh data for all parts ──
        sec("MESH DATA")
        for p in self._parts:
            inc_name = f"{self.name}_part{p.part_id}_{p.name}.inc"
            offset = p.part_id * 1000000
            L += [
                f"//SUBMODEL/{p.part_id}",
                f"{p.name}",
                "# off_def, off_nod, off_ele, off_part, off_mat, off_type, off_sub",
                f"{_i10(0)}{_i10(offset)}{_i10(offset)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}",
                f"#include {inc_name}",
                "//ENDSUB"
            ]

        # ── Node/Part groups for BCS and INIVEL ──
        sec("GROUPS")
        L += [
            "/GRNOD/PART/1", "Ground_Nodes",
            "         6",
            "/GRNOD/PART/2", "Package_Parts",
            "         1         2         4         5",
            "/GRNOD/NODE/3", "Ground_Master_Node",
            "   6000000",
        ]

        sec("RIGID BODIES")
        L += [
            "/NODE",
            "   6000000                 0.0                 0.0                 0.0",
            "/RBODY/1", "Ground_Rigid",
            "#     RBID     ISENS     NSKEW    ISPHER                MASS   Gnod_id     IKREM      ICOG   Surf_id",
            "   6000000         0         0         0                 1.0         1         0         1         0",
            "#                Jxx                 Jyy                 Jzz",
            "                 1.0                 1.0                 1.0",
            "#                Jxy                 Jyz                 Jxz",
            "                 0.0                 0.0                 0.0"
        ]

        sec("BOUNDARY CONDITIONS")
        L += [
            "/BCS/1", "Ground_Fixed",
            "#  Tra rot   skew_ID  grnod_ID",
            f"   111 111{_i10(0)}{_i10(3)}",
        ]

        # Initial velocity
        # MuJoCo velocity is in m/s, rad/s.
        # Translation velocity must be converted to mm/s (1m = 1000mm)
        vx = self.v_vec[0] * 1000.0
        vy = self.v_vec[1] * 1000.0
        vz = self.v_vec[2] * 1000.0

        wx = self.omega_vec[0]
        wy = self.omega_vec[1]
        wz = self.omega_vec[2]

        # If current frame velocity is extremely small, fallback to initial gravity-based drop velocity
        v_norm = np.linalg.norm(self.v_vec)
        w_norm = np.linalg.norm(self.omega_vec)
        if v_norm < 1e-5 and w_norm < 1e-5:
            h_mm = self.h * mm
            v0 = math.sqrt(max(2.0 * G * h_mm, 0.0))   # mm/s
            vz = -v0
            vx = 0.0
            vy = 0.0

        sec("INITIAL CONDITIONS")
        L += [
            "/INIVEL/TRA/1", "Drop_Velocity",
            "#                 Vx                  Vy                  Vz   Gnod_id   Skew_id",
            f"{_f20(vx)}{_f20(vy)}{_f20(vz)}{_i10(2)}{_i10(0)}",
        ]

        # Add initial rotation velocity card if there is angular velocity
        if abs(wx) > 1e-5 or abs(wy) > 1e-5 or abs(wz) > 1e-5:
            L += [
                "/INIVEL/ROT/2", "Drop_Rotation",
                "#                 Wx                  Wy                  Wz   Gnod_id   Skew_id",
                f"{_f20(wx)}{_f20(wy)}{_f20(wz)}{_i10(2)}{_i10(0)}",
            ]

        # Gravity (all nodes, grnd_ID=0 → all)
        L += [
            "/GRAV/1", "Gravity_Z",
            "#   fct_IDT       Dir   skew_ID   sens_ID   grnd_ID              Ascalex              FscaleY",
            f"{_i10(0)}{'Z':>10}{_i10(0)}{_i10(0)}{_i10(0)}{_f20(1.0)}{_f20(-G)}",
        ]

        # ── Transforms ──
        sec("TRANSFORMS")
        self._append_transforms(L)

        # ── Contact TYPE25 (shell parts only — solid parts excluded to avoid surface error) ──
        sec("INTERFACES")
        L += [
            "/INTER/TYPE25/1",
            "All_Shell_Contact",
            "# Surf_ID1  Surf_ID2      Istf      Ithe      Igap   Irem_i2      Idel     Iedge",
            f"{_i10(1)}{_i10(0)}{_i10(4)}{_i10(0)}{_i10(2)}{_i10(0)}{_i10(0)}{_i10(1000)}",
            "# grnd_IDS                     Gap_scale          %mesh_size           Gap_max_s           Gap_max_m",
            f"{_i10(0)}{_f20(0)}{_f20(0)}{_f20(0)}{_f20(0)}",
            "#              Stmin               Stmax     Igap0    Ishape          Edge_angle",
            f"{_f20(0)}{_f20(0)}{_i10(1)}{_i10(1)}{_f20(0)}",
            "#              Stfac                Fric           Tpressfit              Tstart               Tstop",
            f"{_f20(0)}{_f20(0.3)}{_f20(0)}{_f20(0)}{_f20(0)}",
            "#      IBC               IVIS2    Inacti               ViscS    Ithick                          Pmax",
            f"       000{_f20(0)}{_i10(6)}{_f20(0)}{_i10(0)}{_f20(0)}",
            "#    Ifric    Ifiltr               Xfreq             sens_ID                                 fric_ID",
            f"{_i10(0)}{_i10(0)}{_f20(0)}{_i10(0)}{_f20(0)}",
            "/SURF/PART/EXT/1",
            "All_Parts_Surface",
            # Include Box(1), Cushion(2), OpenCell(4), Chassis(5) for global contact
            f"{_i10(1)}{_i10(2)}{_i10(4)}{_i10(5)}",
        ]

        # ── Ground rigid solid contact surface ──
        ground_md = next(p for p in self._parts if p.part_id == 6)
        sec("GROUND SURFACE")
        L += ["/SURF/SEG/2", "Ground_Top_Face"]
        g_offset = ground_md.part_id * 1000000
        for seg in ground_md.top_face_segs:
            # seg = (seg_id, n1, n2, n3, n4)
            seg_id, n1, n2, n3, n4 = seg
            L.append("".join(_i10(v) for v in (seg_id, n1+g_offset, n2+g_offset, n3+g_offset, n4+g_offset)))

        sec("GROUND CONTACT")
        L += [
            "/INTER/TYPE25/2", "Box_To_Ground",
            "# Surf_ID1  Surf_ID2      Istf      Ithe      Igap   Irem_i2      Idel     Iedge",
            f"{_i10(1)}{_i10(2)}{_i10(4)}{_i10(0)}{_i10(2)}{_i10(0)}{_i10(0)}{_i10(1000)}",
            "# grnd_IDS                     Gap_scale          %mesh_size           Gap_max_s           Gap_max_m",
            f"{_i10(0)}{_f20(0)}{_f20(0)}{_f20(0)}{_f20(0)}",
            "#              Stmin               Stmax     Igap0    Ishape          Edge_angle",
            f"{_f20(0)}{_f20(0)}{_i10(1)}{_i10(1)}{_f20(0)}",
            "#              Stfac                Fric           Tpressfit              Tstart               Tstop",
            f"{_f20(0)}{_f20(0.3)}{_f20(0)}{_f20(0)}{_f20(0)}",
            "#      IBC               IVIS2    Inacti               ViscS    Ithick                          Pmax",
            f"       000{_f20(0)}{_i10(6)}{_f20(0)}{_i10(0)}{_f20(0)}",
            "#    Ifric    Ifiltr               Xfreq             sens_ID                                 fric_ID",
            f"{_i10(0)}{_i10(0)}{_f20(0)}{_i10(0)}{_f20(0)}",
        ]

        # ── Time histories ──
        sec("TIME HISTORIES")
        L += [
            "/TH/PART/1", "Part_TH",
            "#     var1      var2",
            "DEF       ",
            "#     Obj1      Obj2      Obj3      Obj4      Obj5      Obj6",
            "         1         2         3         4         5         6",
        ]

        L += ["/END", SEP, ""]
        path.write_text("\n".join(L), encoding="utf-8")
        print(f"[Radioss] starter → {path}")
        return path

    def _write_engine(self) -> None:
        path  = self.out / f"{self.name}_0001.rad"
        h_mm  = self.h * 1000.0
        v0    = math.sqrt(max(2.0 * G * h_mm, 0.0))    # mm/s
        t_end = self.cfg.get("export_radioss_time", self.cfg.get("radioss_sim_duration", 0.05))
        if t_end == 0.05 and "sim_duration" in self.cfg:
            t_end = self.cfg["sim_duration"]
        dt_anim = self.cfg.get("export_radioss_dt_anim", 0.001)
        print_interval = self.cfg.get("radioss_print_interval", -10)

        L = [
            "#RADIOSS ENGINE",
            f"# Generated by WHToolsBox RadiossModelBuilder",
            SEP,
            f"/RUN/{self.name}/1/",
            f"{_f20(t_end)}",
            "/VERS/2022",
            "/STOP",
            "# Emax Mmax Nmax NTH NANIM NERR_POSIT",
            "0.99 0.05 0 1 1 1",
            "/TFILE/4",
            f"{_f20(1e-4)}",
            "/DT/NODA/CST/0",
            "    0.9   0.0   0.0",
            "/ANIM/DT",
            f"{_f20(0.0)}{_f20(dt_anim)}",
            "/ANIM/VECT/VEL",
            "/ANIM/VECT/DISP",
            "/ANIM/ELEM/EPSP",
            "/ANIM/ELEM/VONM",
            "/H3D/DT",
            f"{_f20(0.0)}{_f20(dt_anim)}",
            "/H3D/NODA/VEL",
            "/H3D/NODA/DIS",
            "/H3D/SHELL/EPSP",
            "/H3D/SHELL/VONM",
            "/H3D/SOLID/EPSP",
            "/H3D/SOLID/VONM",
            f"/PRINT/{print_interval}",
            "/PARITH/ON",
            "/END",
            SEP,
            "",
        ]
        path.write_text("\n".join(L), encoding="utf-8")
        print(f"[Radioss] engine  → {path}")

    # ── transform helpers ─────────────────────────────────────────────────────

    def _append_transforms(self, L: list) -> None:
        R = self.R.copy()
        t = self.t.copy()   # mm

        # Determine target group
        if self.mode == 'parts':
            grnod = 2         # package (parts 1-5)
        else:
            grnod = 3         # ground (part 6), apply inverse
            R     = R.T
            t     = -(R @ self.t)

        identity_R = np.allclose(R, np.eye(3), atol=1e-6)
        zero_t     = np.allclose(t, np.zeros(3), atol=1e-3)

        if identity_R and zero_t:
            L.append("# No transform — identity pose")
            return

        # GRNOD/PART/2 (parts 1-5) already defined in initial conditions — reuse it.
        # For ground mode, define a new group.
        if self.mode != 'parts':
            L += ["/GRNOD/PART/3", "Ground_Xform", "         6"]

        xform_id = 1
        ax_node1 = 900001
        ax_node2 = 900002

        if not identity_R:
            angle_rad, axis = self._mat_to_axis_angle(R)
            angle_deg = math.degrees(angle_rad)
            # Reference nodes for rotation axis (added outside any part, very high IDs)
            L += [
                "# Rotation axis reference nodes",
                "/NODE",
                f"{_i10(ax_node1)}{_f20(0.0)}{_f20(0.0)}{_f20(0.0)}",
                f"{_i10(ax_node2)}{_f20(float(axis[0]))}{_f20(float(axis[1]))}{_f20(float(axis[2]))}",
            ]
            L += [
                f"/TRANSFORM/ROT/{xform_id}", "Initial_Rotation",
                "# grnd_ID(10) X1(20) Y1(20) Z1(20) node1(10) node2(10) sub_ID(10)",
                f"{_i10(grnod)}{_f20(0)}{_f20(0)}{_f20(0)}{_i10(ax_node1)}{_i10(ax_node2)}{_i10(0)}",
                "# X2(20) Y2(20) Z2(20) Angle(20)",
                f"{_f20(0)}{_f20(0)}{_f20(0)}{_f20(angle_deg)}",
            ]
            xform_id += 1

        if not zero_t:
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            L += [
                f"/TRANSFORM/TRA/{xform_id}", "Initial_Translation",
                f"{_i10(grnod)}{_f20(tx)}{_f20(ty)}{_f20(tz)}",
            ]

    @staticmethod
    def _mat_to_axis_angle(R: np.ndarray):
        """Rotation matrix → (angle_rad, unit_axis) using Rodrigues formula."""
        cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(math.acos(cos_a))

        if abs(angle) < 1e-8:
            return 0.0, np.array([0.0, 0.0, 1.0])

        if abs(angle - math.pi) < 1e-6:
            diag = np.diag(R)
            axis = np.zeros(3); axis[int(np.argmax(diag))] = 1.0
            return angle, axis

        sin_a = math.sin(angle)
        axis  = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2.0 * sin_a)
        return angle, axis / np.linalg.norm(axis)

    def _write_lsdyna(self) -> Path:
        """Export an LS-DYNA (.k) equivalent file for viewer compatibility."""
        path = self.out / f"{self.name}_LSDYNA.k"
        L = [
            "*KEYWORD",
            "*TITLE",
            "TVDrop LS-DYNA Model Generated by WHToolsBox",
            "*CONTROL_TERMINATION",
            " 0.05",
            "*CONTROL_TIMESTEP",
            " 0.0, 0.9",
        ]
        
        # Nodes and Elements
        for p in self._parts:
            offset = p.part_id * 1000000
            L.append("*NODE")
            for nid, (x, y, z) in sorted(p.nodes.items()):
                L.append(f"{nid + offset:>8d}{x:>16.5f}{y:>16.5f}{z:>16.5f}")
            
            if p.is_shell and p.quads:
                L.append("*ELEMENT_SHELL")
                for eid, ns in sorted(p.quads.items()):
                    L.append(f"{eid + offset:>8d}{p.part_id:>8d}" + "".join([f"{n + offset:>8d}" for n in ns]))
            elif (not p.is_shell) and p.hexas:
                L.append("*ELEMENT_SOLID")
                for eid, ns in sorted(p.hexas.items()):
                    L.append(f"{eid + offset:>8d}{p.part_id:>8d}" + "".join([f"{n + offset:>8d}" for n in ns]))
            elif (not p.is_shell) and p.tetras:
                L.append("*ELEMENT_SOLID")
                for eid, ns in sorted(p.tetras.items()):
                    # LS-DYNA tetras are often written as 4-node solids or with repeated last node
                    # Here we just write the 4 nodes as 4-node solid or repeat 4th. We'll repeat 4th.
                    L.append(f"{eid + offset:>8d}{p.part_id:>8d}{ns[0] + offset:>8d}{ns[1] + offset:>8d}{ns[2] + offset:>8d}{ns[3] + offset:>8d}{ns[3] + offset:>8d}{ns[3] + offset:>8d}{ns[3] + offset:>8d}{ns[3] + offset:>8d}")

        # Parts
        for p in self._parts:
            L.append("*PART")
            L.append(f"{p.name}")
            L.append(f"{p.part_id:>10d}{p.part_id:>10d}{p.part_id:>10d}")

        # Densities calculation
        def _vol(w, h, d): return w * h * d
        cush_w, cush_h, cush_d = self.cfg.get('cw', 100), self.cfg.get('ch', 100), self.cfg.get('cd', 50)
        oc_d, chas_d = self.cfg.get('od', 30), self.cfg.get('ct', 40)
        aw, ah = self.cfg.get('aw', 1100), self.cfg.get('ah', 700)
        
        bw, bh, bd, bt = self.cfg.get('bw', 1300), self.cfg.get('bh', 900), self.cfg.get('bd', 250), self.cfg.get('bt', 5.0)
        sa_box   = 2 * (bw*bh + bh*bd + bw*bd)
        vol_box  = sa_box * bt
        rho_box  = self.cfg.get('mass_paper', 4.0) / 1000.0 / vol_box if vol_box > 0 else 1e-9
        
        vol_cush = _vol(aw, ah, cush_d * 2)
        rho_cush = self.cfg.get('mass_cushion', 2.0) / 1000.0 / vol_cush if vol_cush > 0 else 3e-11
        
        vol_oc   = _vol(aw, ah, oc_d)
        rho_oc   = self.cfg.get('mass_oc', 5.0) / 1000.0 / vol_oc if vol_oc > 0 else 2e-11
        
        vol_ch   = _vol(aw, ah, chas_d)
        rho_ch   = self.cfg.get('mass_chassis', 10.0) / 1000.0 / vol_ch if vol_ch > 0 else 2.7e-9
        
        rho_gnd  = 7.8e-9

        def f10(val): return f"{val:>10.3e}"
        
        # 1 - Box (Shell)
        E_box = self.cfg.get('E_paper', 3000.0)
        L.append("*MAT_ELASTIC")
        L.append(f"{1:>10d}{f10(rho_box)}{f10(E_box)}{f10(0.3)}")
        L.append("*SECTION_SHELL")
        L.append(f"{1:>10d}{10:>10d}{0:>10.3f}{0:>10d}{0:>10.3f}{0:>10.3f}{0:>10d}")
        L.append(f"{f10(self.cfg.get('bt', 5.0))}")
        
        # 2 - Cushion (Solid)
        # Note: In the new unified mesh approach, cushion might be just part 2
        E_c = self.cfg.get('E_cushion', 0.5)
        L.append("*MAT_PIECEWISE_LINEAR_PLASTICITY")
        L.append(f"{2:>10d}{f10(rho_cush)}{f10(E_c)}{f10(0.3)}{f10(0.1)}{f10(100.0)}")
        L.append("*SECTION_SOLID")
        L.append(f"{2:>10d}{1:>10d}")
            
        # 4 - OpenCell (Solid)
        E_oc = self.cfg.get('E_oc', 0.08)
        L.append("*MAT_ELASTIC")
        L.append(f"{4:>10d}{f10(rho_oc)}{f10(E_oc)}{f10(0.3)}")
        L.append("*SECTION_SOLID")
        L.append(f"{4:>10d}{1:>10d}")

        # 5 - Chassis (Shell)
        E_ch = self.cfg.get('E_chassis', 10000.0)
        L.append("*MAT_ELASTIC")
        L.append(f"{5:>10d}{f10(rho_ch)}{f10(E_ch)}{f10(0.33)}")
        L.append("*SECTION_SHELL")
        L.append(f"{5:>10d}{2:>10d}")
        L.append(f"{f10(1.0)}")
        
        # 6 - Ground (Solid, MAT_RIGID)
        L.append("*MAT_RIGID")
        L.append(f"{6:>10d}{f10(rho_gnd)}{f10(210000.0)}{f10(0.3)}")
        L.append(f"{1.0:>10.3f}{7:>10.3f}{7:>10.3f}")
        L.append("*SECTION_SOLID")
        L.append(f"{6:>10d}{1:>10d}")
        
        # Initial Velocity
        omega_ms = np.linalg.norm(self.omega_vec) * 0.001
        v_mag = np.linalg.norm(self.v_vec)
        if omega_ms > 1e-6 or v_mag > 1e-6:
            axis = self.omega_vec / (np.linalg.norm(self.omega_vec) + 1e-12)
            L.append("*INITIAL_VELOCITY_GENERATION")
            L.append(f"{2:>10d}{2:>10d}{axis[0]:>10.3f}{axis[1]:>10.3f}{axis[2]:>10.3f}{self.v_vec[0]:>10.3f}{self.v_vec[1]:>10.3f}{self.v_vec[2]:>10.3f}")
            L.append(f"{0.0:>10.3f}{0.0:>10.3f}{0.0:>10.3f}{0.0:>10.3f}{0.0:>10.3f}{0.0:>10.3f}{0:>10d}")
            
        # Contact
        L.append("*CONTACT_AUTOMATIC_SINGLE_SURFACE")
        L.append(f"{0:>10d}{0:>10d}{0:>10d}") # Card 1
        L.append("") # Card 2
        L.append("") # Card 3
        
        # Gravity
        L.append("*LOAD_BODY_Z")
        L.append(f"{1:>10d}")
        L.append("*DEFINE_CURVE")
        L.append(f"{1:>10d}{0:>10d}{1.0:>10.3f}{1.0:>10.3f}{0.0:>10.3f}{0.0:>10.3f}")
        G_MM_MS2 = 9.81e-3
        L.append(f"{0.0:>20.5f}{-G_MM_MS2:>20.5f}")
        L.append(f"{1.0:>20.5f}{-G_MM_MS2:>20.5f}")

        L.append("*END")
        
        path.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"[LS-DYNA] K file -> {path.name}")
        return path
