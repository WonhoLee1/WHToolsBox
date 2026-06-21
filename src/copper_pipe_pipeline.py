# -*- coding: utf-8 -*-
"""
copper_pipe_pipeline.py
=======================
3D serpentine copper pipe — CalculiX implicit direct dynamic analysis.

Pipeline:
  1. Gmsh 1D beam mesh  (generate_mesh)
  2. Master INP assembly (assemble_inp)
  3. CalculiX solver     (run_solver)
  4. FRD → VTU → VTKHDF (convert_results)

Geometry (undeformed):
  X = 3 × 140 mm = 420 mm   (leg spacing × N_BENDS)
  Y = 450 mm                 (leg length)
  Z = 4 × 100 mm = 400 mm   (Z-rise × legs)
  All dimensions > 333 mm (= 1000/3) and within 1000 mm  ✓

Usage:
  python src/copper_pipe_pipeline.py
"""

import sys
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
WORKSPACE = BASE_DIR / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)

# ── External executables ──────────────────────────────────────────────────────
CALCULIX_EXE = r"D:\SOFTWARE\calculix_2.23_4win\ccx_static.exe"

# ── Geometry parameters ───────────────────────────────────────────────────────
# Flat 2D serpentine in XY plane (z=0).
# Each U-bend split into 2 quarter-arcs for G1-continuous tangent.
#
#  (R,L+R) apex
#     /\
#    /  \            X=0      X=S      X=2S     X=3S
#   /    \           |         |        |         |
# (0,L)  (S,L)      leg0      leg1     leg2      leg3
#   |      |         |   top  |        |   top  |
# (0,0)  (S,0)     y=0      y=0      y=0       y=0  ← End A/B
#          \  nadir /
#        (3S/2,-R)
#
# Bounding box: X=3*S=420mm, Y=-(R) to L+R=(-70 to 520mm) → 590mm, Z=0
LEG_Y      = 450.0   # mm  straight leg length  (> 333 ✓)
LEG_SPACE  = 140.0   # mm  centre-to-centre spacing = 2 × bend_radius
N_BENDS    = 3       # number of U-bends  → 4 straight legs
MESH_SIZE  = 30.0    # mm  target element length

# ── Material: Copper ──────────────────────────────────────────────────────────
E_CU   = 110_000.0   # MPa
NU_CU  = 0.34
RHO_CU = 8.96e-9    # ton/mm³  (= 8 960 kg/m³)

# ── Pipe cross-section ────────────────────────────────────────────────────────
OUTER_R = 5.0   # mm  outer radius
WALL_T  = 1.0   # mm  wall thickness

# ── Dynamic analysis ──────────────────────────────────────────────────────────
EXCIT_FREQ  = 2.0     # Hz   excitation frequency
EXCIT_AMPL  = 100.0   # mm   peak displacement amplitude
EXCIT_DOF   = 3       # 3 = global Z (out-of-plane)
T_END       = 10.0    # s    total analysis duration
DT          = 0.025   # s    time increment (20 steps / period at 2 Hz)
OUT_INCR    = int(round(0.1 / DT))   # increments between outputs = 4

# ── File paths ────────────────────────────────────────────────────────────────
JOB_NAME    = "copper_pipe"
MESH_INP    = WORKSPACE / "copper_pipe_mesh.inp"
MESH_B31    = WORKSPACE / "copper_pipe_mesh_b31.inp"
MASTER_INP  = WORKSPACE / f"{JOB_NAME}.inp"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 – Geometry & Mesh
# ═══════════════════════════════════════════════════════════════════════════════

def generate_mesh() -> Path:
    """
    Flat 2D serpentine pipe in XY plane (z=0), G1-continuous tangent.

    Each U-bend is split into two 90° quarter-circle arcs sharing an apex/nadir
    point.  This ensures the arc tangent matches the leg direction at every
    junction — no kinks.

    Even bends (0, 2, …): bulge UP   (apex at y = L + R)
    Odd  bends (1, 3, …): bulge DOWN (nadir at y = -R)

    Layout for N_BENDS=3:

      x=0   x=S   x=2S  x=3S
       |     |     |     |
      [L]   [L]   [L]   [L]   ← leg tops (y=L)
       |  /*\  |  /*\  |
      [0]   [0]   [0]   [0]   ← leg bottoms (y=0) ← End A & End B
             \_/         (nadir at y=-R)
    """
    import gmsh

    S  = LEG_SPACE   # 140 mm
    L  = LEG_Y       # 450 mm
    R  = S / 2       #  70 mm  bend radius
    ms = MESH_SIZE
    nb = N_BENDS

    gmsh.initialize()
    gmsh.model.add("copper_pipe")

    # ── Leg endpoint coordinates ──────────────────────────────────────────────
    # leg_pts[2i]   = start of leg i   (y=0 for even i, y=L for odd i)
    # leg_pts[2i+1] = end   of leg i   (y=L for even i, y=0 for odd i)
    leg_pts: list[tuple] = []
    for i in range(nb + 1):
        x  = i * S
        y0 = 0.0 if i % 2 == 0 else L
        y1 = L   if i % 2 == 0 else 0.0
        leg_pts.append((x, y0, 0.0))
        leg_pts.append((x, y1, 0.0))

    # ── Bend centres and apex/nadir points ───────────────────────────────────
    # Bend i connects leg i (end) to leg i+1 (start).
    # Centre is horizontally between the two legs, at the height of the bend.
    # Even bends → centre at (i·S+R, L, 0), apex  at (i·S+R, L+R, 0)
    # Odd  bends → centre at (i·S+R, 0, 0), nadir at (i·S+R, -R,  0)
    bend_center_pts: list[tuple] = []
    bend_mid_pts:    list[tuple] = []
    for i in range(nb):
        x_c = i * S + R
        if i % 2 == 0:          # top bend (upward bulge)
            bend_center_pts.append((x_c, L,      0.0))
            bend_mid_pts.append(  (x_c, L + R,   0.0))
        else:                   # bottom bend (downward bulge)
            bend_center_pts.append((x_c, 0.0,    0.0))
            bend_mid_pts.append(  (x_c, -R,      0.0))

    # ── Create Gmsh points ───────────────────────────────────────────────────
    leg_tags    = [gmsh.model.geo.addPoint(x, y, z, ms) for (x, y, z) in leg_pts]
    center_tags = [gmsh.model.geo.addPoint(x, y, z, ms) for (x, y, z) in bend_center_pts]
    mid_tags    = [gmsh.model.geo.addPoint(x, y, z, ms) for (x, y, z) in bend_mid_pts]

    # ── Straight legs ────────────────────────────────────────────────────────
    leg_curves = [
        gmsh.model.geo.addLine(leg_tags[2 * i], leg_tags[2 * i + 1])
        for i in range(nb + 1)
    ]

    # ── Quarter-circle arcs (2 per bend) ────────────────────────────────────
    # Bend i: leg_tags[2i+1] → mid_tags[i] → leg_tags[2*(i+1)]
    # gmsh always picks the SHORT arc (90° here), giving correct tangent.
    bend_curves: list[tuple] = []
    for i in range(nb):
        arc_a = gmsh.model.geo.addCircleArc(
            leg_tags[2 * i + 1], center_tags[i], mid_tags[i]
        )
        arc_b = gmsh.model.geo.addCircleArc(
            mid_tags[i], center_tags[i], leg_tags[2 * (i + 1)]
        )
        bend_curves.append((arc_a, arc_b))

    # ── Full ordered curve list ──────────────────────────────────────────────
    all_curves: list[int] = []
    for i in range(nb + 1):
        all_curves.append(leg_curves[i])
        if i < nb:
            all_curves.extend(bend_curves[i])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, all_curves, tag=1, name="COPPER_PIPE")
    gmsh.model.addPhysicalGroup(0, [leg_tags[0]],   tag=1, name="END_A")
    gmsh.model.addPhysicalGroup(0, [leg_tags[-1]],  tag=2, name="END_B")

    gmsh.option.setNumber("Mesh.MeshSizeMax", ms)
    gmsh.option.setNumber("Mesh.MeshSizeMin", ms)
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(2)   # quadratic → T3D3 → B32R (needed for PIPE section)
    gmsh.write(str(MESH_INP))
    gmsh.finalize()

    print(f"[Mesh] Written: {MESH_INP}")
    return MESH_INP


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 – INP Assembly
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_nodes(mesh_text: str) -> dict[int, tuple[float, float, float]]:
    """Return {node_id: (x, y, z)} from the *NODE section of an INP file."""
    nodes: dict[int, tuple[float, float, float]] = {}
    in_node = False
    for line in mesh_text.splitlines():
        ls = line.strip()
        if ls.upper().startswith("*NODE"):
            in_node = True
            continue
        if in_node:
            if ls.startswith("*"):
                in_node = False
                continue
            parts = ls.split(",")
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[nid] = (x, y, z)
                except ValueError:
                    pass
    return nodes


def _closest_node(
    nodes: dict[int, tuple[float, float, float]],
    target: tuple[float, float, float],
) -> int:
    """Return the node id whose coordinates are closest to target."""
    tx, ty, tz = target
    best_id, best_d2 = -1, float("inf")
    for nid, (x, y, z) in nodes.items():
        d2 = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
        if d2 < best_d2:
            best_d2, best_id = d2, nid
    return best_id


def assemble_inp(mesh_inp: Path) -> Path:
    """
    Post-process Gmsh INP (T3D2 → B31), inject NSETs for END_A / END_B /
    PIPE_NODES, and write the master CalculiX INP.

    Boundary conditions:
      End A : all 6 DOF = 0  (fully clamped)
      End B : DOF 1,2,4,5,6 = 0 ; DOF 3 = EXCIT_AMPL × sin(2π × EXCIT_FREQ × t)
    """
    # Patch element type and strip the *Heading block (master INP provides its own)
    raw     = mesh_inp.read_text(encoding="utf-8", errors="replace")
    # T3D3 = 3-node quadratic line (Gmsh quadratic export) → B32R (CalculiX PIPE-capable beam)
    # T3D2 = 2-node linear line → B31 (fallback for linear mesh)
    patched = re.sub(r"TYPE\s*=\s*T3D3", "TYPE=B32R", raw, flags=re.IGNORECASE)
    patched = re.sub(r"TYPE\s*=\s*T3D2", "TYPE=B31",  patched, flags=re.IGNORECASE)
    # Remove *Heading + its content line so the included file has no duplicate header
    patched = re.sub(r"^\*Heading\b[^\n]*\n[^\n]*\n", "", patched,
                     flags=re.IGNORECASE | re.MULTILINE)

    # Identify end nodes by coordinate proximity
    nodes     = _parse_nodes(patched)
    node_a    = _closest_node(nodes, (0.0, 0.0, 0.0))
    node_b    = _closest_node(nodes, (N_BENDS * LEG_SPACE, 0.0, 0.0))
    all_nodes = sorted(nodes.keys())

    # Format node list with max 16 entries per line (CalculiX limit)
    def _nset_lines(nids: list[int]) -> str:
        rows = [nids[i : i + 16] for i in range(0, len(nids), 16)]
        return "\n".join(", ".join(str(n) for n in row) for row in rows) + "\n"

    # Inject NSETs at the end of the patched mesh
    nset_block = (
        f"\n*NSET, NSET=END_A\n {node_a},\n"
        f"*NSET, NSET=END_B\n {node_b},\n"
        f"*NSET, NSET=PIPE_NODES\n"
        + _nset_lines(all_nodes)
    )
    patched += nset_block

    MESH_B31.write_text(patched, encoding="utf-8")
    print(f"[INP]  Patched mesh → {MESH_B31.name}  (END_A=node{node_a}, END_B=node{node_b})")

    # Boundary condition lines for End B (all DOF except EXCIT_DOF)
    dofs_before = list(range(1, EXCIT_DOF))
    dofs_after  = list(range(EXCIT_DOF + 1, 7))
    endb_fixed_lines: list[str] = []
    if dofs_before:
        endb_fixed_lines.append(
            f"END_B, {dofs_before[0]}, {dofs_before[-1]}, 0.0"
        )
    if dofs_after:
        endb_fixed_lines.append(
            f"END_B, {dofs_after[0]}, {dofs_after[-1]}, 0.0"
        )
    endb_fixed_block = "\n".join(endb_fixed_lines)

    inc_total = int(T_END / DT) + 20   # total increment budget with buffer

    # Hollow circular pipe section properties (GENERAL section, B31-compatible)
    import math as _math
    Ro, Ri = OUTER_R, OUTER_R - WALL_T
    A_sec  = _math.pi * (Ro**2 - Ri**2)
    I_sec  = _math.pi * (Ro**4 - Ri**4) / 4.0   # I11 = I22
    IT_sec = _math.pi * (Ro**4 - Ri**4) / 2.0   # torsional constant J

    # Tabular amplitude for sin(2π*f*t): DEFINITION=PERIODIC not supported
    # by this CalculiX binary, so use tabular with 20 pts/period
    dt_amp  = DT  # same as analysis dt → exact values at output times
    n_pts   = int(T_END / dt_amp) + 1
    amp_pairs: list[str] = []
    for i in range(n_pts):
        t_a = i * dt_amp
        a_a = _math.sin(2.0 * _math.pi * EXCIT_FREQ * t_a)
        amp_pairs.append(f"{t_a:.4f}, {a_a:.8f}")
    # 4 pairs per line (CalculiX amplitude table format)
    amp_lines = []
    for i in range(0, len(amp_pairs), 4):
        amp_lines.append(", ".join(amp_pairs[i : i + 4]))
    amp_table = "\n".join(amp_lines)

    inp = f"""\
** ============================================================
** AutoCalculix: 3D Serpentine Copper Pipe
** Implicit Direct Dynamic Analysis (non-modal)
** Bounding box (undeformed): X={int(N_BENDS*LEG_SPACE)} mm, Y={int(LEG_Y + LEG_SPACE)} mm, Z=0 mm (flat 2-D)
** Excitation: DOF {EXCIT_DOF} (Z) = {EXCIT_AMPL} mm × sin(2π×{EXCIT_FREQ}×t), 0≤t≤{T_END} s
** ============================================================
*HEADING
Copper pipe U-bend implicit dynamic analysis

*INCLUDE, INPUT={MESH_B31.name}

** ── Material: Copper ────────────────────────────────────────
*MATERIAL, NAME=COPPER
*ELASTIC
{E_CU:.1f}, {NU_CU}
*DENSITY
{RHO_CU}

** ── Beam section: circular pipe (PIPE, B32R) ────────────────────────
** outer_radius={OUTER_R} mm, wall_thickness={WALL_T} mm
** local-1 direction hint: (0, 0, 1)
*BEAM SECTION, SECTION=PIPE, ELSET=COPPER_PIPE, MATERIAL=COPPER
{OUTER_R}, {WALL_T}
0., 0., 1.

** ── Self-contact NOTE ─────────────────────────────────────────
** CalculiX beam (B32R) elements are 1-D and have no element faces,
** so face-based self-contact (*CONTACT PAIR, TYPE=NODE TO SURFACE)
** cannot be constructed.  The geometry has bend radius {LEG_SPACE/2:.0f} mm >>
** pipe outer radius {OUTER_R} mm, so inter-pipe contact is geometrically
** prevented for the expected deformation amplitudes.
** If contact is required, replace beam elements with 3-D pipe solids.

** ── Sinusoidal amplitude: tabular sin(2π×{EXCIT_FREQ}×t), 0≤t≤{T_END} s ──
** (DEFINITION=PERIODIC not available in this CalculiX build)
*AMPLITUDE, NAME=SINE2HZ
{amp_table}

** ── Step: Implicit direct dynamic ───────────────────────────
*STEP, INC={inc_total}
*DYNAMIC, DIRECT
{DT:.4f}, {T_END:.1f}

** End A: fully clamped (6 DOF)
*BOUNDARY
END_A, 1, 6, 0.0

** End B: fixed DOF 1,2,4,5,6; sinusoidal DOF {EXCIT_DOF}
*BOUNDARY
{endb_fixed_block}

*BOUNDARY, AMPLITUDE=SINE2HZ
END_B, {EXCIT_DOF}, {EXCIT_DOF}, {EXCIT_AMPL:.1f}

** ── Output: every {OUT_INCR} increments = {OUT_INCR * DT:.3f} s ────────────────
*NODE FILE, FREQUENCY={OUT_INCR}
U, V, RF

*END STEP
"""

    MASTER_INP.write_text(inp, encoding="utf-8")
    print(f"[INP]  Master INP → {MASTER_INP}")
    return MASTER_INP


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 – Solver
# ═══════════════════════════════════════════════════════════════════════════════

def run_solver(job_name: str) -> bool:
    """Run CalculiX on the master INP in WORKSPACE."""
    print(f"[Solver] Running CalculiX: {job_name}")
    result = subprocess.run(
        [CALCULIX_EXE, job_name],
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("[Solver] Completed successfully.")
        return True
    else:
        print(f"[Solver] ERROR (rc={result.returncode})")
        tail = (result.stdout or "")[-3000:]
        if tail:
            print(tail)
        err = (result.stderr or "")[-1000:]
        if err:
            print(err)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 – Post-processing: FRD → VTU → VTKHDF
# ═══════════════════════════════════════════════════════════════════════════════

def convert_results(frd_path: Path) -> Path | None:
    """
    FRD → VTU (ccx2paraview) → VTKHDF (h5py).
    Returns path to the VTKHDF file, or PVD path on VTKHDF failure.
    """
    if not frd_path.exists():
        print(f"[Post] FRD not found: {frd_path}")
        return None

    # Step 1: FRD → VTU with ccx2paraview
    print("[Post] Converting FRD → VTU (ccx2paraview)…")
    try:
        from ccx2paraview import Converter
        import logging
        logging.basicConfig(level=logging.WARNING)
        Converter(str(frd_path), ["vtu"]).run()
        print("[Post] ccx2paraview done.")
    except Exception as exc:
        print(f"[Post] ccx2paraview failed: {exc}")
        return None

    # ccx2paraview writes a PVD next to the FRD file
    pvd_path = frd_path.with_suffix(".pvd")
    if not pvd_path.exists():
        # try alternative naming
        candidates = list(WORKSPACE.glob(f"{JOB_NAME}*.pvd"))
        pvd_path = candidates[0] if candidates else None

    if not pvd_path or not pvd_path.exists():
        print("[Post] PVD file not found — skipping VTKHDF conversion.")
        return None

    # Step 2: VTU series → VTKHDF
    vtkhdf_path = frd_path.with_suffix(".vtkhdf")
    print(f"[Post] Converting PVD+VTU → VTKHDF: {vtkhdf_path.name}…")
    from src.utils.vtkhdf_converter import PVDToVTKHDFConverter
    ok = PVDToVTKHDFConverter.convert(pvd_path, vtkhdf_path)
    if ok:
        print(f"[Post] VTKHDF ready: {vtkhdf_path}")
        return vtkhdf_path
    else:
        print("[Post] VTKHDF failed — VTU+PVD files remain usable in ParaView.")
        return pvd_path


def _pvd_to_vtkhdf(pvd_path: Path, vtkhdf_path: Path) -> bool:
    """
    Read a PVD+VTU time series and write a transient VTKHDF file.
    Static mesh topology; transient PointData arrays concatenated over time.
    """
    try:
        import h5py
        import numpy as np
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    except ImportError as exc:
        print(f"[VTKHDF] Missing dependency ({exc}). Skipping.")
        return False

    # Parse PVD
    tree       = ET.parse(pvd_path)
    collection = tree.getroot().find("Collection")
    if collection is None:
        print("[VTKHDF] Malformed PVD.")
        return False

    entries = sorted(
        [
            (float(ds.get("timestep", 0.0)), pvd_path.parent / ds.get("file", ""))
            for ds in collection.findall("DataSet")
        ],
        key=lambda x: x[0],
    )
    if not entries:
        print("[VTKHDF] No datasets in PVD.")
        return False

    print(f"[VTKHDF] {len(entries)} timesteps found.")

    # ── Read mesh topology from first VTU ────────────────────────────────────
    def _read_vtu(path: Path):
        rdr = vtkXMLUnstructuredGridReader()
        rdr.SetFileName(str(path))
        rdr.Update()
        return rdr.GetOutput()

    g0    = _read_vtu(entries[0][1])
    npts  = g0.GetNumberOfPoints()
    ncells = g0.GetNumberOfCells()

    import numpy as np

    pts = np.array([g0.GetPoint(i) for i in range(npts)], dtype=np.float64)

    conn, offs, types = [], [0], []
    for ci in range(ncells):
        cell = g0.GetCell(ci)
        nids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        conn.extend(nids)
        offs.append(len(conn))
        types.append(g0.GetCellType(ci))

    conn  = np.array(conn,  dtype=np.int64)
    offs  = np.array(offs,  dtype=np.int64)
    types = np.array(types, dtype=np.uint8)

    pd0          = g0.GetPointData()
    field_names  = [pd0.GetArrayName(i) for i in range(pd0.GetNumberOfArrays())]
    nsteps       = len(entries)
    times        = np.array([t for t, _ in entries], dtype=np.float64)

    # ── Collect transient field data ─────────────────────────────────────────
    fd: dict[str, list] = {n: [] for n in field_names}
    for step_idx, (_t, vtu_file) in enumerate(entries):
        g   = _read_vtu(vtu_file)
        pd  = g.GetPointData()
        for name in field_names:
            arr = pd.GetArray(name)
            if arr is not None:
                nc   = arr.GetNumberOfComponents()
                data = np.array(
                    [[arr.GetComponent(i, c) for c in range(nc)] for i in range(npts)],
                    dtype=np.float64,
                )
            else:
                nc   = 1
                data = np.zeros((npts, 1), dtype=np.float64)
            fd[name].append(data)
        if (step_idx + 1) % 20 == 0:
            print(f"[VTKHDF]   read {step_idx + 1}/{nsteps} steps…")

    # ── Write VTKHDF ─────────────────────────────────────────────────────────
    with h5py.File(vtkhdf_path, "w") as f:
        grp = f.create_group("VTKHDF")
        grp.attrs["Type"]    = np.bytes_("UnstructuredGrid")
        grp.attrs["Version"] = np.array([2, 0], dtype=np.int64)

        grp["Points"]                  = pts
        grp["Connectivity"]            = conn
        grp["Offsets"]                 = offs
        grp["Types"]                   = types
        grp["NumberOfPoints"]          = np.array([npts],       dtype=np.int64)
        grp["NumberOfCells"]           = np.array([ncells],     dtype=np.int64)
        grp["NumberOfConnectivityIds"] = np.array([len(conn)],  dtype=np.int64)

        # Steps group (static mesh: all offsets are zero)
        # NSteps must be an HDF5 *attribute* of the Steps group,
        # not a dataset — vtkHDFUtilities reads it via GetAttribute().
        sg = grp.create_group("Steps")
        sg.attrs["NSteps"]          = np.int64(nsteps)
        sg["TimeValues"]            = times
        sg["NumberOfParts"]         = np.ones(nsteps,  dtype=np.int64)
        sg["PartOffsets"]           = np.zeros(nsteps, dtype=np.int64)
        sg["PointOffsets"]          = np.zeros(nsteps, dtype=np.int64)
        sg["CellOffsets"]           = np.zeros(nsteps, dtype=np.int64)
        sg["ConnectivityIdOffsets"] = np.zeros(nsteps, dtype=np.int64)

        # PointData: concatenate all timesteps [nsteps*npts, ncomp]
        if field_names:
            pdg  = grp.create_group("PointData")
            spd  = sg.create_group("PointData")
            spdo = spd.create_group("Offsets")
            for name in field_names:
                if fd[name]:
                    cat = np.concatenate(fd[name], axis=0)
                    pdg[name] = cat
                    # Offset of each step's data block within the concatenated array
                    spdo[name] = (np.arange(nsteps + 1, dtype=np.int64) * npts)

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" Copper Pipe U-bend — Implicit Dynamic Pipeline")
    print("=" * 60)
    print(f" Bounding box  : X={int(N_BENDS*LEG_SPACE)} × Y={int(LEG_Y + LEG_SPACE)} mm (flat 2-D, z=0)")
    print(f" Excitation    : DOF {EXCIT_DOF} (Z), {EXCIT_AMPL} mm @ {EXCIT_FREQ} Hz, {T_END} s")
    print(f" Mesh size     : {MESH_SIZE} mm  |  dt = {DT} s  |  output every {OUT_INCR * DT:.2f} s")
    print()

    print("── Step 1/4: Geometry & Mesh ──────────────────────────────")
    mesh_inp = generate_mesh()

    print()
    print("── Step 2/4: INP Assembly ─────────────────────────────────")
    assemble_inp(mesh_inp)

    print()
    print("── Step 3/4: CalculiX Solver ──────────────────────────────")
    frd_path = WORKSPACE / f"{JOB_NAME}.frd"
    ok = run_solver(JOB_NAME)
    if not ok:
        print("[!] Solver failed. Aborting post-processing.")
        return

    # Verify FRD
    if not frd_path.exists():
        print(f"[!] FRD file not found: {frd_path}")
        return
    print(f"[Solver] FRD size: {frd_path.stat().st_size / 1024:.0f} kB")

    print()
    print("── Step 4/4: FRD → VTU → VTKHDF ──────────────────────────")
    result = convert_results(frd_path)
    if result:
        print(f"[Done]  Result file: {result}")
    else:
        print("[!] Post-processing failed.")

    print()
    print("=" * 60)
    print(" Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
