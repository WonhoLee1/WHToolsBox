# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import meshio

ELSET_NAME = "SHELL_ALL"

# meshio cell type → CalculiX shell element type
_CCX_TYPE = {
    'triangle':  'S3',
    'triangle3': 'S3',
    'triangle6': 'S6',
    'quad':      'S4',
    'quad4':     'S4',
    'quad8':     'S8',
}

_SURFACE_TYPES = set(_CCX_TYPE.keys())

_SUPPORTED_EXT = {'.inp', '.fem', '.bdf', '.nas', '.dat'}


class ExternalMeshLoader:
    """
    Abaqus INP 또는 OptiStruct/Nastran BDF·FEM 파일을 읽어
    CalculiX 호환 INP 파일로 변환합니다.

    입력 파일 최소 요건
    -------------------
    Abaqus INP : *NODE + *ELEMENT (TYPE=S3/S4/S4R 등 셸 타입)
    OptiStruct  : GRID + CQUAD4/CTRIA3 등 (PSHELL·MAT·SPC 불필요)
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def load(self, mesh_file: Path) -> tuple:
        """
        Returns (inp_path_in_workspace, elset_name)
        """
        mesh_file = Path(mesh_file)
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        suffix = mesh_file.suffix.lower()
        if suffix not in _SUPPORTED_EXT:
            raise ValueError(f"Unsupported format '{suffix}'. Supported: {sorted(_SUPPORTED_EXT)}")

        print(f"[Loader] Reading {suffix.upper()} : {mesh_file.name}")
        mesh = meshio.read(str(mesh_file))

        out_path = self.workspace / (mesh_file.stem + '_mesh.inp')
        n_elem = self._write_inp(mesh, out_path)
        print(f"[Loader] {len(mesh.points)} nodes, {n_elem} elements → {out_path.name}")
        return out_path, ELSET_NAME

    # ------------------------------------------------------------------

    def _write_inp(self, mesh: meshio.Mesh, out_path: Path) -> int:
        points = mesh.points
        if points.shape[1] == 2:                          # 2D → 3D 좌표 보장
            points = np.hstack([points, np.zeros((len(points), 1))])

        surface_blocks = [b for b in mesh.cells if b.type in _SURFACE_TYPES]
        if not surface_blocks:
            found = [b.type for b in mesh.cells]
            raise ValueError(f"No surface elements found. Types in file: {found}")

        with open(out_path, 'w', encoding='utf-8') as f:
            # Nodes (meshio 0-based → CalculiX 1-based)
            f.write("*NODE\n")
            for i, (x, y, z) in enumerate(points):
                f.write(f"{i+1}, {x:.10g}, {y:.10g}, {z:.10g}\n")

            # Elements (타입별 블록, 모두 같은 ELSET으로)
            eid = 1
            for block in surface_blocks:
                ccx_type = _CCX_TYPE[block.type]
                f.write(f"\n*ELEMENT, TYPE={ccx_type}, ELSET={ELSET_NAME}\n")
                for conn in block.data:
                    node_ids = ', '.join(str(int(n) + 1) for n in conn)
                    f.write(f"{eid}, {node_ids}\n")
                    eid += 1

        return eid - 1
