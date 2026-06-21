# -*- coding: utf-8 -*-
import gmsh
from pathlib import Path
from src.core.config import TrayGeometryConfig


class GmshMesher:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def create_tray_geometry_and_mesh(self, geo: TrayGeometryConfig = None):
        if geo is None:
            geo = TrayGeometryConfig()

        gmsh.initialize()
        gmsh.model.add("tray_model")

        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(geo.length, 0, 0)
        p3 = gmsh.model.geo.addPoint(geo.length, geo.width, 0)
        p4 = gmsh.model.geo.addPoint(0, geo.width, 0)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([cl1])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], name=geo.elset_name)

        gmsh.option.setNumber("Mesh.MeshSizeMax", geo.mesh_size)
        gmsh.model.mesh.generate(2)

        return geo.elset_name

    def export_mesh_inp(self, output_filename: str = "mesh.inp", show_gui: bool = False) -> Path:
        out_path = self.workspace / output_filename
        gmsh.write(str(out_path))

        with open(out_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # *HEADING 블록 제거 (Master INP와 충돌 방지)
        cleaned_lines = []
        in_heading = False
        for line in lines:
            if line.strip().upper().startswith('*HEADING'):
                in_heading = True
                continue
            if in_heading:
                if line.strip().startswith('*'):
                    in_heading = False
                else:
                    continue
            cleaned_lines.append(line)

        content = "".join(cleaned_lines)

        # Gmsh 평면 요소(CPS/CPE) → CalculiX Shell 요소(S) 치환
        for src, dst in [("CPS3", "S3"), ("CPE3", "S3"),
                         ("CPS4", "S4"), ("CPE4", "S4"),
                         ("CPS8", "S8"), ("CPE8", "S8")]:
            content = content.replace(f"type={src}", f"type={dst}")

        # Gmsh 후행 콤마 제거
        content = content.rstrip(", \n\r") + "\n"

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)

        if show_gui:
            gmsh.fltk.run()
        gmsh.finalize()
        return out_path
