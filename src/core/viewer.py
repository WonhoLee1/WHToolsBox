# -*- coding: utf-8 -*-
import pyvista as pv
import numpy as np
from pathlib import Path


class ModeShapeViewer:
    def show_mode(self, vtu_base: Path, mode_number: int = 1):
        """
        ccx2paraview가 생성한 VTU 파일에서 특정 모드를 시각화합니다.
        vtu_base: FRD 경로에서 확장자를 제거한 경로 (e.g. workspace/test_tray)
        mode_number: 1-based 모드 번호
        """
        # ccx2paraview는 1-based 2자리 zero-padding으로 파일 생성: <base>.01.vtu, <base>.02.vtu, ...
        vtu_file = Path(f"{vtu_base}.{mode_number:02d}.vtu")
        if not vtu_file.exists():
            vtu_files = sorted(Path(vtu_base).parent.glob(f"{Path(vtu_base).name}.*.vtu"))
            print(f"[Error] VTU file not found: {vtu_file}")
            print(f"  Available: {[f.name for f in vtu_files]}")
            return

        mesh = pv.read(str(vtu_file))

        # 변위 배열 탐색 (ccx2paraview 출력명 후보: 'U', 'DISP', 'Displacements')
        disp_key = next(
            (k for k in mesh.point_data.keys() if k.upper() in ('U', 'DISP', 'DISPLACEMENTS')),
            None
        )
        if disp_key is None:
            print(f"[Error] Displacement array not found. Available: {list(mesh.point_data.keys())}")
            return

        disp = mesh.point_data[disp_key]
        if disp.ndim == 1:
            disp = disp.reshape(-1, 1)

        # 3성분 변위 벡터 추출
        uvw = disp[:, :3] if disp.shape[1] >= 3 else np.hstack([disp, np.zeros((len(disp), 3 - disp.shape[1]))])
        max_disp = np.max(np.linalg.norm(uvw, axis=1))

        bounds = mesh.bounds
        max_length = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        factor = (max_length * 0.1) / max_disp if max_disp > 0 else 1.0

        warped = mesh.warp_by_vector(disp_key, factor=factor)

        plotter = pv.Plotter()
        plotter.add_mesh(warped, scalars=disp_key, show_edges=True, cmap="jet")
        plotter.add_text(f"Modal Analysis - Mode {mode_number}", font_size=14, color="black")
        plotter.show()
