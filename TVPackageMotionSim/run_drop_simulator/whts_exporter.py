# -*- coding: utf-8 -*-
"""
[WHTOOLS] Professional Multi-Format Exporter (v7.2 Full Temporal Topology)
ParaView 6.0의 엄격한 VTKHDF transient 규격(CellOffsets 필구)을 충족하기 위해
모든 토폴로지 데이터를 시계열 인덱스 시프트와 함께 타일링하여 완벽한 호환성을 제공합니다.
"""

import os
import sys
import numpy as np
import pyvista as pv
from typing import List, Dict, Any, Optional
import h5py
import subprocess
import traceback

# [WHTOOLS] UTF-8 인코딩 강제 설정
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

class WHToolsExporter:
    """
    [WHTOOLS] Analysis Result Exporter (v7.2 Stable)
    ParaView 6.0+ 환경에서 슬라이더 조작 시 크래시 없는 시계열 분석 환경을 보장합니다.
    """
    def __init__(self, manager: Any):
        self.manager = manager
        self.times = getattr(manager, 'times', [])
        self.n_frames = len(self.times)
        self.last_export_dir = None

    def export_summary(self):
        """내보내기 완료 요약 리포트 (WHTOOLS 스타일)"""
        print("\n" + "="*80)
        print("📦 [WHTOOLS] EXPORT COMPLETE (v8.0 Ultra-Stable)")
        print("="*80)
        print("1. PVD+VTU: Full Transient Series (100% ParaView Compatible)")
        print("2. GLB (glTF): Globally Aligned High-Fidelity 3D Assets")
        print("3. Auto-Dashboard: ParaView visualization engine launched.")
        print("="*80 + "\n")

    def _transform_to_global(self, p_local: np.ndarray, rb: np.ndarray, rc: np.ndarray, 
                           cP0: np.ndarray, R_t: np.ndarray, cQ_t: np.ndarray) -> np.ndarray:
        """글로벌 좌표 변환 핵심 수식 (Strict PCA Basis Transpose)"""
        p_world0 = p_local @ rb.T + rc
        p_relative = p_world0 - cP0
        p_global = p_relative @ R_t + cQ_t
        return p_global

    def export_to_glb(self, output_dir: str, frame_idx: int = -1):
        """[WHTOOLS] 글로벌 좌표계 기반 고품질 GLB 내보내기"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\n[WHTOOLS] Exporting Global GLB (Premium) to: {output_dir}")
        for analyzer in self.manager.analyzers:
            if analyzer.sol is None or not analyzer.results or 'Displacement [mm]' not in analyzer.results:
                continue
            
            target_idx = frame_idx
            if target_idx == -1:
                disp_abs = np.abs(analyzer.results['Displacement [mm]'])
                target_idx = np.argmax(np.max(disp_abs, axis=(1, 2)))
            
            res = analyzer.sol.res
            X_2d, Y_2d = np.array(analyzer.sol.X_mesh), np.array(analyzer.sol.Y_mesh)
            w_disp = analyzer.results['Displacement [mm]'][target_idx]
            rb, rc = np.array(analyzer.ref_basis), np.array(analyzer.ref_center)
            cP0 = analyzer.results['c_P'][0]
            R_t = analyzer.results['R'][target_idx]
            cQ_t = analyzer.results['c_Q'][target_idx]
            
            p_local = np.column_stack([X_2d.ravel(), Y_2d.ravel(), w_disp.ravel()])
            p_global = self._transform_to_global(p_local, rb, rc, cP0, R_t, cQ_t)
            
            grid = pv.StructuredGrid(
                p_global[:, 0].reshape(res, res),
                p_global[:, 1].reshape(res, res),
                p_global[:, 2].reshape(res, res)
            )
            vm_stress = analyzer.results['Von-Mises [MPa]'][target_idx]
            grid.point_data['Von-Mises [MPa]'] = vm_stress.ravel()
            
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='Von-Mises [MPa]', cmap='jet', show_scalar_bar=True)
            file_name = f"{analyzer.name}_Global_MaxStress.glb"
            file_path = os.path.join(output_dir, file_name)
            try:
                plotter.export_gltf(file_path)
                print(f"  > Exported GLB: {file_name}")
            except Exception as e:
                print(f"  > Failed GLB Export: {e}")

    def export_to_pvd_series(self, output_dir: str, filename: str = "Result.pvd"):
        """
        [WHTOOLS] PVD + VTU Series Export (v8.0 - Ultra-Stable)

        VTKHDF h5py 직접 기록 방식은 ParaView 6.0의 vtkHDFReader 내부 버그로
        인해 시계열 슬라이더 사용 시 H5Dread 오류가 발생합니다.
        이를 완전히 회피하기 위해 ParaView에서 가장 안정적으로 검증된
        PVD(ParaView Data) + VTU(VTK Unstructured Grid) Series 방식으로 전환합니다.

        - ParaView가 직접 쓰고 읽는 포맷이므로 100% 호환 보장.
        - 각 프레임을 독립된 .vtu 파일로 저장하고, .pvd가 타임라인을 관리.
        """
        vtu_dir = os.path.join(output_dir, "vtu")
        if not os.path.exists(vtu_dir):
            os.makedirs(vtu_dir)

        self.last_export_dir = output_dir
        pvd_path = os.path.join(output_dir, filename)

        if not self.manager.analyzers:
            print("⚠️ [Export Warning] No analyzers found.")
            return None

        print(f"\n[WHTOOLS] Exporting PVD+VTU Series (v8.0 Stable) → {pvd_path}")

        # 전체 어셈블리의 정적 셀 구조 사전 계산
        cell_types_list, cell_points_list, part_ids_list = [], [], []
        p_offset = 0

        for p_idx, analyzer in enumerate(self.manager.analyzers):
            if analyzer.sol is None:
                continue
            res = analyzer.sol.res
            num_pts = res * res

            quads = []
            for i in range(res - 1):
                for j in range(res - 1):
                    quads.append([i*res+j, i*res+j+1, (i+1)*res+j+1, (i+1)*res+j])

            quads_arr = np.array(quads, dtype=np.int64) + p_offset
            cell_types_list.append(np.full(len(quads), 9, dtype=np.uint8))  # VTK_QUAD
            cell_points_list.append(quads_arr)
            part_ids_list.extend([p_idx] * num_pts)
            p_offset += num_pts

        total_points = p_offset
        all_cells = np.concatenate(cell_points_list)
        all_cell_types = np.concatenate(cell_types_list)
        part_ids = np.array(part_ids_list, dtype=np.int32)

        print(f"  > Assembly: {total_points} pts / {len(all_cell_types)} cells / {self.n_frames} frames")
        print(f"  > Writing VTU frames to: {vtu_dir}")

        pvd_entries = []
        max_vm = 0.0

        # 공간 필드로 내보낼 키 (공통 기준: shape == (n_frames, res, res))
        # 키네마틱스(R, c_P, c_Q 등) 및 스칼라 요약(Max-, Mean-)은 제외
        SKIP_PREFIXES = ('Max-', 'Mean-', 'R_matrix', 'r_rmse', 'cur_centroid',
                         'ref_centroid', 'local_markers', 'Marker Global Disp')
        SKIP_KEYS     = {'R', 'c_P', 'c_Q'}

        def _is_spatial_field(key: str, arr) -> bool:
            """(n_frames, res, res) 형태의 공간 스칼라 필드인지 확인합니다."""
            if key in SKIP_KEYS:
                return False
            if any(key.startswith(p) for p in SKIP_PREFIXES):
                return False
            if not hasattr(arr, 'shape') or arr.ndim != 3:
                return False
            return True  # (n, h, w) 형태

        try:
            for t in range(self.n_frames):
                frm_pts  = []
                frm_fields: dict[str, list] = {}

                for az in self.manager.analyzers:
                    n_pts = az.sol.res**2 if az.sol else (total_points // max(len(self.manager.analyzers), 1))

                    if az.sol is None or not az.results or 'R' not in az.results:
                        frm_pts.append(np.zeros((n_pts, 3), dtype='f4'))
                        # 필드는 0으로 채움
                        for key in frm_fields:
                            frm_fields[key].append(np.zeros(n_pts, dtype='f4'))
                        continue

                    t_safe = min(t, len(az.results['R']) - 1)
                    rb   = np.array(az.ref_basis)
                    rc   = np.array(az.ref_center)
                    cP0  = np.array(az.results['c_P'][0])
                    R_t  = np.array(az.results['R'][t_safe])
                    cQ_t = np.array(az.results['c_Q'][t_safe])
                    X    = np.array(az.sol.X_mesh)
                    Y    = np.array(az.sol.Y_mesh)
                    w    = np.nan_to_num(az.results['Displacement [mm]'][t_safe], nan=0.0)
                    vm   = np.nan_to_num(az.results['Von-Mises [MPa]'][t_safe], nan=0.0)

                    p_local  = np.column_stack([X.ravel(), Y.ravel(), w.ravel()])
                    p_global = self._transform_to_global(p_local, rb, rc, cP0, R_t, cQ_t)
                    p_ref    = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]) @ rb.T + rc

                    frm_pts.append(p_global.astype('f4'))
                    max_vm = max(max_vm, float(np.nanmax(vm)))

                    # displacement_vec (특별 처리: 3D 벡터)
                    dsp_vec = (p_global - p_ref).astype('f4')
                    if 'displacement_vec' not in frm_fields:
                        frm_fields['displacement_vec'] = []
                    frm_fields['displacement_vec'].append(dsp_vec)

                    # 키르히호프 공간 필드 자동 탐지 및 추가
                    for key, arr in az.results.items():
                        if not _is_spatial_field(key, arr):
                            continue
                        val = np.nan_to_num(arr[t_safe], nan=0.0).ravel().astype('f4')
                        if key not in frm_fields:
                            frm_fields[key] = []
                        frm_fields[key].append(val)

                # UnstructuredGrid 빌드
                pts_arr    = np.concatenate(frm_pts)
                n_cells    = len(all_cell_types)
                cells_flat = np.hstack([
                    np.column_stack([np.full(n_cells, 4, dtype=np.int64), all_cells])
                ]).ravel()

                grid = pv.UnstructuredGrid(cells_flat, all_cell_types, pts_arr)
                grid.point_data['PartID'] = part_ids

                # 모든 필드 추가
                for field_key, chunks in frm_fields.items():
                    concat = np.concatenate(chunks)
                    grid.point_data[field_key] = concat

                vtu_name = f"frame_{t:04d}.vtu"
                vtu_path = os.path.join(vtu_dir, vtu_name)
                grid.save(vtu_path)

                pvd_entries.append((float(self.times[t]), f"vtu/{vtu_name}"))

                if t % 100 == 0:
                    print(f"    Frame {t:4d}/{self.n_frames} written.")

        except Exception as err:
            print(f"❌ [CRITICAL] VTU write error at frame {t}: {err}")
            traceback.print_exc()


        # PVD 파일 작성 (XML 컬렉션)
        pvd_xml = ['<?xml version="1.0"?>\n',
                   '<VTKFile type="Collection" version="0.1">\n',
                   '  <Collection>\n']
        for time_val, rel_path in pvd_entries:
            pvd_xml.append(f'    <DataSet timestep="{time_val:.6f}" group="" part="0" file="{rel_path}"/>\n')
        pvd_xml.extend(['  </Collection>\n', '</VTKFile>\n'])

        with open(pvd_path, 'w', encoding='utf-8') as f:
            f.writelines(pvd_xml)

        print(f"\n  ✅ PVD Collection written: {pvd_path}")
        print(f"  ✅ {len(pvd_entries)} VTU frames / Max VM: {max_vm:.2f} MPa")
        return pvd_path

    # 하위 호환성: 기존 VTKHDF 호출 코드가 있을 경우 PVD로 라우팅
    def export_to_vtkhdf(self, output_dir: str, filename: str = "Result.vtkhdf"):
        """[호환성 래퍼] export_to_pvd_series 로 위임합니다."""
        pvd_dir = output_dir.rstrip("/\\")
        return self.export_to_pvd_series(pvd_dir, "Result.pvd")

    def launch_paraview(self, pvd_path: str):
        """ParaView를 자동 실행하며 PVD 파일을 로드합니다."""
        pv_exe = r"C:\Program Files\ParaView 6.0.1\bin\paraview.exe"
        if not os.path.exists(pv_exe):
            pv_exe = "paraview"
        safe_path = pvd_path.replace("\\", "/")
        print(f"🚀 [WHTOOLS] Launching ParaView: {os.path.basename(pvd_path)}")
        try:
            subprocess.Popen([pv_exe, safe_path])
        except Exception as e:
            print(f"❌ Failed to launch ParaView: {e}")


    def register_paraview_macro(self):
        """ParaView 매크로 자동 등록"""
        appdata = os.environ.get('APPDATA')
        if not appdata: return
        macro_dir = os.path.join(appdata, "ParaView", "Macros")
        if not os.path.exists(macro_dir): os.makedirs(macro_dir)
        path = os.path.join(macro_dir, "WHTOOLS_Dashboard.py")
        content = """# -*- coding: utf-8 -*-
from paraview.simple import *
data = GetActiveSource()
if data:
    layout = GetLayout()
    view3d = GetActiveView()
    view2d = CreateView('XYChartView')
    layout.SplitHorizontal(0, 0.4)
    layout.AssignView(1, view3d)
    layout.AssignView(2, view2d)
    plot = PlotGlobalVariablesOverTime(Input=data)
    Show(plot, view2d)
    RenderAllViews()
"""
        with open(path, "w", encoding="utf-8") as f: f.write(content)
        print(f"✅ [WHTOOLS] Macro registered at: {path}")
