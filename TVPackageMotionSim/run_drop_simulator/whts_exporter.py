# -*- coding: utf-8 -*-
"""
[WHTOOLS] Professional Multi-Format Exporter (Cleaned & Fixed v6.4)
구조 해석 결과를 VTKHDF 및 GLB 포맷으로 완벽하게 내보내고 ParaView 대시보드를 자동화합니다.
"""

import os
import sys
import numpy as np
import pyvista as pv
from typing import List, Dict, Any, Optional
import h5py
import subprocess
import glob

# [WHTOOLS] UTF-8 인코딩 강제 설정
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

class WHToolsExporter:
    """
    [WHTOOLS] Analysis Result Exporter
    해석 결과를 ParaView(VTKHDF) 및 프레젠테이션용 GLB로 저장합니다.
    """
    def __init__(self, manager: Any):
        self.manager = manager
        self.times = manager.times
        self.n_frames = len(self.times)
        self.last_export_dir = None

    def export_summary(self):
        """내보내기 완료 요약 리포트 (WHTOOLS 스타일)"""
        print("\n" + "="*80)
        print("📦 [WHTOOLS] EXPORT COMPLETE")
        print("="*80)
        print("1. VTKHDF: Unified transient data for ParaView.")
        print("2. GLB (glTF): Premium Global 3D assets for presentations.")
        print("3. Auto-Dashboard: ParaView visualization engine launched.")
        print("="*80 + "\n")

    def export_to_glb(self, output_dir: str, frame_idx: int = -1):
        """[WHTOOLS] 글로벌 좌표계 기반 고품질 GLB 내보내기"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\n[WHTOOLS] Exporting Global GLB (Premium) to: {output_dir}")
        
        for analyzer in self.manager.analyzers:
            if not analyzer.results or 'Displacement [mm]' not in analyzer.results:
                continue
            
            target_idx = frame_idx
            if target_idx == -1:
                # 최대 변위 프레임 자동 탐색
                disp_abs = np.abs(analyzer.results['Displacement [mm]'])
                target_idx = np.argmax(np.max(disp_abs, axis=(1, 2)))
            
            res = analyzer.sol.res
            X_2d, Y_2d = np.array(analyzer.sol.X_mesh), np.array(analyzer.sol.Y_mesh)
            w_disp = analyzer.results['Displacement [mm]'][target_idx]
            
            ref_basis = np.array(analyzer.ref_basis)
            ref_center = np.array(analyzer.ref_center)
            R_rigid = analyzer.results['R'][target_idx]
            c_P = analyzer.results['c_P'][target_idx]
            c_Q = analyzer.results['c_Q'][target_idx]
            
            P_local = np.column_stack([X_2d.ravel(), Y_2d.ravel(), w_disp.ravel()])
            P_global = (P_local @ ref_basis + ref_center - c_P) @ R_rigid + c_Q
            
            grid = pv.StructuredGrid(
                P_global[:, 0].reshape(res, res),
                P_global[:, 1].reshape(res, res),
                P_global[:, 2].reshape(res, res)
            )
            
            vm_stress = analyzer.results['Von-Mises [MPa]'][target_idx]
            grid.point_data['Von-Mises [MPa]'] = vm_stress.ravel()
            
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars='Von-Mises [MPa]', cmap='jet', show_scalar_bar=True)
            
            file_name = f"{analyzer.name}_Global_MaxStress.glb"
            file_path = os.path.join(output_dir, file_name)
            try:
                plotter.export_gltf(file_path)
                print(f"  > Exported Global GLB: {file_name}")
            except Exception as e:
                print(f"  > Failed to export Global GLB: {e}")

    def export_to_vtkhdf(self, output_dir: str, filename: str = "Result.vtkhdf"):
        """[WHTOOLS] VTKHDF 1.0 Transient Unified Mesh Export"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.last_export_dir = output_dir
        filepath = os.path.join(output_dir, filename)
        
        if not self.manager.analyzers:
            print("⚠️ [Export Warning] No analyzers found. Skipping VTKHDF export.")
            return None
        
        print(f"\n[WHTOOLS] Exporting COMPLIANT VTKHDF (Transient) to: {filepath}")
        
        # 1. Topology Generation
        all_points_f0, all_conn, all_offsets, all_types, all_part_ids = [], [], [0], [], []
        point_offset = 0
        
        for p_idx, analyzer in enumerate(self.manager.analyzers):
            res = analyzer.sol.res
            num_pts = res * res
            quads = []
            for i in range(res - 1):
                for j in range(res - 1):
                    quads.append([i*res+j, i*res+j+1, (i+1)*res+j+1, (i+1)*res+j])
            
            quads = np.array(quads) + point_offset
            X_2d, Y_2d = np.array(analyzer.sol.X_mesh), np.array(analyzer.sol.Y_mesh)
            ref_basis, ref_center = np.array(analyzer.ref_basis), np.array(analyzer.ref_center)
            P_world0 = np.column_stack([X_2d.ravel(), Y_2d.ravel(), np.zeros(num_pts)]) @ ref_basis + ref_center
            
            all_points_f0.append(P_world0)
            all_conn.extend(quads.ravel())
            for _ in range(len(quads)):
                all_offsets.append(all_offsets[-1] + 4)
                all_types.append(9) 
            all_part_ids.extend([p_idx] * num_pts)
            point_offset += num_pts
            
        total_points, total_cells, total_conn_ids = point_offset, len(all_types), len(all_conn)
        
        with h5py.File(filepath, 'w') as f:
            vtkhdf = f.create_group("VTKHDF")
            vtkhdf.attrs.create("Version", [1, 0], dtype='i4')
            vtkhdf.attrs.create("Type", np.bytes_("UnstructuredGrid"))
            
            # [WHTOOLS] Transient Metadata (GZIP Compression for Stability)
            vtkhdf.create_dataset("Connectivity", data=np.tile(np.array(all_conn, dtype=np.int64), self.n_frames), compression="gzip", compression_opts=4)
            vtkhdf.create_dataset("Types", data=np.tile(np.array(all_types, dtype=np.uint8), self.n_frames), compression="gzip", compression_opts=4)
            
            f_offs = [np.array(all_offsets[:-1], dtype=np.int64) + t*total_conn_ids for t in range(self.n_frames)]
            f_offs.append(np.array([self.n_frames * total_conn_ids], dtype=np.int64))
            vtkhdf.create_dataset("Offsets", data=np.concatenate(f_offs), compression="gzip", compression_opts=4)
            
            vtkhdf.create_dataset("NumberOfPoints", data=np.full(self.n_frames, total_points, dtype=np.int64))
            vtkhdf.create_dataset("NumberOfCells", data=np.full(self.n_frames, total_cells, dtype=np.int64))
            vtkhdf.create_dataset("NumberOfConnectivityIds", data=np.full(self.n_frames, total_conn_ids, dtype=np.int64))
            
            steps = vtkhdf.create_group("Steps")
            steps.attrs.create("NSteps", self.n_frames, dtype='i8')
            steps.create_dataset("Values", data=np.array(self.times, dtype='f4'))
            steps.create_dataset("PointOffsets", data=np.arange(self.n_frames + 1, dtype=np.int64) * total_points)
            steps.create_dataset("CellOffsets", data=np.arange(self.n_frames + 1, dtype=np.int64) * total_cells)
            steps.create_dataset("ConnectivityOffsets", data=np.arange(self.n_frames + 1, dtype=np.int64) * total_conn_ids)
            # [WHTOOLS] [v6.9d] PartOffsets (Required for temporal data in some ParaView versions)
            steps.create_dataset("PartOffsets", data=np.zeros(self.n_frames + 1, dtype=np.int64))
            
            ds_points = vtkhdf.create_dataset("Points", shape=(self.n_frames * total_points, 3), dtype='f4', compression="gzip", compression_opts=4, chunks=(total_points, 3))
            pd = vtkhdf.create_group("PointData")
            ds_disp = pd.create_dataset("displacement_vec", shape=(self.n_frames * total_points, 3), dtype='f4', compression="gzip", compression_opts=4, chunks=(total_points, 3))
            ds_vm = pd.create_dataset("Von-Mises [MPa]", shape=(self.n_frames * total_points,), dtype='f4', compression="gzip", compression_opts=4, chunks=(total_points,))
            pd.create_dataset("PartID", data=np.tile(np.array(all_part_ids, dtype='i4'), self.n_frames), compression="gzip", compression_opts=4)
            
            print("  > Streaming Transient Data...")
            max_vm = 0.0
            for t in range(self.n_frames):
                frm_pts, frm_dsp, frm_vm = [], [], []
                for az in self.manager.analyzers:
                    res_m = az.sol.res
                    X, Y = np.array(az.sol.X_mesh), np.array(az.sol.Y_mesh)
                    rb, rc = np.array(az.ref_basis), np.array(az.ref_center)
                    w = np.nan_to_num(az.results['Displacement [mm]'][t], nan=0.0) if 'Displacement [mm]' in az.results else np.zeros((res_m, res_m))
                    vm = np.nan_to_num(az.results['Von-Mises [MPa]'][t], nan=0.0) if 'Von-Mises [MPa]' in az.results else np.zeros((res_m, res_m))
                    R, cP, cQ = az.results['R'][t], az.results['c_P'][t], az.results['c_Q'][t]
                    
                    P_g = (np.column_stack([X.ravel(), Y.ravel(), w.ravel()]) @ rb + rc - cP) @ R + cQ
                    P_w0 = np.column_stack([X.ravel(), Y.ravel(), np.zeros(res_m*res_m)]) @ rb + rc
                    frm_pts.append(P_g)
                    frm_dsp.append(P_g - P_w0)
                    frm_vm.append(vm.ravel())
                    max_vm = max(max_vm, float(np.max(vm)))
                
                s, e = t*total_points, (t+1)*total_points
                # [WHTOOLS] [v6.9a] ParaView 안정성을 위한 최종 수치 클리핑
                final_pts = np.concatenate(frm_pts)
                final_dsp = np.clip(np.concatenate(frm_dsp), -500.0, 500.0)
                final_vm = np.clip(np.concatenate(frm_vm), 0.0, 10000.0)
                
                ds_points[s:e], ds_disp[s:e], ds_vm[s:e] = final_pts, final_dsp, final_vm
            
            ds_vm.attrs.create("max", max_vm)
        return filepath

    def launch_paraview(self, vtkhdf_path: str):
        """[WHTOOLS] ParaView를 자동 실행하며 전용 대시보드 스크립트를 주입합니다."""
        pv_exe = r"C:\Program Files\ParaView 6.0.1\bin\paraview.exe"
        if not os.path.exists(pv_exe): pv_exe = "paraview"
            
        filename = os.path.basename(vtkhdf_path)
        script_path = vtkhdf_path + "_dashboard.py"
        
        # [WHTOOLS] [FIX] UnicodeEscape Error 방지를 위해 슬래시(/) 치환
        safe_path = vtkhdf_path.replace("\\", "/")
        
        script_content = f"""# -*- coding: utf-8 -*-
from paraview.simple import *
# 1. Load Data
data = OpenDataFile('{safe_path}')
# 2. Setup Dashboard
layout = GetLayout()
SetActiveView(None)
r_view = GetActiveViewOrCreate('RenderView')
warp = WarpByVector(Input=data)
warp.Vectors = ['POINTS', 'displacement_vec']
warp_disp = Show(warp, r_view)
ColorBy(warp_disp, ('POINTS', 'Von-Mises [MPa]'))
warp_disp.SetScalarBarVisibility(r_view, True)

# Chart (Native Calculation)
stats = DescriptiveStatistics(Input=data)
# [WHTOOLS] [v6.9d] Variables -> ModelVariables (ParaView 5.10+ / 6.0 API 대응)
if hasattr(stats, 'ModelVariables'):
    stats.ModelVariables = ['Von-Mises [MPa]']
else:
    stats.Variables = ['Von-Mises [MPa]']
plot = PlotGlobalVariablesOverTime(Input=stats)
c_view = CreateView('XYChartView')
plot_disp = Show(plot, c_view)
plot_disp.SeriesVisibility = ['Maximum']
layout.SplitHorizontal(0, 0.4)
layout.AssignView(1, r_view)
layout.AssignView(2, c_view)
RenderAllViews()
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
            
        print(f"🚀 [WHTOOLS] Launching ParaView Dashboard: {filename}")
        try:
            subprocess.Popen([pv_exe, "--script=" + script_path])
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
# [WHTOOLS] Macro for unified viewer
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
