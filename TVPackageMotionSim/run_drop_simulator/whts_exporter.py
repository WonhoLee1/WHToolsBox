# -*- coding: utf-8 -*-
"""
[WHTOOLS] Professional Multi-Format Exporter
구조 해석 결과를 VTKHDF 및 GLB(glTF) 포맷으로 전문적으로 내보내는 모듈입니다.
"""

import os
import sys
import numpy as np
import pyvista as pv
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
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
    PlateAssemblyManager의 해석 결과를 다양한 포맷으로 저장합니다.
    """
    def __init__(self, manager: Any):
        self.manager = manager
        self.times = manager.times
        self.n_frames = len(self.times)
        self.last_export_dir = None

    def export_all_to_vtk(self, output_dir: str):
        """
        모든 파트의 결과를 ParaView 호환 포맷(.vts)으로 내보내고 PVD 컬렉션을 생성합니다.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.last_export_dir = output_dir
        print(f"\n[WHTOOLS] Exporting ParaView Files (.vts + .pvd) to: {output_dir}")
        print(f"  > Total Analyzers: {len(self.manager.analyzers)}")
        
        all_parts_files = [] # 마스터 PVD용
        
        for idx, analyzer in enumerate(self.manager.analyzers):
            if not analyzer.results:
                print(f"    ! No results found for {analyzer.name}")
                continue
            
            file_path = os.path.join(output_dir, f"{analyzer.name}.vts")
            frame_files = self._save_analyzer_to_vts(analyzer, file_path)
            
            # 부품별 PVD 생성
            pvd_path = os.path.join(output_dir, f"{analyzer.name}.pvd")
            self._write_pvd_file(pvd_path, frame_files)
            all_parts_files.append((idx, frame_files))

        # 마스터 PVD 생성 (전체 어셈블리 한꺼번에 열기용)
        master_pvd_path = os.path.join(output_dir, "_Assembly_Full_Sequence.pvd")
        self._write_master_pvd(master_pvd_path, all_parts_files)

    def _save_analyzer_to_vts(self, analyzer: Any, file_path: str) -> list:
        res = analyzer.sol.res
        X_2d, Y_2d = np.array(analyzer.sol.X_mesh), np.array(analyzer.sol.Y_mesh)
        grid = pv.StructuredGrid(X_2d, Y_2d, np.zeros_like(X_2d))
        
        base_name, _ = os.path.splitext(file_path)
        step = max(1, self.n_frames // 10)
        
        ref_basis = np.array(analyzer.ref_basis)
        ref_center = np.array(analyzer.ref_center)
        
        saved_files = []
        
        for i in range(0, self.n_frames, step):
            w_disp = analyzer.results['Displacement [mm]'][i]
            P_local = np.column_stack([X_2d.ravel(), Y_2d.ravel(), w_disp.ravel()])
            P_init = P_local @ ref_basis + ref_center
            
            R_rigid = analyzer.results['R'][i]
            c_P = analyzer.results['c_P'][i]
            c_Q = analyzer.results['c_Q'][i]
            P_global = (P_init - c_P) @ R_rigid + c_Q
            
            grid.points = P_global
            
            for field_name, field_data in analyzer.results.items():
                if isinstance(field_data, np.ndarray) and field_data.shape == (self.n_frames, res, res):
                    grid.point_data[field_name] = field_data[i].ravel()
            
            frame_rel_path = f"{os.path.basename(base_name)}_{i:04d}.vts"
            frame_full_path = os.path.join(os.path.dirname(file_path), frame_rel_path)
            grid.save(frame_full_path)
            
            # PVD 기록용 정보 (시간, 상대경로)
            saved_files.append({"time": float(self.times[i]), "file": frame_rel_path})
            
        return saved_files

    def _write_pvd_file(self, pvd_path: str, frame_files: list):
        """단일 부품 시계열 관리를 위한 PVD 작성"""
        with open(pvd_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            for frame in frame_files:
                f.write(f'    <DataSet timestep="{frame["time"]:.6f}" group="" part="0" file="{frame["file"]}"/>\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')

    def _write_master_pvd(self, pvd_path: str, all_parts_files: list):
        """전체 어셈블리 통합 시계열 관리를 위한 Master PVD 작성"""
        with open(pvd_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            
            # 모든 시간 단계를 수집하여 정렬
            all_times = sorted(list(set(f['time'] for _, files in all_parts_files for f in files)))
            
            for t in all_times:
                for part_idx, frame_files in all_parts_files:
                    # 해당 시간에 맞는 프레임 찾기 (근사치 가능)
                    best_match = min(frame_files, key=lambda x: abs(x['time'] - t))
                    if abs(best_match['time'] - t) < 1e-7:
                        f.write(f'    <DataSet timestep="{t:.6f}" group="" part="{part_idx}" file="{best_match["file"]}"/>\n')
            
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')

    def export_to_glb(self, output_dir: str, frame_idx: int = -1):
        """
        [WHTOOLS] 글로벌 좌표계 기반 고품질 GLB 내보내기
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\n[WHTOOLS] Exporting Global GLB (Premium) to: {output_dir}")
        
        for analyzer in self.manager.analyzers:
            if not analyzer.results: continue
            
            target_idx = frame_idx
            if target_idx == -1:
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

    def export_summary(self):
        """내보내기 완료 요약 리포트 (WHTOOLS 스타일)"""
        print("\n" + "="*80)
        print("📦 [WHTOOLS] EXPORT COMPLETE")
        print("="*80)
        print("1. VTK Sequence (.vts) & Time-Series Collection (.pvd)")
        print("   > Tip: Open '_Assembly_Full_Sequence.pvd' in ParaView for full animation.")
        print("2. GLB (glTF): Premium Global 3D assets for presentations.")
        print("="*80)

    def launch_paraview(self, output_dir: str = None):
        """
        [WHTOOLS] 내보내진 결과물을 ParaView에서 자동으로 엽니다. (VTKHDF + Auto Dashboard)
        """
        target_dir = output_dir or self.last_export_dir
        if not target_dir or not os.path.exists(target_dir):
            print("⚠️ Launch failed: No valid export directory found.")
            return

        # 1. VTKHDF 우선 로드
        hdf_files = glob.glob(os.path.join(target_dir, "*.vtkhdf"))
        target_file = hdf_files[0] if hdf_files else None
        
        if not target_file:
            # Fallback to PVD
            pvd_files = glob.glob(os.path.join(target_dir, "*.pvd"))
            target_file = pvd_files[0] if pvd_files else None
            
        if not target_file:
            print("⚠️ No valid data files found to launch ParaView.")
            return

        pv_paths = [
            r"C:\Program Files\ParaView 6.0.1\bin\paraview.exe",
            r"C:\Program Files\ParaView 5.12.1\bin\paraview.exe",
            "paraview", 
        ]
        
        pv_exe = None
        for p in pv_paths:
            if p == "paraview" or os.path.exists(p):
                pv_exe = p
                break
        
        if not pv_exe:
            print("⚠️ ParaView executable not found.")
            return

        # --- Dashboard 자동화 스크립트 작성 (User 노하우 반영) ---
        script_path = os.path.join(target_dir, "whts_auto_dashboard.py")
        abs_target = os.path.abspath(target_file).replace("\\", "/")
        
        script_content = f"""
from paraview.simple import *
# 1. Load Data
reader = OpenDataFile('{abs_target}')
if not reader:
    print('Failed to open file: {abs_target}')
else:
    UpdatePipeline()
    # 2. Setup Render View (3D)
    view3d = GetActiveViewOrCreate('RenderView')
    view3d.Background = [0.1, 0.1, 0.15] # Elegant Dark Mode
    
    # WarpByVector (if displacement exists)
    warp = WarpByVector(Input=reader)
    warp.Vectors = ['POINTS', 'displacement_vec']
    warp.ScaleFactor = 2.0 
    
    display3d = Show(warp, view3d)
    display3d.Representation = 'Surface With Edges'
    
    # 3. Coloring (Von-Mises)
    ColorBy(display3d, ('POINTS', 'Von-Mises [MPa]'))
    lut = GetColorTransferFunction('Von_Mises_MPa') # ParaView replaces [ ] with _
    lut.ApplyPreset('Jet', True)
    display3d.SetScalarBarVisibility(view3d, True)
    
    # 4. Split Layout & XY Chart (2D Graph)
    layout = GetLayout()
    view2d = CreateView('XYChartView')
    layout.SplitHorizontal(0, 0.5)
    layout.AssignView(1, view3d)
    layout.AssignView(2, view2d)
    
    # Create Graph: Plot Global Max Stress over Time
    plot = PlotGlobalVariablesOverTime(Input=reader)
    display2d = Show(plot, view2d)
    display2d.SeriesVisibility = ['Max Von-Mises [MPa]']
    
    ResetCamera(view3d)
    RenderAllViews()
"""
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        print(f"\n🚀 Launching ParaView Dashboard (Automation Script Mode)...")
        try:
            cmd = f'"{pv_exe}" --script="{script_path}"'
            subprocess.Popen(cmd, shell=True)
            print(">> ParaView Dashboard launched successfully.")
        except Exception as e:
            print(f"⚠️ Failed to launch ParaView: {e}")

    def register_paraview_macro(self):
        """
        [WHTOOLS] ParaView 매크로 자동 등록 (AppData/Roaming)
        """
        import shutil
        appdata = os.environ.get('APPDATA')
        if not appdata: return
        
        macro_dir = os.path.join(appdata, "ParaView", "Macros")
        if not os.path.exists(macro_dir):
            try:
                os.makedirs(macro_dir)
            except: return
            
        # 매크로 스크립트 작성
        macro_path = os.path.join(macro_dir, "WHTOOLS_Dashboard.py")
        macro_content = """
from paraview.simple import *
# [WHTOOLS] Dashboard Setup Macro
reader = GetActiveSource()
if reader:
    view3d = GetActiveView()
    layout = GetLayout()
    view2d = CreateView('XYChartView')
    layout.SplitHorizontal(0, 0.5)
    layout.AssignView(1, view3d)
    layout.AssignView(2, view2d)
    
    plot = PlotGlobalVariablesOverTime(Input=reader)
    Show(plot, view2d)
    RenderAllViews()
    print("Dashboard Configured.")
"""
        with open(macro_path, 'w', encoding='utf-8') as f:
            f.write(macro_content)
        print(f"📦 ParaView Macro Registered at: {macro_path}")

    def export_to_vtkhdf(self, output_dir: str, filename: str = "Result.vtkhdf"):
        """
        [WHTOOLS] VTKHDF 1.0 Transient Unified Mesh Export (Fixed Topology)
        - ParaView 시계열 규격에 맞춰 /VTKHDF/Steps 그룹 및 평탄화된 데이터 구조로 저장
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.last_export_dir = output_dir
        filepath = os.path.join(output_dir, filename)
        
        # [WHTOOLS] Safety Guard: Ensure there is data to export
        if not self.manager.analyzers:
            print("⚠️ [Export Warning] No analyzers found in manager. Skipping VTKHDF export.")
            return None
        
        # --- Robust Filename Handling ---
        actual_path = filepath
        counter = 1
        base_f, ext_f = os.path.splitext(filepath)
        while True:
            try:
                test_f = h5py.File(actual_path, 'a')
                test_f.close()
                break
            except OSError:
                actual_path = f"{base_f}_{counter}{ext_f}"
                counter += 1
                if counter > 50: break
        
        print(f"\n[WHTOOLS] Exporting COMPLIANT VTKHDF (Transient) to: {actual_path}")
        
        # 1. Topology Generation (Frame 0 기준 단일 메쉬)
        all_points_f0 = []
        all_conn = []
        all_offsets = [0]
        all_types = []
        all_part_ids = []
        point_offset = 0
        
        for p_idx, analyzer in enumerate(self.manager.analyzers):
            res = analyzer.sol.res
            num_pts = res * res
            
            quads = []
            for i in range(res - 1):
                for j in range(res - 1):
                    p1 = i * res + j
                    p2 = i * res + (j + 1)
                    p3 = (i + 1) * res + (j + 1)
                    p4 = (i + 1) * res + j
                    quads.append([p1, p2, p3, p4])
            
            quads = np.array(quads) + point_offset
            
            X_2d, Y_2d = np.array(analyzer.sol.X_mesh), np.array(analyzer.sol.Y_mesh)
            ref_basis = np.array(analyzer.ref_basis)
            ref_center = np.array(analyzer.ref_center)
            P_local0 = np.column_stack([X_2d.ravel(), Y_2d.ravel(), np.zeros(num_pts)])
            P_world0 = P_local0 @ ref_basis + ref_center
            
            all_points_f0.append(P_world0)
            all_conn.extend(quads.ravel())
            for _ in range(quads.shape[0]):
                all_offsets.append(all_offsets[-1] + 4)
                all_types.append(9) 
                
            all_part_ids.extend([p_idx] * num_pts)
            point_offset += num_pts
            
        total_points = point_offset
        total_cells = len(all_types)
        total_conn_ids = len(all_conn)
        
        # --- ParaView Transient Compatibility: Topology must be duplicated & indexed per step ---
        single_conn = np.array(all_conn, dtype=np.int64)
        single_types = np.array(all_types, dtype=np.uint8)
        single_offsets = np.array(all_offsets, dtype=np.int64)
        
        full_conn = np.tile(single_conn, self.n_frames)
        full_types = np.tile(single_types, self.n_frames)
        
        # Shift offsets for each frame by the number of connectivity IDs
        f_offsets = []
        for t_idx in range(self.n_frames):
            f_offsets.append(single_offsets[:-1] + t_idx * total_conn_ids)
        f_offsets.append(np.array([self.n_frames * total_conn_ids], dtype=np.int64))
        full_offsets = np.concatenate(f_offsets)
        
        # 2. HDF5 파일 작성
        with h5py.File(actual_path, 'w') as f:
            vtkhdf = f.create_group("VTKHDF")
            
            # Identity Header
            vtkhdf.attrs.create("Version", [1, 0], dtype='i4')
            vtkhdf.attrs.create("Type", np.bytes_("UnstructuredGrid"))
            vtkhdf.create_dataset("Version", data=np.array([1, 0], dtype=np.int32))
            dt_str = h5py.string_dtype(encoding='ascii', length=32)
            ds_type = vtkhdf.create_dataset("Type", shape=(), dtype=dt_str)
            ds_type[()] = "UnstructuredGrid"
            
            # Transient Topology (Duplicated per step for ParaView 1.0)
            vtkhdf.create_dataset("Connectivity", data=full_conn)
            vtkhdf.create_dataset("Offsets", data=full_offsets)
            vtkhdf.create_dataset("Types", data=full_types)
            
            # [WHTOOLS] 각 타임스텝의 크기를 배열로 명시 (Transient 정석)
            vtkhdf.create_dataset("NumberOfPoints", data=np.full(self.n_frames, total_points, dtype=np.int64))
            vtkhdf.create_dataset("NumberOfCells", data=np.full(self.n_frames, total_cells, dtype=np.int64))
            vtkhdf.create_dataset("NumberOfConnectivityIds", data=np.full(self.n_frames, total_conn_ids, dtype=np.int64))
            
            # --- TRANSIENT: Steps Group ---
            steps = vtkhdf.create_group("Steps")
            steps.attrs.create("NSteps", self.n_frames, dtype='i8')
            steps.create_dataset("NumberOfSteps", data=np.array([self.n_frames], dtype=np.int64))
            steps.create_dataset("Values", data=np.array(self.times, dtype='f4'))
            
            point_offsets = np.arange(self.n_frames + 1, dtype=np.int64) * total_points
            cell_offsets = np.arange(self.n_frames + 1, dtype=np.int64) * total_cells
            conn_offsets = np.arange(self.n_frames + 1, dtype=np.int64) * total_conn_ids
            
            steps.create_dataset("PointOffsets", data=point_offsets)
            steps.create_dataset("CellOffsets", data=cell_offsets)
            # ParaView 버전에 따라 ConnectivityOffsets 또는 ConnectivityIdOffsets를 찾으므로 둘 다 생성
            steps.create_dataset("ConnectivityOffsets", data=conn_offsets)
            steps.create_dataset("ConnectivityIdOffsets", data=conn_offsets)
            
            part_offsets = np.arange(self.n_frames + 1, dtype=np.int64)
            steps.create_dataset("PartOffsets", data=part_offsets)
            
            # Time-varying Datasets
            ds_points = vtkhdf.create_dataset("Points", shape=(self.n_frames * total_points, 3), dtype='f4')
            
            pd = vtkhdf.create_group("PointData")
            ds_disp_vec = pd.create_dataset("displacement_vec", shape=(self.n_frames * total_points, 3), dtype='f4')
            ds_vm_stress = pd.create_dataset("Von-Mises [MPa]", shape=(self.n_frames * total_points,), dtype='f4')
            
            # PartID 평탄화 기록
            part_id_flat = np.tile(np.array(all_part_ids, dtype='i4'), self.n_frames)
            pd.create_dataset("PartID", data=part_id_flat)
            
            # [WHTOOLS] [FIX] Transient FieldData causes H5Dread -1 crash in ParaView 1.0. 
            # We will calculate Max Stress history natively within the ParaView script invece.
            
            # 데이터 채우기 Loop
            print("  > Streaming Transient Data (Failsafe Mode)...")
            global_all_max_vm = 0.0
            
            for t_idx in range(self.n_frames):
                frame_points = []
                frame_disp_vec = []
                frame_vm_stress = []
                frame_max_stress = 0.0
                
                for analyzer in self.manager.analyzers:
                    res_m = analyzer.sol.res
                    num_pts_m = res_m * res_m
                    X_2d, Y_2d = np.array(analyzer.sol.X_mesh), np.array(analyzer.sol.Y_mesh)
                    
                    ref_basis, ref_center = np.array(analyzer.ref_basis), np.array(analyzer.ref_center)
                    w_disp = analyzer.results['Displacement [mm]'][t_idx]
                    vm_stress = analyzer.results['Von-Mises [MPa]'][t_idx]
                    R_rigid = analyzer.results['R'][t_idx]
                    c_P, c_Q = analyzer.results['c_P'][t_idx], analyzer.results['c_Q'][t_idx]
                    
                    P_local = np.column_stack([X_2d.ravel(), Y_2d.ravel(), w_disp.ravel()])
                    P_global = (P_local @ ref_basis + ref_center - c_P) @ R_rigid + c_Q
                    
                    P_local0 = np.column_stack([X_2d.ravel(), Y_2d.ravel(), np.zeros(num_pts_m)])
                    P_world0 = P_local0 @ ref_basis + ref_center
                    disp_vec = P_global - P_world0
                    
                    frame_points.append(P_global)
                    frame_disp_vec.append(disp_vec)
                    frame_vm_stress.append(vm_stress.ravel())
                    frame_max_stress = max(frame_max_stress, float(np.max(vm_stress)))
                
                start_row = t_idx * total_points
                end_row = (t_idx + 1) * total_points
                ds_points[start_row:end_row] = np.concatenate(frame_points, axis=0)
                ds_disp_vec[start_row:end_row] = np.concatenate(frame_disp_vec, axis=0)
                ds_vm_stress[start_row:end_row] = np.concatenate(frame_vm_stress, axis=0)
                global_all_max_vm = max(global_all_max_vm, frame_max_stress)

            # [ParaView Compatibility] 데이터 범위 메타데이터 추가
            ds_vm_stress.attrs.create("min", 0.0)
            ds_vm_stress.attrs.create("max", global_all_max_vm)

        return actual_path

    def register_paraview_macro(self):
        """ParaView 매크로 폴더에 WHTOOLS 대시보드 스크립트 등록"""
        macro_name = "WHTOOLS_Dashboard.py"
        appdata = os.environ.get('APPDATA')
        if not appdata: return
        
        macro_dir = os.path.join(appdata, "ParaView", "Macros")
        if not os.path.exists(macro_dir): os.makedirs(macro_dir)
        
        # VTKHDF 구조에 맞춘 차세대 자동화 스크립트
        script_content = f"""# -*- coding: utf-8 -*-
from paraview.simple import *

# 1. 기존 뷰 정리 및 설정
layout = GetLayout()
SetActiveView(None)

# 2. 데이터 로드
data_source = GetActiveSource()
if not data_source:
    print("[WHTOOLS] No data source found.")
else:
    # 3. 3D Deformation View (Left)
    r_view = GetActiveViewOrCreate('RenderView')
    r_view.Background = [0.1, 0.1, 0.1] # Dark
    
    warp = WarpByVector(Input=data_source)
    warp.Vectors = ['POINTS', 'displacement_vec']
    
    warp_disp = Show(warp, r_view)
    ColorBy(warp_disp, ('POINTS', 'Von-Mises [MPa]'))
    warp_disp.RescaleTransferFunctionToDataRange(True, False)
    warp_disp.SetScalarBarVisibility(r_view, True)
    
    # 4. 2D Stress History Chart (Right - Native Calculation)
    stats = DescriptiveStatistics(Input=data_source)
    stats.Variables = ['Von-Mises [MPa]']
    
    # 시간에 따른 데이터 추출 (Plot Global Variables Over Time)
    plot = PlotGlobalVariablesOverTime(Input=stats)
    
    c_view = CreateView('XYChartView')
    plot_disp = Show(plot, c_view)
    plot_disp.SeriesVisibility = ['Maximum'] # 'Von-Mises [MPa]'의 최대값
    plot_disp.SeriesLabel = ['Maximum', 'Max Von-Mises [MPa]']
    
    # 레이아웃 분할 (2:3 비율)
    layout.SplitHorizontal(0, 0.4)
    layout.AssignView(1, r_view)
    layout.AssignView(2, c_view)
    
    RenderAllViews()
"""
        with open(os.path.join(macro_dir, macro_name), "w", encoding="utf-8") as f:
            f.write(script_content)
        print(f"✅ [WHTOOLS] ParaView Macro registered: {macro_name}")

    def launch_paraview(self, vtkhdf_path):
        """해석 결과를 들고 ParaView를 자동 실행 (대시보드 모드)"""
        import subprocess
        paraview_exe = r"C:\Program Files\ParaView 6.0.1\bin\paraview.exe"
        if not os.path.exists(paraview_exe):
            paraview_exe = "paraview" # PATH 의존
            
        filename = os.path.basename(vtkhdf_path)
        temp_script_path = vtkhdf_path + "_dashboard.py"
        
        script_content = f"""# -*- coding: utf-8 -*-
from paraview.simple import *
# 1. Load Data
data = HDFReader(registrationName='{filename}', FileNames=['{vtkhdf_path}'])
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
        with open(temp_script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
            
        print(f"🚀 [WHTOOLS] Launching ParaView Dashboard: {filename}")
        try:
            # [FIX] Escape backslashes for ParaView's internal Python interpreter
            safe_path = vtkhdf_path.replace("\\", "/")
            script_content = script_content.replace(vtkhdf_path, safe_path)
            
            cmd = [paraview_exe, "--script=" + temp_script_path]
            subprocess.Popen(cmd)
        except Exception as e:
            print(f"❌ Failed to launch ParaView: {e}")

if __name__ == "__main__":
    # 라이브러리 직접 실행 테스트용
    print("[WHTOOLS] Exporter Module Ready.")
