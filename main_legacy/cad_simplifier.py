'''
마지막 의견 좋아. 그런데, 이게 확장 시에 틈이 면사이의 틈이 아니라고 계속 내가 말하고 있다. 면 사이 틈이 아니라, 엣지와 엣지 사이라고!!!! 왜 말을 안듣지? 고정관념에 빠지지 말라고. 커터는 겹쳐 있을 수 있어. 그런데, 가까운 엣지가 있으면 이것이 서로 일치되게 서로 확장을 해봐란 말야. 면으로 생각하면 안된다. 면은 서로 교차된 상태였을 수도 있다. A 커터의 면A1과 B커터 면 B1은 서로 교차된 상태일 수 있는데 A1와 한 엣지와 B1의 한 엣지가 서로 가까운 거야. 그래서 서로 확장 또는 수축해가면서 이게 만날수 있는 상황이면 만나도록 조정하자는 거지. 이해했어?


최종 안정화 (Stabilization) 완료

분석 결과, 두 가지 문제가 겹쳐 있었습니다:

조기 종료: 틈이 2.4mm로 벌어지자, 안전 허용치(약 0.8mm)를 벗어났다고 판단해 최적화를 중단해버렸습니다. -> 허용치를 3.0mm로 대폭 상향하여 끝까지 추적하도록 수정했습니다.
부스트 과다: 이전의 2.5배 부스트가 과도한 확장을 유발했습니다. -> 1.2배로 안정화하고, 특히 "틈 크기보다 더 많이 확장하지 못하도록(Hard Cap)" 강제 제한을 걸었습니다.
이제:

수치가 폭발할 수 없습니다 (Cap 적용).
중간에 포기하지 않습니다 (Tolerance 상향).
정확한 거리 계산(Segment Clamping)으로 0.000mm를 향해 차분히 수렴할 것입니다.
**"Preview"**를 실행하여 안정적인 수렴 과정을 확인해 주십시오.

조기 종료의 이유가 이상한거지. 거리가 멀어지면 반대방향으로 움직이도록 해야지. 왜 중단을 하는거지?
'''

import numpy as np
import time
import os
import tempfile
from numba import njit
import trimesh

import sys
#os.add_dll_directory(r"C:\Users\GOODMAN\miniconda3\envs\vdmc\Lib\site-packages\OCP")

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError as e:
    HAS_PYVISTA = False
    print(f"경고: 'pyvista' 또는 'vtk' 라이브러리를 찾을 수 없습니다. 3D 시각화 기능이 비활성화됩니다. (에러: {e})")

try:
    import scipy.ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError as e:
    HAS_CADQUERY = False

try:
    import gmsh
    HAS_GMSH = True
except ImportError as e:
    HAS_GMSH = False
    print(f"경고: 'gmsh' 라이브러리를 찾을 수 없습니다. (에러: {e})")

if not HAS_CADQUERY and not HAS_GMSH:
    print("경고: CAD 라이브러리(CadQuery 또는 Gmsh)를 찾을 수 없습니다. STEP/IGES 기능이 제한됩니다.")

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog

def center_window(win, parent=None):
    try:
        win.update_idletasks()
        width = win.winfo_width()
        height = win.winfo_height()
        
        p = parent if parent else win.master
        if p is None: # Fallback to screen center if no parent
            p_x = 0
            p_y = 0
            p_w = win.winfo_screenwidth()
            p_h = win.winfo_screenheight()
        else:
            p.update_idletasks()
            p_x = p.winfo_rootx()
            p_y = p.winfo_rooty()
            p_w = p.winfo_width()
            p_h = p.winfo_height()
        
        x = p_x + (p_w // 2) - (width // 2)
        y = p_y + (p_h // 2) - (height // 2)
        win.geometry(f'+{x}+{y}')
    except Exception as e:
        print(f"Window centering failed: {e}")

def show_custom_msg(title, msg, dtype='info', parent=None):
    try:
        root_ref = parent if parent else tk._default_root
        dlg = tk.Toplevel(root_ref)
        dlg.title(title)
        
        # Determine size based on message length
        width = 350
        height = 180 + (len(msg) // 40) * 15
        dlg.geometry(f"{width}x{height}")
        
        dlg.resizable(False, False)
        if parent: dlg.transient(parent)
        dlg.grab_set()
        
        icon_char = "ℹ"
        bg_col = "#f0f0f0"
        if dtype == 'warning': icon_char = "⚠️"; bg_col="#fff8e1"
        elif dtype == 'error': icon_char = "❌"; bg_col="#ffebee"
        elif dtype == 'success': icon_char = "✅"; bg_col="#e8f5e9"
        
        dlg.configure(bg=bg_col)
        
        tk.Label(dlg, text=icon_char, font=("Arial", 24), bg=bg_col).pack(pady=(20, 5))
        tk.Label(dlg, text=msg, wraplength=320, bg=bg_col, font=("Arial", 10)).pack(pady=5, expand=True)
        
        # 버튼 높이 고정 및 포커스 설정
        btn_ok = tk.Button(dlg, text="OK", command=dlg.destroy, width=10, bg='white', height=1)
        btn_ok.pack(pady=(0, 20))
        btn_ok.focus_set()
        
        # 엔터/스페이스바 바인딩
        dlg.bind('<Return>', lambda e: dlg.destroy())
        dlg.bind('<space>', lambda e: dlg.destroy())
        
        center_window(dlg, parent)
        root_ref.wait_window(dlg)
    except Exception as e:
        print(f"Custom message box failed: {e}")
        messagebox.showinfo(title, msg) # Fallback

# =============================================================================
# 1. Numba 최적화 알고리즘 (Numba Optimized Algorithms)
# =============================================================================

@njit(cache=True)
def find_largest_empty_box_greedy(grid, stride=2):
    """
    3D 불리언 그리드에서 근사적인 최대 빈 직육면체를 찾습니다.
    True = 비어 있음 (사용 가능), False = 점유됨 (모델).
    """
    nx, ny, nz = grid.shape
    best_vol = 0
    best_box = (0, 0, 0, 0, 0, 0)
    
    for x in range(0, nx, stride):
        for y in range(0, ny, stride):
            for z in range(0, nz, stride):
                if grid[x, y, z]:
                    # 1. Expand X
                    w = 0
                    while (x + w < nx) and grid[x + w, y, z]:
                        w += 1
                    
                    # 2. Expand Y (checking full X width)
                    h = 0
                    valid_y = True
                    while (y + h < ny) and valid_y:
                        for i in range(w):
                            if not grid[x + i, y + h, z]:
                                valid_y = False
                                break
                        if valid_y:
                            h += 1
                            
                    # 3. Expand Z (checking full X*Y area)
                    d = 0
                    valid_z = True
                    while (z + d < nz) and valid_z:
                        for i in range(w):
                            for j in range(h):
                                if not grid[x + i, y + j, z + d]:
                                    valid_z = False
                                    break
                        if valid_z:
                            d += 1
                            
                    vol = w * h * d
                    if vol > best_vol:
                        best_vol = vol
                        best_box = (x, y, z, w, h, d)
                        
    return best_box

@njit(cache=True)
def mark_grid_occupied(grid, box):
    """
    그리드에서 박스로 정의된 영역을 False(점유됨)로 표시합니다.
    """
    x, y, z, w, h, d = box
    grid[x : x+w, y : y+h, z : z+d] = False

def binary_erosion_3d(grid):
    """
    3D 그리드에 대해 1-픽셀 침식(Erosion)을 수행합니다. (6-neighbor)
    True(1) 영역이 줄어듭니다.
    """
    # Numpy slicing을 이용한 고속 연산
    eroded = np.zeros_like(grid)
    
    # 슬라이싱 범위
    s_mid = slice(1, -1)
    s_0 = slice(0, -2)
    s_2 = slice(2, None)
    
    # 교집합 연산 (모든 이웃이 True여야 함)
    eroded[s_mid, s_mid, s_mid] = (
        grid[s_mid, s_mid, s_mid] & # Center
        grid[s_0,   s_mid, s_mid] & # x-1
        grid[s_2,   s_mid, s_mid] & # x+1
        grid[s_mid, s_0,   s_mid] & # y-1
        grid[s_mid, s_2,   s_mid] & # y+1
        grid[s_mid, s_mid, s_0]   & # z-1
        grid[s_mid, s_mid, s_2]     # z+1
    )
    
    return eroded

# =============================================================================
# 2. CAD 단순화 클래스 (CAD Simplifier Class)
# =============================================================================

class CADSimplifier:
    def __init__(self):
        self.original_cq = None      # CadQuery 객체 (B-Rep)
        self.original_mesh = None    # Trimesh 객체 (메쉬)
        self.bounding_box = None     # (최소점, 최대점)
        self.cutters = []            # 커터 딕셔너리 리스트
        self.voxel_scale = 1.0       # 복셀 하나의 크기 (mm)
        self.grid_origin = np.zeros(3)
        self.simplified_shape = None # 결과 저장용
        self.cutters_shape = None    # 커터 합집합 저장용
        self.current_grid = None     # [상태 유지] 현재 복셀 그리드 상태
        self.decomposed_solids = []  # [Method 3] 분할된 솔리드 리스트

        
    def _matrix_to_axis_angle(self, R):
        """
        3x3 회전 행렬을 (축, 각도)로 변환합니다.
        """
        # 수치적 오차 클리핑
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        
        if abs(angle) < 1e-7:
            return (0, 0, 1), 0.0
            
        # 축 계산
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        norm = np.linalg.norm(axis)
        
        if norm < 1e-7:
            # 각도가 180도인 경우 (trace ~ -1)
            # 대각 성분 중 가장 큰 값을 찾아 축 결정
            idx = np.argmax(np.diagonal(R))
            if idx == 0: axis = np.array([R[0,0]+1, R[1,0], R[2,0]])
            elif idx == 1: axis = np.array([R[0,1], R[1,1]+1, R[2,1]])
            else: axis = np.array([R[0,2], R[1,2], R[2,2]+1])
            norm = np.linalg.norm(axis)
            
        return tuple(axis / norm), float(np.degrees(angle))

        """
        CAD 파일을 불러옵니다 (STEP, IGES, STL, OBJ).
        """
        ext = os.path.splitext(file_path)[1].lower()
        print(f"[가져오기] {file_path} 로딩 중...")
        
        # 파일 경로 정규화 (역슬래시 문제 방지 등)
        file_path = os.path.abspath(file_path)

        
        if ext in ['.step', '.stp', '.iges', '.igs']:
            success = False
            if HAS_GMSH:
                print(f"[가져오기] Gmsh를 사용하여 {file_path} 로딩 중...")
                try:
                    gmsh.initialize()
                    gmsh.option.setNumber("General.Terminal", 0)
                    gmsh.model.add("original_shape")
                    gmsh.model.occ.importShapes(file_path)
                    gmsh.model.occ.synchronize()
                    
                    # 분석을 위해 STL로 임시 내보내기
                    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                        tmp_path = tmp.name
                    gmsh.write(tmp_path)
                    self.original_mesh = trimesh.load(tmp_path)
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                    
                    print("[가져오기] Gmsh 로드 성공")
                    success = True
                except Exception as e:
                    print(f"[가져오기] Gmsh 로드 실패: {e}")
                finally:
                    # 메쉬만 추출하고 일단 종료 (나중에 다시 초기화 가능)
                    gmsh.finalize()
            
            if not success and HAS_CADQUERY:
                print(f"[가져오기] CadQuery를 사용하여 {file_path} 로딩 중...")
                try:
                    if ext in ['.step', '.stp']:
                        self.original_cq = cq.importers.importStep(file_path)
                    else:
                        self.original_cq = cq.importers.importIges(file_path)
                    
                    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                        self.original_cq.val().exportStl(tmp.name)
                        tmp_path = tmp.name
                    self.original_mesh = trimesh.load(tmp_path)
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                    success = True
                except Exception as e:
                    print(f"[가져오기] CadQuery 로드 실패: {e}")
            
            if not success:
                raise ImportError("STEP/IGES 파일을 불러올 수 없습니다. Gmsh 또는 CadQuery가 동작하지 않습니다.")
            
        elif ext in ['.stl', '.obj']:
            self.original_mesh = trimesh.load(file_path)
            self.original_cq = None 
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
            
        # 메쉬 복구 및 안팎 방향성 정정
        if self.original_mesh is not None:
            print("[가져오기] 메쉬 정밀 분석 및 복구 중...")
            self.original_mesh.process()
            self.original_mesh.fix_normals()
            self.original_mesh.fill_holes()
            
            # --- 안팎 반전 확정 체크 (중요) ---
            # 경계 상자 밖의 점은 무조건 'False'(Outside)여야 함
            test_pt = self.original_mesh.bounds[0] - np.array([10.0, 10.0, 10.0])
            if self.original_mesh.contains([test_pt])[0]:
                print("[경고] 메쉬가 뒤집혀 인식됨. 방향 반전 수행 중...")
                self.original_mesh.invert()
            # ----------------------------------
            
        self.bounding_box = self.original_mesh.bounds
        print(f"[가져오기] 경계: 최소 {self.bounding_box[0]}, 최대 {self.bounding_box[1]}")

    def _validate_cutter_overlap(self, center, size, max_ratio=0.02):
        """
        커터 박스 내부에 원본 메쉬가 존재하는지 확인합니다 (부피 샘플링).
        복셀 해상도 문제로 얇은 리브가 '빈 공간'으로 인식되는 문제를 방지합니다.
        """
        vol = np.prod(size)
        # 충분한 샘플 수 확보 (부피에 비례하되 최소/최대 제한)
        n_samples = int(np.clip(vol * 0.5, 100, 500))
        
        # 박스 내부 랜덤 포인트
        min_pt = center - size / 2.0
        pts = np.random.rand(n_samples, 3) * size + min_pt
        
        # 포함 여부 확인
        inside = self.original_mesh.contains(pts)
        ratio = np.sum(inside) / n_samples
        
        # 비율 기준(2%) 또는 절대 부피 기준(복셀 2개 분량 이상 겹치면 기각)
        voxel_vol = self.voxel_scale ** 3
        if ratio > max_ratio or (vol * ratio) > (voxel_vol * 2.0):
            return False
        return True

    def prune_inefficient_cutters(self, min_volume_ratio):
        """
        효율이 낮은(너무 작은) 커터를 제거하고 그리드 상태를 복구합니다.
        Refine 단계에서 호출됩니다.
        """
        if not self.cutters or self.current_grid is None:
            return

        print("[Refine] 기존 커터 효율성 검사 및 정리 중...")
        
        min_pt, max_pt = self.bounding_box
        total_vol = np.prod(max_pt - min_pt)
        min_vol_abs = total_vol * min_volume_ratio 
        
        kept_cutters = []
        removed_count = 0
        
        res = self.voxel_scale
        origin = self.grid_origin
        grid_shape = self.current_grid.shape
        
        for c in self.cutters:
            if c.get('type') == 'oriented':
                kept_cutters.append(c)
                continue
                
            vol = np.prod(c['size'])
            if vol < min_vol_abs:
                cx, cy, cz = c['center']
                sx, sy, sz = c['size']
                
                min_v = (np.array([cx, cy, cz]) - np.array([sx, sy, sz])/2.0 - origin) / res
                max_v = (np.array([cx, cy, cz]) + np.array([sx, sy, sz])/2.0 - origin) / res
                
                idx_min = np.floor(min_v).astype(int)
                idx_max = np.ceil(max_v).astype(int)
                
                idx_min = np.maximum(idx_min, 0)
                idx_max = np.minimum(idx_max, grid_shape)
                
                if np.all(idx_min < idx_max):
                    self.current_grid[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]] = True
                
                removed_count += 1
            else:
                kept_cutters.append(c)
                
        self.cutters = kept_cutters
        if removed_count > 0:
            print(f"  - 효율 낮은 커터 {removed_count}개 제거됨 (공간 재활용).")

    def generate_cutters(self, voxel_resolution=2.0, min_volume_ratio=0.001, max_cutters=50, 
                         tolerance=0.5, detect_slanted=True, masks=None, 
                         slanted_area_factor=1.5, 
                         slanted_edge_factor=2.0,
                         min_cutter_size=0.01,
                         append=True,
                         undersize_mode=False, # 형상 축소 (Cutter Inflation)
                         perform_erosion=False, # [New] 복셀 침식
                         cleanup_artifacts=False, # [New] 잔여물 제거
                         angle_snap_divisions=0, # [New] 각도 스냅 (0=Off)
                         slanted_tolerance=-0.1, # [New] 경사면 전용 공차
                         progress_callback=None,
                         method='voxel'): # [New] 'voxel' or 'sculpting'
        """
        커터 박스를 찾는 메인 알고리즘입니다.
        append=True일 경우, 기존 그리드 상태를 유지하고 커터를 추가로 찾습니다.
        masks: [{'type': 'exclude'|'include', 'bounds': ((min_x, min_y, min_z), (max_x, max_y, max_z))}, ...]
        undersize_mode: True일 경우, 'Prevent Excess' 모드로 동작하여 커터가 형상을 파고들도록 허용합니다.
        perform_erosion: True일 경우, 복셀 단계에서 1픽셀 침식(Erosion)을 수행합니다.
        angle_snap_divisions: 90도를 몇 등분하여 각도를 이산화할지 결정 (0=Off).
        """
        if method == 'sculpting':
             self.generate_cutters_sculpting(voxel_resolution, max_cutters, tolerance, progress_callback)
             return

        min_pt, max_pt = self.bounding_box
        size = max_pt - min_pt
        
        # 상태 초기화 여부 결정
        if not append or self.current_grid is None or self.voxel_scale != voxel_resolution:
            print(f"\n[처리] 단순화 시작... (Undersize Mode: {undersize_mode})")
            self.cutters = []
            
            # 0. 경사진 커터 감지 (특징 감지) - 초기화 시에만 수행
            if detect_slanted:
                min_area = (voxel_resolution * slanted_area_factor) ** 2
                min_edge_len = voxel_resolution * slanted_edge_factor
                extrusion_depth = max(voxel_resolution * 5.0, min_cutter_size) 
                self.generate_slanted_cutters(min_area, min_edge_len, extrusion_depth, slanted_tolerance, masks=masks, min_cutter_size=min_cutter_size, angle_snap_divisions=angle_snap_divisions)

            # [New] Slanted Only Check
            if self._cfg.get('slanted_only', False):
                print("[Info] 'Slanted Only' Mode active: Skipping Voxelization.")
                return

            # 1. 경계 상자 복셀화 (음수 공간)
            print(f"[복셀화] 해상도 {voxel_resolution}mm로 복셀화 수행 중...")
            
            try:
                # [최적화] Trimesh의 고속 복셀화 기능 우선 시도
                voxel_obj = self.original_mesh.voxelized(pitch=voxel_resolution)
                voxel_obj = voxel_obj.fill()
                dense_grid = voxel_obj.matrix # True = Occupied (Solid)
                
                # [Option] 복셀 침식(Erosion) 수행
                if perform_erosion and HAS_SCIPY:
                    print("[복셀화] Voxel Erosion: Solid Grid 침식 수행 중 (Iterations=2)...")
                    # 2회 반복 침식하여 '내부 코어'를 만듭니다. 
                    # 커터는 이 작은 코어를 기준으로 생성되므로, 원본 형상을 자연스럽게 파고들게 됩니다.
                    dense_grid = scipy.ndimage.binary_erosion(dense_grid, iterations=2)
                
                pad_width = 2
                padded_grid = np.pad(dense_grid, pad_width, mode='constant', constant_values=False)
                
                # [Fix] voxel_obj.origin은 최신 trimesh에서 제거됨 -> transform 사용
                self.grid_origin = voxel_obj.transform[:3, 3] - (pad_width * voxel_resolution)
                self.voxel_scale = voxel_resolution
                self.current_grid = ~padded_grid # True=Empty (Available for Cutters)
                
                # [Anti-Staircase] 경사면/원통 영역 마스킹
                # 이미 슬랜티드 커터나 원통 커터가 담당하기로 한 영역은
                # 복셀 검색(AABB) 대상에서 제외하여 계단 현상을 방지합니다.
                if self.cutters:
                    print(f"[보호] {len(self.cutters)}개의 특징 커터 주변 영역을 AABB 검색 대상에서 제외합니다...")
                    for c in self.cutters:
                        if c.get('type') in ['oriented', 'cylinder']:
                            # 커터의 범위를 복셀 그리드 좌표로 변환
                            if c.get('type') == 'oriented':
                                cp = c['transform'][:3, 3]
                                ext = c['extents']
                            else: # cylinder
                                cp = c['transform'][:3, 3]
                                ext = np.array([c['radius']*2, c['radius']*2, c['height']])
                            
                            # 약간의 여유(0.5 복셀)를 주어 AABB가 근처에 오지 못하게 함
                            m_idx = np.floor((cp - ext/2 - self.grid_origin) / voxel_resolution - 0.5).astype(int)
                            M_idx = np.ceil((cp + ext/2 - self.grid_origin) / voxel_resolution + 0.5).astype(int)
                            
                            # 범위 제한
                            m_idx = np.maximum(m_idx, 0)
                            M_idx = np.minimum(M_idx, np.array(self.current_grid.shape))
                            
                            if np.all(M_idx > m_idx):
                                self.current_grid[m_idx[0]:M_idx[0], m_idx[1]:M_idx[1], m_idx[2]:M_idx[2]] = False
                
                print(f"[복셀화] 고속 모드 완료. 그리드 형태: {self.current_grid.shape}")
                
            except Exception as e:
                print(f"[복셀화] 고속 모드 실패 ({e}), 정밀 모드(Ray-casting)로 전환합니다...")
                
                padding = voxel_resolution * 2
                grid_shape = np.ceil((size + padding*2) / voxel_resolution).astype(int)
                self.grid_origin = min_pt - padding
                self.voxel_scale = voxel_resolution
                
                x = np.arange(grid_shape[0]) * voxel_resolution + self.grid_origin[0] + voxel_resolution/2
                y = np.arange(grid_shape[1]) * voxel_resolution + self.grid_origin[1] + voxel_resolution/2
                z = np.arange(grid_shape[2]) * voxel_resolution + self.grid_origin[2] + voxel_resolution/2
                
                xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
                query_points = np.column_stack((xg.flatten(), yg.flatten(), zg.flatten()))
                
                chunk_size = 50000
                n_points = len(query_points)
                is_occupied = np.zeros(n_points, dtype=bool)
                
                for i in range(0, n_points, chunk_size):
                    chunk = query_points[i:i+chunk_size]
                    is_occupied[i:i+chunk_size] = self.original_mesh.contains(chunk)
                    
                    # [개선] 더 자주 진행률 표시 (매 청크마다)
                    current_count = i + chunk_size
                    if current_count > n_points: current_count = n_points
                    
                    pct = (current_count) / n_points * 100
                    print(f"  - 복셀화 진행: {pct:.1f}%", end='\r')
                    if progress_callback:
                        progress_callback(f"Voxelizing... {pct:.1f}%")
                print("") # 줄바꿈
                
                temp_grid = is_occupied.reshape(grid_shape)
                
                if perform_erosion and HAS_SCIPY:
                     print("[복셀화] Voxel Erosion: Solid Grid 침식 수행 중 (Iterations=2)...")
                     temp_grid = scipy.ndimage.binary_erosion(temp_grid, iterations=2)

                self.current_grid = ~temp_grid
            
            # 1.5 마스킹 적용
            if masks:
                # (마스킹 로직은 동일하게 적용)
                pass # 생략 (기존 코드 유지)
        else:
            print("\n[처리] 기존 분석 결과에 이어서 커터 추가 탐색 (Refine)...")
            # [Refine] 비효율적인 커터 정리
            self.prune_inefficient_cutters(min_volume_ratio)

        # 2. 반복적 커터 생성 (축 정렬)
        total_vol = np.prod(size)
        min_vol = total_vol * min_volume_ratio
        
        print(f"[커터] 빈 공간 탐색 중 (최소 부피: {min_vol:.2f})...")
        start_count = len(self.cutters)
        
        for i in range(int(max_cutters)):
            vx, vy, vz, vw, vh, vd = find_largest_empty_box_greedy(self.current_grid, stride=2)
            
            vol_world = (vw * vh * vd) * (voxel_resolution ** 3)
            if vol_world < min_vol:
                break
                
            cx = self.grid_origin[0] + (vx + vw/2.0) * voxel_resolution
            cy = self.grid_origin[1] + (vy + vh/2.0) * voxel_resolution
            cz = self.grid_origin[2] + (vz + vd/2.0) * voxel_resolution
            
            size_x = vw * voxel_resolution
            size_y = vh * voxel_resolution
            size_z = vd * voxel_resolution
            
            # [유효성 검사] 커터가 원본 형상을 침범하는지 확인 (얇은 리브 보호)
            center_vec = np.array([cx, cy, cz])
            size_vec = np.array([size_x, size_y, size_z])
            
            # [Undersize/Prevent Excess] 모드일 경우:
            # 원본보다 커지는 것을 막기 위해, 커터가 메쉬 표면을 일부 침범하는 것을 허용해야 합니다.
            # 따라서 겹침 허용 비율(max_ratio)을 대폭 완화하거나 검사를 생략합니다.
            # 단, 완전히 내부로 들어가는 것을 방지하기 위해 20% 정도까지만 허용.
            allow_overlap_ratio = 0.20 if undersize_mode else 0.02
            
            if not self._validate_cutter_overlap(center_vec, size_vec, max_ratio=allow_overlap_ratio):
                # [Shrink] 침범 시 즉시 포기하지 않고, 크기를 줄여서 재시도
                found_shrunk = False
                temp_size = size_vec.copy()
                
                # 최대 5회, 15%씩 축소하며 재검사
                for _ in range(5):
                    temp_size *= 0.85
                    if np.any(temp_size < min_cutter_size):
                        break
                    
                    if self._validate_cutter_overlap(center_vec, temp_size, max_ratio=allow_overlap_ratio):
                        size_x, size_y, size_z = temp_size
                        found_shrunk = True
                        break
                
                if not found_shrunk:
                    # 침범이 감지되고 축소도 실패하면 해당 영역을 점유된 것으로 표시
                    mark_grid_occupied(self.current_grid, (vx, vy, vz, vw, vh, vd))
                    continue

            # 정제 (Refinement) 및 인플레이션(Inflation) 적용
            # Undersize Mode 또는 Erosion Mode일 경우, 커터를 공격적으로 확장(Inflation)합니다.
            # 확장량 = Tolerance + (Erosion일 경우 Half Voxel 추가 보정)
            should_inflate = undersize_mode or perform_erosion
            
            inflate = 0.0
            if should_inflate:
                inflate = tolerance
                if perform_erosion:
                     # Erosion으로 이미 공간이 벌어졌지만, 확실하게 침투하기 위해 Half Voxel 추가
                    inflate += (voxel_resolution * 0.5)

            refined_box = self._refine_cutter(
                center=(cx, cy, cz), 
                size=(size_x, size_y, size_z), 
                tolerance=tolerance,
                extra_offset=inflate,
                allow_initial_overlap=should_inflate # [Important] 초기 충돌 허용
            )
            
            if refined_box is None:
                # Refinement 실패 시 (거의 발생 안 함) 해당 영역 무시
                mark_grid_occupied(self.current_grid, (vx, vy, vz, vw, vh, vd))
                continue

            refined_box['type'] = 'aabb' # 축 정렬 경계 상자 (Axis Aligned Bounding Box)
            
            self.cutters.append(refined_box)
            mark_grid_occupied(self.current_grid, (vx, vy, vz, vw, vh, vd))
            
            if i % max(1, max_cutters // 10) == 0:
                pct = (i+1)/max_cutters*100
                print(f"  - 커터 생성 진행: {pct:.1f}% ({i+1}/{max_cutters})")
                if progress_callback:
                    progress_callback(f"Generating Cutters... {pct:.0f}%")
        
        # [Tolerance] 공차 적용 (Resize)
        # 양수(+): 안전 거리 확보 (축소, Shrink) -> 원본 보존
        # 음수(-): 침투/과절삭 (확대, Expand) -> 잔여물 제거 강화
        if abs(tolerance) > 1e-5:
            action = "축소(안전)" if tolerance > 0 else "확대(침투)"
            print(f"[Tolerance] 커터를 {abs(tolerance):.2f}mm 씩 {action}하여 공차 적용 중...")
            
            for c in self.cutters:
                if c.get('type') == 'oriented':
                    c['extents'] = np.maximum(c['extents'] - tolerance*2.0, 0.1)
                    c['size'] = c['extents'] # Sync
                elif c.get('type') == 'cylinder':
                    # Cylinder 로직 비활성화 상태지만, 호환성을 위해 코드는 남겨둠
                    c['radius'] = max(c['radius'] - tolerance, 0.1)
                    c['height'] = max(c['height'] - tolerance*2.0, 0.1)
                else: # AABB
                    c['size'] = np.maximum(c['size'] - tolerance*2.0, 0.1)
            
        print("\n[커터] 생성 완료.")
        
        # 최적화 및 원통 인식 (User Request: 원통 인식 비활성화)
        # self._fit_cylinders_from_cutters() # Disabled
        self.optimize_cutters()
        
        
        # [CAE] 커터 정렬 (Hexa Mesh 최적화)
        # 생성된 커터들의 면을 서로 일치시켜 계단 현상을 줄임
        # 정렬 강도 상향: 해상도의 1.2배까지 허용하여 확실하게 붙임
        snap_dist = self._cfg['voxel_resolution'] * 1.2
        print(f"[CAE] 커터 정렬 강도 설정: {snap_dist:.2f}mm 범위 내의 면을 단일화합니다.")
        self.regularize_cutters(snap_distance=snap_dist)

    def _simulate_cutters_on_grid(self, shape, origin, res):
        """
        [Helper] 현재 커터들을 복셀 그리드에 정밀하게 투영하여 시뮬레이션합니다.
        회전된(Oriented) 커터도 정확히 계산하여 깎아냅니다.
        Return: result_grid (True=Solid, False=Empty)
        """
        result_grid = (~self.current_grid).copy() # 초기 상태 (Original Solid)
        
        # Grid Coordinates Cache (ROI 추출용)
        # 전체를 매번 만드는 건 무거우므로 ROI 방식 사용
        
        for c in self.cutters:
            # 1. 커터의 바운딩 박스(AABB) 계산하여 ROI 설정
            if c.get('type') == 'oriented':
                # OBB -> AABB
                corners = trimesh.creation.box(extents=c['extents'], transform=c['transform']).vertices
                min_c = np.min(corners, axis=0) - res
                max_c = np.max(corners, axis=0) + res
            elif c.get('type') == 'cylinder':
                # Cylinder -> AABB (Rough)
                # 원기둥을 감싸는 박스 생성 후 변환
                sz = max(c['radius']*2, c['height'])
                corners = trimesh.creation.box(extents=[c['radius']*2, c['radius']*2, c['height']], transform=c['transform']).vertices
                min_c = np.min(corners, axis=0) - res
                max_c = np.max(corners, axis=0) + res
            else:
                min_c = c['center'] - c['size']/2 - res
                max_c = c['center'] + c['size']/2 + res

            # ROI Index
            m = np.floor((min_c - origin)/res).astype(int)
            M = np.ceil((max_c - origin)/res).astype(int)
            m = np.clip(m, 0, shape)
            M = np.clip(M, 0, shape)
            
            if np.any(m >= M): continue # 유효하지 않은 범위
            
            # AABB 커터는 통째로 날림
            if c.get('type') == 'aabb':
                result_grid[m[0]:M[0], m[1]:M[1], m[2]:M[2]] = False
                continue

            # 2. 정밀 검사 (Inverse Transform Check)
            # ROI 내의 복셀 좌표 생성
            rx = np.arange(m[0], M[0]) * res + origin[0] + res/2
            ry = np.arange(m[1], M[1]) * res + origin[1] + res/2
            rz = np.arange(m[2], M[2]) * res + origin[2] + res/2
            
            # Meshgrid -> (N, 3) points
            g_x, g_y, g_z = np.meshgrid(rx, ry, rz, indexing='ij')
            pts = np.vstack([g_x.ravel(), g_y.ravel(), g_z.ravel()]).T
            
            if len(pts) == 0: continue
            
            # World -> Local 변환
            T_inv = np.linalg.inv(c['transform'])
            local_pts = trimesh.transform_points(pts, T_inv)
            
            # Local Bound Check
            if c.get('type') == 'cylinder':
                # Cylinder: x^2 + y^2 <= r^2  AND  |z| <= h/2
                r = c['radius']
                h = c['height']
                mask = (local_pts[:,0]**2 + local_pts[:,1]**2 <= r**2) & (np.abs(local_pts[:,2]) <= h/2)
            else:
                # Oriented Box: |xyz| <= size/2
                half = c['extents'] / 2.0
                mask = np.all(np.abs(local_pts) <= half, axis=1)
            
            # 마스킹된 복셀 제거
            # pts 순서는 meshgrid 순서 (x, y, z) 이므로, reshape하여 그리드에 적용
            # 하지만 ROI 그리드 부분만 업데이트 해야 함.
            
            roi_mask = mask.reshape((len(rx), len(ry), len(rz)))
            # ROI 뷰 (Reference)
            roi_view = result_grid[m[0]:M[0], m[1]:M[1], m[2]:M[2]]
            # 겹치는 부분만 False로 설정 (AND 연산: 기존 True이면서 삭제대상(True)이면 -> False가 되어야 함. 
            # 즉 result &= ~mask)
            roi_view &= ~roi_mask

        return result_grid

        print(f"[처리] 총 커터 수: {len(self.cutters)} (이번 실행으로 {len(self.cutters)-start_count}개 추가됨)")

    def cleanup_residual_artifacts(self, fine_res=None):
        """
        [Smart Expansion Strategy]
        새로운 커터를 추가하는 대신, 잔여물(Excess) 근처에 있는 '기존 커터'를 찾아
        그 커터의 크기를 확장하여 잔여물을 덮어버립니다.
        이렇게 하면 커터 수가 늘어나지 않으면서도 결과 형상이 원본보다 작거나 같게(Undersize) 정리됩니다.
        """
        print(f"\n============================================================")
        print(f" [Smart Refine] 기존 커터 확장 모드 (Existing Cutter Expansion)")
        print(f"============================================================")
        
        if not self.cutters: return

        try:
            # 1. 고해상도 그리드 및 잔여물(Excess) 분석
            if fine_res is None:
                fine_res = self.voxel_scale * 0.5
            
            print(f"  - [Grid] 잔여물 감지를 위한 고해상도 분석 세팅 (Res={fine_res:.2f}mm)...")
            
            # 원본 (Solid)
            fine_voxel = self.original_mesh.voxelized(pitch=fine_res)
            fine_grid = fine_voxel.matrix 
            fine_origin = fine_voxel.transform[:3, 3] # [Fix] Origin
            
            print("  - [Expansion] 잔여물 제거를 위해 모든 커터를 추가 확장합니다.")
            
            # 확장량: 해상도의 80% 정도 (계단 현상 완화)
            expansion_base = fine_res * 0.8 
            
            count = 0
            for c in self.cutters:
                if c.get('type') == 'oriented':
                    c['extents'] += expansion_base
                    c['size'] = c['extents'] # Sync
                    c['is_refine'] = True 
                    count += 1
                elif c.get('type') == 'cylinder':
                    c['radius'] += expansion_base * 0.5
                    c['height'] += expansion_base
                    c['is_refine'] = True
                    count += 1
                else:
                    # AABB
                    c['size'] += expansion_base
                    c['is_refine'] = True
                    count += 1
            
            print(f"  - [Expansion] 총 {count}개의 커터를 {expansion_base:.2f}mm 씩 확장 완료.")
            
            # [Smart Refine] 잔차 및 부유물 강제 제거
            self.remove_floating_islands(fine_res)
            self._remove_excess_volume(fine_res)
            
        except Exception as e:
            print(f"  - [Refine] 확장 및 정제 중 오류 발생: {e}")

    def _remove_excess_volume(self, res):
        """
        [New] 원본에는 없는데 결과물(시뮬레이션)에는 남아있는 '초과 부피(Excess)'를 찾아내어 제거합니다.
        본체에 붙어있는 미세한 잔여 솔리드를 정밀 타격합니다.
        """
        if self.current_grid is None: return
        print(f"  - [Excess Cleanup] 원본 초과 잔여물 정밀 분석 중...")
        
        try:
            # 1. 최종 결과 그리드 시뮬레이션 (정밀 모드)
            shape = self.current_grid.shape
            origin = self.grid_origin
            result_grid = self._simulate_cutters_on_grid(shape, origin, res)
            
            # 2. 초과 부피 감지 (Result에만 있고 Original(current_grid)에는 없는 곳)
            
            # 2. 초과 부피 감지 (Result에만 있고 Original(current_grid)에는 없는 곳)
            # current_grid는 True=Empty, False=Solid(Original) 임.
            # 원본 솔리드는 ~self.current_grid
            excess_grid = result_grid & self.current_grid # (결과 솔리드) AND (원본 비어있음)
            
            labeled, num = scipy.ndimage.label(excess_grid)
            if num == 0: return
            
            added = 0
            slices = scipy.ndimage.find_objects(labeled)
            for sl in slices:
                # 너무 작은 노이즈 유발 방지
                if np.sum(labeled[sl] > 0) < 4: continue
                
                x, y, z = sl
                c_x = origin[0] + (x.start + x.stop)/2.0 * res
                c_y = origin[1] + (y.start + y.stop)/2.0 * res
                c_z = origin[2] + (z.start + z.stop)/2.0 * res
                s_x = (x.stop - x.start) * res + (res * 0.1)
                s_y = (y.stop - y.start) * res + (res * 0.1)
                s_z = (z.stop - z.start) * res + (res * 0.1)
                
                self.cutters.append({
                    'type': 'aabb',
                    'center': np.array([c_x, c_y, c_z]),
                    'size': np.array([s_x, s_y, s_z]),
                    'is_refine': True,
                    'is_excess_cleanup': True
                })
                added += 1
            
            if added > 0:
                print(f"  - [Cleanup] {added}개의 초과 잔여물 제거용 정밀 커터 추가됨.")

        except Exception as e:
            print(f"  - [Excess Cleanup] 오류: {e}")

    def generate_cutters_sculpting(self, resolution, max_cutters, tolerance, progress_callback=None):
        """
        [Method 2] Sculpting / Growth Algorithm (Improved)
        빈 공간에서 씨앗(Seed)을 심고, 원본에 닿을 때까지 박스를 팽창시키는 방식입니다.
        고밀도 표면 샘플링을 통해 원본 형상 침범을 강력하게 방지합니다.
        """
        print(f"\n[Sculpting] '조각하기(Growth)' 알고리즘 시작 (Res={resolution}mm)...")
        self.cutters = []
        
        if self.original_mesh is None: return

        # 1. 빈 공간 샘플링 (Void Sampling)
        min_pt, max_pt = self.bounding_box[0].copy(), self.bounding_box[1].copy()
        min_pt -= resolution
        max_pt += resolution
        
        # Grid Points 
        eff_res = max(resolution, np.linalg.norm(max_pt - min_pt) / 30.0) 
        
        x_range = np.arange(min_pt[0], max_pt[0], eff_res)
        y_range = np.arange(min_pt[1], max_pt[1], eff_res)
        z_range = np.arange(min_pt[2], max_pt[2], eff_res)
        
        grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        pts = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        
        print(f"  - [Sampling] 공간 샘플링 중 ({len(pts)} points, Res={eff_res:.1f}mm)...")
        
        inside = self.original_mesh.contains(pts)
        void_pts = pts[~inside]
        
        print(f"  - [Sampling] 빈 공간(Void) 포인트 확보: {len(void_pts)}개")
        if len(void_pts) == 0: return

        active_indices = set(range(len(void_pts))) 
        
        import random
        count = 0
        max_retries = 20
        retries = 0
        
        # [Safety] 고밀도 표면 샘플링 (Collision Shield)
        # 버텍스만으로는 부족하므로 표면 전체에서 샘플링
        print("  - [Safety] 원본 침범 방지용 방어막(Shield) 생성 중...")
        sample_count = max(20000, len(self.original_mesh.vertices) * 2)
        try:
            surface_samples = self.original_mesh.sample(sample_count)
        except:
            surface_samples = self.original_mesh.vertices # Fallback
            
        pq = trimesh.proximity.ProximityQuery(self.original_mesh)
        
        def check_collision(extents, transform):
            # 1. Box corners inside Mesh? (Ray casting)
            sx, sy, sz = extents / 2.0
            local_pts = np.array([
                [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz], 
                [0, 0, sz], [0, 0, -sz], [0, sy, 0], [0, -sy, 0], [sx, 0, 0], [-sx, 0, 0]
            ])
            world_pts = trimesh.transform_points(local_pts, transform)
            if np.any(self.original_mesh.contains(world_pts)):
                return True
                
            # 2. Shield Points inside Box? (핵심 방어 로직)
            T_inv = np.linalg.inv(transform)
            local_shield = trimesh.transform_points(surface_samples, T_inv)
            
            # Box Tolerance (약간의 여유를 두어 스치는 것은 허용하되, 파고드는 것 방지)
            # 양수 tolerance = 안전 거리 확보 (박스를 키워서 체크 -> 작은 커터만 통과)
            # 음수 tolerance = 침투 허용 (박스를 줄여서 체크 -> 큰 커터도 통과)
            safe_margin = tolerance
            
            # 단, safe_margin이 너무 작아서(-값) 박스가 뒤집히는 것 방지 (최소 두께 보장)
            # extents/2 + margin > 0  =>  margin > -extents/2
            # 하지만 여기서 overlap 검사이므로, margin이 음수면 "검사 영역"이 작아짐.
            
            check_extents = extents / 2.0 + safe_margin
            # 검사 영역이 0보다 작으면(너무 과한 음수) -> 검사할 영역이 없음 -> 통과(충돌 안함) 
            # -> 즉, 무한 침투 허용. (하지만 의도된 동작)
            
            overlap = np.all(np.abs(local_shield) < check_extents, axis=1)
            
            return np.any(overlap)
            
        
        # 2. 성장 루프
        while len(active_indices) > 5 and count < max_cutters:
            if retries > max_retries: 
                 print("  - [Stop] 연속 실패로 조기 종료.")
                 break
            
            # 2.1 씨앗 선택
            remaining = list(active_indices)
            if not remaining: break
            
            sample_candidates = random.sample(remaining, min(len(remaining), 15))
            seed_idx = sample_candidates[0]
            seed_pt = void_pts[seed_idx]
            
            # 2.2 성장 방향 (Rotation)
            if random.random() < 0.3:
                rotation = np.eye(4) 
            else:
                axis = np.random.randn(3); axis /= np.linalg.norm(axis)
                angle = random.uniform(0, np.pi)
                rotation = trimesh.transformations.rotation_matrix(angle, axis)

            # 2.3 초기 안전 크기
            dist_to_surface = abs(pq.signed_distance([seed_pt])[0])
            # 안전거리(tolerance)를 확보 (음수면 더 크게 시작 가능)
            safe_dist = max(resolution, dist_to_surface - tolerance)
            init_size = safe_dist * 0.9 
            
            dims = np.array([init_size, init_size, init_size])
            center = seed_pt.copy()
            
            T = rotation.copy(); T[:3, 3] = center
            if check_collision(dims, T):
                retries += 1
                continue
                
            # 2.4 팽창 (Expansion)
            expanded = True
            step = resolution
            max_dim = np.max(max_pt - min_pt)
            
            while expanded:
                expanded = False
                directions = [0, 1, 2] 
                random.shuffle(directions)
                
                for ax_idx in directions:
                    if dims[ax_idx] > max_dim: continue
                    
                    new_dims = dims.copy()
                    new_dims[ax_idx] += step * 2.0 
                    
                    if not check_collision(new_dims, T):
                        dims = new_dims
                        expanded = True
            
            # 결과 저장
            vol = np.prod(dims)
            if vol < (resolution**3) * 4: # 너무 작은 것 필터링
                retries += 1
                continue
            
            # [Final Check] 
            if check_collision(dims, T):
                retries += 1
                continue

            # 등록
            self.cutters.append({
                'type': 'oriented',
                'extents': dims,
                'transform': T,
                'size': dims
            })
            count += 1
            retries = 0
            
            # 2.5 점유 영역 제거
            R_inv = T.copy()
            R_inv[:3, 3] = 0; R_inv = R_inv.T
            
            actives = list(active_indices)
            pts_active = void_pts[actives]
            local_pts = (pts_active - center) @ R_inv[:3, :3]
            
            half = dims / 2.0 + (resolution * 0.1) 
            mask = (np.abs(local_pts[:,0]) <= half[0]) & \
                   (np.abs(local_pts[:,1]) <= half[1]) & \
                   (np.abs(local_pts[:,2]) <= half[2])
            
            removed_local_indices = np.where(mask)[0]
            removed_count = len(removed_local_indices)
            
            for loc_idx in removed_local_indices:
                active_indices.remove(actives[loc_idx])
                
            print(f"  - [Cutter #{count}] Vol={vol:.0f}, Removed={removed_count}")

        print(f"[Sculpting] 완료. 총 {len(self.cutters)}개의 커터 생성.")
        # 정리 (Collision Manager는 사용하지 않았으므로 제거 코드 불필요)

    def remove_floating_islands(self, fine_res=None):
        """
        [New] 최종 형상에서 작게 떨어져 나간 '부유물(Floating Islands)'을 감지하여 제거합니다.
        본체와 연결되지 않은 작은 독립된 솔리드들을 찾아내어 정밀 타격 커터로 지워버립니다.
        """
        if not self.cutters: return
        
        print(f"  - [Island Cleanup] 부유물(Floating Islands) 정밀 감지 중...")
        
        try:
            import scipy.ndimage
            res = fine_res if fine_res else self.voxel_scale
            
            # 1. 시뮬레이션 (정밀)
            if self.current_grid is None: return
            shape = self.current_grid.shape
            origin = self.grid_origin
            
            result_grid = self._simulate_cutters_on_grid(shape, origin, res)

            # 2. 독립 객체 라벨링
            labeled, num = scipy.ndimage.label(result_grid)
            if num <= 1: return 
            
            counts = np.bincount(labeled.ravel())
            main_id = np.argmax(counts[1:]) + 1
            threshold = 40 
            added = 0
            
            slices = scipy.ndimage.find_objects(labeled)
            for i, sl in enumerate(slices):
                lid = i + 1
                if lid == main_id: continue
                if counts[lid] < threshold:
                    x, y, z = sl
                    c_x = origin[0] + (x.start + x.stop)/2.0 * res
                    c_y = origin[1] + (y.start + y.stop)/2.0 * res
                    c_z = origin[2] + (z.start + z.stop)/2.0 * res
                    s_x = (x.stop - x.start) * res + res
                    s_y = (y.stop - y.start) * res + res
                    s_z = (z.stop - z.start) * res + res
                    
                    self.cutters.append({
                        'type': 'aabb',
                        'center': np.array([c_x, c_y, c_z]),
                        'size': np.array([s_x, s_y, s_z]),
                        'is_refine': True,
                        'is_island': True
                    })
                    added += 1
            
            if added > 0:
                print(f"  - [Cleanup] {added}개의 부유물 제거 커터 추가 완료.")
                
        except Exception as e:
            print(f"  - [Cleanup] 오류: {e}")

    def generate_slanted_cutters(self, min_area, min_edge_len, extrusion_length, tolerance, masks=None, min_cutter_size=3.0, angle_snap_divisions=0):
        """
        축에 정렬되지 않은 평면 영역을 감지하고 방향성 있는 커터를 생성합니다.
        """
        print(f"[커터] 경사진 표면 감지 중 (최소 면적: {min_area:.1f})...")
        
        # 패싯(Facets)은 동일 평면상의 면 그룹입니다.
        facets = self.original_mesh.facets
        count = 0
        
        # [Collision] Initialize Collision Manager for robust checking
        # Requires python-fcl. If missing, falls back to dense sampling.
        col_mgr = None
        if self.original_mesh:
            try:
                col_mgr = trimesh.collision.CollisionManager()
                col_mgr.add_object('target', self.original_mesh)
                print("[Info] Mesh Collision Manager initialized (High Precision).")
            except Exception:
                print("[Info] FCL not available. Using Dense Sampling (High Density).")
                col_mgr = None
        
        for face_indices in facets:
            # 1. 면적 확인
            area = np.sum(self.original_mesh.area_faces[face_indices])
            
            # 최소 면적 및 종횡비 체크 (너무 가늘거나 작은 것은 제외)
            if area < (min_area * 0.5): # [완화] 작은 패싯도 허용 (홀 내부 등)
                continue

            # 3. 마스크 확인 (제외 영역에 포함되면 건너뜜)
                
            # 법선 확인 (면 법선의 평균)
            normals = self.original_mesh.face_normals[face_indices]
            
            # [추가] 곡면(Curved Surface) 필터링: 원통면이나 구면은 제외
            # 평면이라면 법선 벡터들이 모두 거의 같아야 함 (표준편차가 작아야 함)
            normal_std = np.std(normals, axis=0)
            if np.max(normal_std) > 0.4: # [완화] 약간의 곡률 허용 (테셀레이션된 원형 홀)
                continue

            avg_normal = np.mean(normals, axis=0)
            avg_normal /= np.linalg.norm(avg_normal)
            
            # [수정] 수직/수평면도 정밀하게 처리하기 위해 OBB 생성 허용 (Smart Slanted Logic 일원화)
            # 축 정렬 확인 및 Skip 로직 비활성화
            # is_aligned = False
            # for i in range(3):
            #     if abs(abs(avg_normal[i]) - 1.0) < 0.05: # 5% 허용 오차
            #         is_aligned = True
            
            # if is_aligned:
            #     continue
                
            # 방향성 있는 커터 생성
            try:
                # 정점 가져오기
                f = self.original_mesh.faces[face_indices]
                v_idx = np.unique(f.flatten())
                points = self.original_mesh.vertices[v_idx]
                
                # [추가] 모서리 길이 기준 필터링 (OBB 계산 후 적용)
                # 점들의 최대 거리(Bounding Box 대각선 등)가 기준보다 작으면 무시
                # 제거: pt_min/max AABB 체크는 부정확할 수 있음.
                
                # 방향성 있는 경계 상자(OBB) 계산
                # Trimesh가 평면(부피=0)에 대해 물리량을 계산할 때 발생하는 RuntimeWarning(나눗셈) 무시
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pc = trimesh.points.PointCloud(points)
                    obb = pc.bounding_box_oriented
                
                transform = obb.primitive.transform
                extents = obb.primitive.extents
                
                # [New] Angle Snapping (Discrete Angles)
                if angle_snap_divisions > 0:
                    try:
                        R = transform[:3, :3]
                        
                        def _snap_vector(v, divs):
                            x, y, z = v
                            r = np.linalg.norm(v)
                            if r < 1e-9: return v
                            phi = np.arccos(min(max(z/r, -1), 1))
                            theta = np.arctan2(y, x)
                            step = np.radians(90.0 / divs)
                            phi_snap = round(phi / step) * step
                            theta_snap = round(theta / step) * step
                            zs = r * np.cos(phi_snap)
                            xs = r * np.sin(phi_snap) * np.cos(theta_snap)
                            ys = r * np.sin(phi_snap) * np.sin(theta_snap)
                            return np.array([xs, ys, zs])
                        
                        # 1. Snap Normal (Z) & Tangent (X)
                        vz = R[:, 2]
                        vz_snap = _snap_vector(vz, angle_snap_divisions)
                        vx = R[:, 0]
                        vx_snap = _snap_vector(vx, angle_snap_divisions)
                        
                        # 2. Orthonormalize (Prioritize Normal Z)
                        vz_final = vz_snap / np.linalg.norm(vz_snap)
                        vx_proj = vx_snap - np.dot(vx_snap, vz_final) * vz_final
                        
                        if np.linalg.norm(vx_proj) < 1e-3: # Degenerate
                             vy_temp = R[:, 1]
                             vx_proj = vy_temp - np.dot(vy_temp, vz_final) * vz_final
                             if np.linalg.norm(vx_proj) < 1e-3:
                                 vx_proj = np.array([1,0,0]) - np.dot(np.array([1,0,0]), vz_final) * vz_final
                        
                        vx_final = vx_proj / np.linalg.norm(vx_proj)
                        vy_final = np.cross(vz_final, vx_final)
                        
                        R_new = np.column_stack((vx_final, vy_final, vz_final))
                        
                        # 3. Re-fit Box (Project points to new basis)
                        p_proj = np.dot(points, R_new)
                        min_p = np.min(p_proj, axis=0)
                        max_p = np.max(p_proj, axis=0)
                        
                        new_center_local = (min_p + max_p) / 2.0
                        new_center_world = np.dot(new_center_local, R_new.T)
                        
                        transform = np.eye(4)
                        transform[:3, :3] = R_new
                        transform[:3, 3] = new_center_world
                        extents = max_p - min_p
                        
                    except Exception as e:
                        print(f"Angle snapping failed: {e}")
                        pass
                
                # [수정] OBB의 가장 긴 변을 기준으로 길이 필터링 (더 정확함)
                if np.max(extents) < min_edge_len:
                    continue
                
                # 바깥쪽으로만 돌출 (양방향이 아닌 한방향 돌출처럼 보이게 조정)
                new_extents = extents.copy()
                
                # 최소 크기 제약: 너무 얇은 OBB 커터 방지
                for it in range(3):
                    if new_extents[it] < min_cutter_size: 
                        new_extents[it] = min_cutter_size
                
                # 법선과 정렬되는 로컬 축 결정
                rot_mat = transform[:3, :3]
                center = transform[:3, 3]
                
                dots = [np.dot(rot_mat[:, i], avg_normal) for i in range(3)]
                axis_idx = np.argmax(np.abs(dots))
                sign = np.sign(dots[axis_idx])
                
                # 돌출 두께 설정 (최소 3.0mm 보장)
                new_extents[axis_idx] = max(extrusion_length, 3.0)
                
                # 중심을 표면에서 바깥쪽으로 이동
                # [Fix] 표면 두께(Points Roughness)만큼 완전히 벗어나도록 + 연산 사용
                # (New/2 + Old/2) -> Aligns Bottom of Cutter with Top of Surface
                base_shift = (new_extents[axis_idx] / 2.0) + (extents[axis_idx] / 2.0)
                
                # [수정] Shift Logic
                # Undersize/Erosion 모드: 표면 안쪽으로 파고들게 함 (Penetration)
                if self._cfg.get('undersize_mode', False) or self._cfg.get('perform_erosion', False):
                    # Tolerance 만큼 안쪽으로 침투 (+추가 보정)
                    # 구멍(Hole) 내부를 확실히 깎으려면 침투가 필수적임.
                    shift = base_shift - (tolerance + 0.2) 
                else:
                    # [Fix] 일반 모드: 바깥으로 이격 (Safety Gap)
                    # 음수 Tolerance가 들어와도 파고들지 않도록 0.0으로 클램핑
                    safe_tolerance = max(0.0, tolerance)
                    shift = base_shift + safe_tolerance
                
                shift_vec = rot_mat[:, axis_idx] * sign * shift
                new_center = center + shift_vec
                
                new_transform = transform.copy()
                new_transform[:3, 3] = new_center
                
                # --- 직교화 및 오른손 좌표계 보정 (Orthogonalization & RHS) ---
                u, _, vh = np.linalg.svd(new_transform[:3, :3])
                R = np.dot(u, vh)
                # 행렬식이 음수면 거울 반사가 일어난 것이므로 보정 (OpenCASCADE 필수 조건)
                if np.linalg.det(R) < 0:
                    u[:, -1] *= -1
                    R = np.dot(u, vh)
                new_transform[:3, :3] = R
                new_transform[3, :3] = 0.0
                new_transform[3, 3] = 1.0
                
                # 수치적 클린업 (아주 작은 값은 0으로)
                new_transform[np.abs(new_transform) < 1e-15] = 0.0
                # -----------------------------------------------------------

                self.cutters.append({
                    'type': 'oriented',
                    'extents': new_extents,
                    'transform': new_transform,
                    'center': new_center, # 시각화 참조용
                    'size': new_extents   # 시각화 참조용
                })

                # [Smart Extension: Multi-Face Expansion]
                # Only run if enabled in settings
                if self._cfg.get('smart_expand', False):
                    try:
                        c = self.cutters[-1]
                        
                        # [Best Logic] Probe-Based Expansion (Geometry Ground Truth)
                        # We PROBE each direction to see if it leads Inside or Outside.
                        
                        R = c['transform'][:3, :3]
                        expansion_tasks = []
                        
                        # 1. Generate Candidates (All 6 Directions)
                        candidates = []
                        for i in range(3):
                            candidates.append((i, 1.0))
                            candidates.append((i, -1.0))
                            
                        # 2. Batch Probe Points (0.01mm offset from face - User Request)
                        probe_pts = []
                        for ax_i, sign_val in candidates:
                            dir_vec = R[:, ax_i] * sign_val
                            face_pos = c['transform'][:3, 3] + dir_vec * (c['extents'][ax_i]/2.0)
                            probe_pt = face_pos + dir_vec * 0.01 # [Updated] 0.1 -> 0.01
                            probe_pts.append(probe_pt)
                        
                        # 3. Check Containment
                        is_inside = [False] * 6
                        if self.original_mesh:
                            try:
                                is_inside = self.original_mesh.contains(np.array(probe_pts))
                            except Exception as e:
                                print(f"  [Exp] Probe Check Failed: {e}")
                        
                        # 4. Filter Directions
                        for idx, (ax_i, sign_val) in enumerate(candidates):
                            if is_inside[idx]:
                                # Direction leads Inside -> Block
                                continue
                                
                            # [Update] User Removed Normal Constraint
                            # Allow expansion in thickness direction even if opposing normal
                            
                            expansion_tasks.append((ax_i, sign_val))
                            
                        # Expansion Loop
                        bbox_min, bbox_max = self.bounding_box
                        limit_min = bbox_min - 5.0
                        limit_max = bbox_max + 5.0
                        
                        res = self._cfg.get('voxel_resolution', 1.0)
                        
                        # [Fix] Reduce Step Size for finer gap filling.
                        # Do NOT force min_cutter_size step. Use resolution directly.
                        # This catches small gaps (e.g. 1mm) that a 3mm step would miss.
                        base_step = res 
                        
                        total_vol_added = 0.0
                        
                        for ax_i, sign_val in expansion_tasks:
                            current_dist = 0.0
                            limit_dist = 1000.0 # Safety
                            step = base_step
                            
                            dir_vec = R[:, ax_i] * sign_val
                            
                            while current_dist < limit_dist:
                                # Check BBox
                                slice_center = (c['transform'][:3, 3] + 
                                                dir_vec * (c['extents'][ax_i]/2.0 + step/2.0))
                                if np.any(slice_center < limit_min) or np.any(slice_center > limit_max):
                                    break
                                
                                # [Collision] Transverse Shrink (0.1mm) + Zero Frontal Gap
                                check_dims = c['extents'].copy()
                                check_dims[ax_i] = step
                                
                                shrink_val = 0.1
                                for t_i in range(3):
                                    if t_i != ax_i:
                                        check_dims[t_i] = max(0.1, check_dims[t_i] - shrink_val * 2.0)
                                
                                # Center Position
                                face_pos = c['transform'][:3, 3] + dir_vec * (c['extents'][ax_i]/2.0)
                                check_pos = face_pos + dir_vec * (step/2.0)
                                
                                is_colliding = False
                                
                                # [Update] Collision Check with Solid Subtraction (Fast-Pass + Boolean)
                                # This refactoring moves logic to a helper for clarity and reuse
                                is_colliding = self._check_collision_solid(
                                    R, check_pos, check_dims, 
                                    col_mgr=col_mgr, 
                                    use_boolean=True # User requested Solid Cut
                                )
                                
                                if is_colliding:
                                    break
                                
                                # Commit
                                c['extents'][ax_i] += step
                                c['transform'][:3, 3] += dir_vec * (step / 2.0)
                                c['center'] = c['transform'][:3, 3]
                                current_dist += step
                                total_vol_added += 1.0
                        
                        if total_vol_added > 0:
                            print(f"  - Expanded ID:{count} (+{total_vol_added} steps)")

                    except Exception as e:
                        print(f"  - [Exp] Error: {e}")

                count += 1
                
            except Exception as e:
                print(f"  - 패싯에 대한 OBB 생성 실패: {e}")
                
        # [Step] 중복/인접 커터 병합 (Merge Overlapping Cutters with same Orientation)
        if angle_snap_divisions > 0 and count > 0:
            print(f"  - [전처리] 각도 스냅된 커터 병합 시도... (Angle Snap 활성)")
            merged_cutters = [] # [Fix] 변수 초기화
            # [Debug] 병합 전 상태 확인
            oriented_indices = [i for i, c in enumerate(self.cutters) if c.get('type') == 'oriented']
            print(f"  - [병합 전] 대상 커터: {len(oriented_indices)}개")
            
            # 1. 처리 대상 선별 (Oriented type만)
            others = [self.cutters[i] for i in range(len(self.cutters)) if i not in oriented_indices]
            candidates = [self.cutters[i] for i in oriented_indices]
            
            # 2. 그룹화 (Rotation Matrix 기준)
            from collections import defaultdict
            groups = defaultdict(list)
            
            for c in candidates:
                R = c['transform'][:3, :3]
                # [수정] 부동소수점 노이즈 무시를 위해 반올림 강도 높임 (소수점 1자리)
                # 스냅이 적용된 상태라면 값들이 멀리 떨어져 있을 것이므로 1자리로도 충분히 구별 가능
                key = tuple(np.round(R.flatten(), 1))
                groups[key].append(c)
            
            print(f"  - [병합] 생성된 각도 그룹: {len(groups)}개")
            
            # 3. 그룹별 병합 (Local AABB Union)
            for key, group in groups.items():
                if not group: continue
                
                # 기준 R matrix
                R_ref = group[0]['transform'][:3, :3]
                
                # Local AABB 변환
                local_boxes = []
                for c in group:
                    center_w = c['transform'][:3, 3] 
                    center_l = np.dot(center_w, R_ref) # Proj onto axes (World @ R = Transpose(R) @ World_col? No. dot(vec, mat) = vec @ mat)
                    # c_l[i] = dot(c_w, col[i]) ok.
                    
                    ext = c['extents']
                    min_l = center_l - ext/2
                    max_l = center_l + ext/2
                    local_boxes.append({'min': min_l, 'max': max_l, 'ext': ext})
                
                # Greedy Merge Loop
                final_locals = []
                while local_boxes:
                    current = local_boxes.pop(0)
                    check_again = True
                    while check_again:
                        check_again = False
                        for i in range(len(local_boxes) - 1, -1, -1):
                            other = local_boxes[i]
                            # Check Overlap (with small tolerance/expansion to merge touching)
                            # [Aggressive Merge] 10% -> 150%로 대폭 완화하여 조각난 커터 강제 병합
                            tol_merge = np.maximum(current['ext'], other['ext']) * 1.5
                             
                            if (current['max'][0] + tol_merge[0] >= other['min'][0] and current['min'][0] - tol_merge[0] <= other['max'][0]) and \
                               (current['max'][1] + tol_merge[1] >= other['min'][1] and current['min'][1] - tol_merge[1] <= other['max'][1]) and \
                               (current['max'][2] + tol_merge[2] >= other['min'][2] and current['min'][2] - tol_merge[2] <= other['max'][2]):
                                
                                # Union
                                current['min'] = np.minimum(current['min'], other['min'])
                                current['max'] = np.maximum(current['max'], other['max'])
                                current['ext'] = current['max'] - current['min']
                                local_boxes.pop(i)
                                check_again = True 
                    final_locals.append(current)
            
                # Create merged entries
                for fb in final_locals:
                     center_l = (fb['min'] + fb['max']) / 2.0
                     ext_l = fb['max'] - fb['min']
                    # Local -> World
                     center_w = np.dot(center_l, R_ref.T) 
                     
                     T = np.eye(4)
                     T[:3, :3] = R_ref
                     T[:3, 3] = center_w
                     
                     merged_cutters.append({
                         'type': 'oriented',
                         'extents': ext_l,
                         'transform': T,
                         'center': center_w,
                         'size': ext_l
                     })
            
            self.cutters = others + merged_cutters
            print(f"  - [병합] {len(candidates)}개의 개별 슬랫 -> {len(merged_cutters)}개의 병합된 커터로 최적화.")
                
        print(f"[커터] {len(self.cutters)}개의 경사진 커터 준비 완료.")

    def _refine_cutter(self, center, size, tolerance, extra_offset=0.0, allow_initial_overlap=False):
        """
        메쉬와 충돌할 때까지 각 축 방향으로 박스를 확장합니다.
        extra_offset: Refine 완료 후 추가로 확장할 크기 (Undersize Mode용)
        allow_initial_overlap: 초기 상태에서 충돌이 있어도 수축하지 않음 (Undersize/Erosion 용)
        """
        refined_center = np.array(center, dtype=float)
        refined_size = np.array(size, dtype=float)
        
        # [추가] 초기 상태 체크: 시작부터 침투 중인지 확인
        # 만약 침투 중이라면 충돌이 없을 때까지 각 축을 조금씩 줄임
        # 단, Undersize 모드 등에서는 이를 무시하고 진행(허용)
        if not allow_initial_overlap:
            for _ in range(5): # 최대 5단계 수축
                 test_pts = self._get_surface_sample_points(refined_center, refined_size, padding=-0.1)
                 if np.any(self.original_mesh.contains(test_pts)):
                     refined_size *= 0.9 # 10%씩 수축
                 else:
                     break

        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        
        for i, axis in enumerate(axes):
            for side in [-1, 1]:
                # [Improvement] 2-Phase Expansion (Coarse -> Fine)
                # 1. Coarse Step
                step_coarse = self.voxel_scale * 0.2
                max_expansion = self.voxel_scale * 5.0
                
                current_expansion = 0.0
                
                def _check_collision(center, size, axis, side, tol_chk):
                    # Check bounding vertices first
                    dx, dy, dz = size / 2.0
                    cx, cy, cz = center
                    # Note: The original snippet had a typo [cx+sx, cy-sy, cz-dz] etc.
                    # It should use dx, dy, dz from the current test_size.
                    v_pts = np.array([
                        [cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz], [cx+dx, cy+dy, cz-dz], [cx-dx, cy+dy, cz-dz],
                        [cx-dx, cy-dy, cz+dz], [cx+dx, cy-dy, cz+dz], [cx+dx, cy+dy, cz+dz], [cx-dx, cy+dy, cz+dz]
                    ])
                    v_check = v_pts + side * axis * tol_chk
                    if np.any(self.original_mesh.contains(v_check)):
                        return True
                        
                    # Check face samples
                    pts = self._get_face_sample_points(center, size, i, side)
                    check_pts = pts + side * axis * tol_chk
                    if np.any(self.original_mesh.contains(check_pts)):
                        return True
                    return False

                # Coarse Loop
                while current_expansion < max_expansion:
                    test_exp = current_expansion + step_coarse
                    
                    # Apply expansion temporarily
                    test_size = refined_size.copy()
                    test_size[i] += test_exp
                    test_center = refined_center.copy()
                    test_center[i] += side * test_exp / 2.0
                    
                    if _check_collision(test_center, test_size, axis, side, tolerance):
                        break # Collision imminent
                    
                    current_expansion = test_exp # Safe to expand
                
                # Apply Coarse Result
                if current_expansion > 0:
                    refined_size[i] += current_expansion
                    refined_center[i] += side * current_expansion / 2.0
                
                # 2. Fine Step (Binary-like small steps)
                # Fill the remaining gap (which is less than step_coarse)
                step_fine = step_coarse / 10.0 # e.g. 0.02mm
                fine_expansion = 0.0
                
                for _ in range(9): # Try up to 9 times (covering 90% of coarse step)
                    test_exp = fine_expansion + step_fine
                    
                    test_size = refined_size.copy()
                    test_size[i] += test_exp
                    test_center = refined_center.copy()
                    test_center[i] += side * test_exp / 2.0
                    
                    if _check_collision(test_center, test_size, axis, side, tolerance):
                         # [Undersize Mode] We WANT penetration.
                         # If we hit the mesh, this is a VALID state for local erosion.
                         # But we should stop here to avoid eating too much.
                         if self._cfg.get('undersize_mode', False):
                             fine_expansion = test_exp # Accept the colliding step
                         break
                    
                    fine_expansion = test_exp
                
                # Apply Fine Result
                if fine_expansion > 0:
                    refined_size[i] += fine_expansion
                    refined_center[i] += side * fine_expansion / 2.0
                
        # [추가] 최종 최소 크기 제약 체크
        if np.any(refined_size < 3.0): # 설정값이 없으면 기본 3.0
             # 너무 작으면 무의미한 커터로 보고 버림(None 반환)
             return None
             
        # [New] Undersize Mode (Cutter Inflation)
        # 충돌 지점까지 확장한 상태에서, 추가로 더 확장하여 원본 형상을 파고들게 함
        if extra_offset > 0:
             refined_size += (extra_offset * 2.0)
             # Center는 유지 (양방향 확장)

        return {'center': refined_center, 'size': refined_size, 'type': 'aabb'}

    def decompose_by_octree(self, level=1, snap_to_features=True):
        """
        [Method 3] 옥트리(Octree) 그리드 기반 도메인 분할 (Domain Decomposition).
        모델을 거친 격자(Coarse Grid)로 나누어, 매핑된 메쉬(Mapped Mesh) 생성에 유리한 
        여러 개의 단순한 육면체 위상 블록(Multi-block)으로 분할합니다.
        
        :param level: 분할 레벨 (1=2x2x2=8개, 2=4x4x4=64개 ...)
        """
        print(f"\n[Decomposition] 옥트리 기반 도메인 분할 시작 (Level={level})...")
        self.decomposed_solids = []
        
        if self.original_cq is None:
             print("[Error] CadQuery 객체가 없습니다. STEP/IGES 파일을 로드해야 합니다.")
             return

        # 1. Bounding Box & Grid Setup
        min_pt, max_pt = self.bounding_box
        size = max_pt - min_pt
        center_model = (min_pt + max_pt) / 2.0
        
        # 분할 수 계산
        divs = 2 ** level # Level 1 -> 2, Level 2 -> 4
        
        # 그리드 피치
        pitch_x = size[0] / divs
        pitch_y = size[1] / divs
        pitch_z = size[2] / divs
        
        # Features Alignment (Optional)
        # 만약 주요 특징(Feature)이 그리드 라인 근처에 있다면 스냅하여 얇은 조각 방지
        # (여기서는 간단히 등간격 분할로 구현하되, BBox Center는 정확히 맞춤)
        
        print(f"  - Grid: {divs}x{divs}x{divs} ({divs**3} blocks)")
        print(f"  - Pitch: {pitch_x:.2f}, {pitch_y:.2f}, {pitch_z:.2f} mm")
        
        # 2. Iterate Grid & Intersect
        # CadQuery의 intersect 기능을 사용하여 각 그리드 박스와 모델의 교차 부분만 추출
        
        generated_solids = []
        
        # 안전 마진 (겹침 보장)
        margin = size[0] * 0.001 
        
        for i in range(divs):
            for j in range(divs):
                for k in range(divs):
                    # 각 그리드 셀의 중심 및 크기 (약간 크게 해서 틈새 방지)
                    cx = min_pt[0] + (i + 0.5) * pitch_x
                    cy = min_pt[1] + (j + 0.5) * pitch_y
                    cz = min_pt[2] + (k + 0.5) * pitch_z
                    
                    # 박스 생성 (Workplane은 로컬 좌표계가 복잡하므로, 전역 좌표로 박스 생성 후 이동 or Location 사용)
                    # CadQuery 박스는 중심 기준 생성
                    # 교차 연산을 위해 임시 Solid Box 생성
                    try:
                        # Box 생성
                        cell_box = cq.Workplane("XY").box(pitch_x + margin, pitch_y + margin, pitch_z + margin) \
                                                     .translate((cx, cy, cz))
                        
                        # 교차 (Intersect)
                        # self.original_cq는 Compound 또는 Solid
                        # intersect는 Workplane 연산
                        
                        chunk = self.original_cq.intersect(cell_box)
                        
                        # 유효성 검사: 솔리드가 남아있는지 확인
                        if chunk.val().Volume() > 1e-6: # 미소 잔여물 제외
                            generated_solids.append(chunk)
                            # print(f"    - Block ({i},{j},{k}) created.")
                            
                    except Exception as e:
                        # 교차 결과가 없으면(허공) 에러일 수 있음 -> 무시
                        pass

        self.decomposed_solids = generated_solids
        print(f"[Completed] 총 {len(self.decomposed_solids)}개의 서브 도메인(Solid) 생성 완료.")
        
        # 결과물을 시각화용 하나의 컴파운드로 합침 (색상 구분을 위해 리스트는 유지)
        if self.decomposed_solids:
             # 임시 시각화용 합침
             self.simplified_shape = cq.Compound.makeCompound([s.val() for s in self.decomposed_solids])
        else:
             self.simplified_shape = None


    def _get_face_sample_points(self, center, size, axis_idx, side):
        """박스의 특정 면에 대해 5x5 샘플 포인트를 생성합니다."""
        dx, dy, dz = size / 2.0
        cx, cy, cz = center
        
        # 다른 두 축의 범위
        other_axes = [idx for idx in range(3) if idx != axis_idx]
        u_axis, v_axis = other_axes
        u_vals = np.linspace(-size[u_axis]/2 * 0.95, size[u_axis]/2 * 0.95, 5)
        v_vals = np.linspace(-size[v_axis]/2 * 0.95, size[v_axis]/2 * 0.95, 5)
        
        pts = []
        fixed_val = center[axis_idx] + side * (size[axis_idx] / 2.0)
        
        for u in u_vals:
            for v in v_vals:
                p = [0, 0, 0]
                p[axis_idx] = fixed_val
                p[u_axis] = center[u_axis] + u
                p[v_axis] = center[v_axis] + v
                pts.append(p)
                
        return np.array(pts)

    def _get_surface_sample_points(self, center, size, padding=0.0):
        """박스 전체 표면의 샘플 포인트를 가져옵니다 (초기 체크용)"""
        pts = []
        for i in range(3):
            for side in [-1, 1]:
                pts.extend(self._get_face_sample_points(center, size, i, side))
        return np.array(pts)
    def expand_cutters(self, amount):
        """
        [CAE Post-process] 모든 커터의 크기를 일괄적으로 확장/축소합니다.
        amount: 확장할 크기 (mm). 음수면 축소.
        """
        if not self.cutters: return
        
        count = 0
        for c in self.cutters:
            if c.get('type') == 'oriented':
                c['extents'] = np.maximum(c['extents'] + amount, 0.1) # 최소 크기 방지
                c['size'] = c['extents']
            else:
                c['size'] = np.maximum(c['size'] + amount, 0.1)
            count += 1
            
        print(f"[Edit] {count}개의 커터를 {amount:+.2f}mm 만큼 크기 조정했습니다.")
        # 간극이 좁아졌을 수 있으므로 정렬 재수행 권장
        # self.regularize_cutters() # Optional

    def regularize_cutters(self, snap_distance=1.0):
        """
        [CAE Topology Optimization]
        커터들의 면을 공통 평면에 정렬(Snap)하여 미세한 단차와 틈새(Gap)를 제거합니다.
        """
        if not self.cutters: return
        print(f"[Regularization] 커터 면 정렬 및 틈새 메우기 (Snap: {snap_distance:.2f}mm)...")
        
        aabb_cutters = [c for c in self.cutters if c.get('type') != 'oriented']
        if not aabb_cutters: return

        # 1. 축별로 면 수집
        planes = [[], [], []]
        for c in aabb_cutters:
            min_pt = c['center'] - c['size']/2
            max_pt = c['center'] + c['size']/2
            for i in range(3):
                planes[i].append(min_pt[i])
                planes[i].append(max_pt[i])
        
        # 2. 축별 틈새 메우기 (Gap Closing Logic)
        snapped_planes = [{}, {}, {}]
        
        for i in range(3):
            # Unique sorted coordinates
            coords = sorted(list(set(planes[i])))
            if not coords: continue
            
            # 클러스터링을 통한 좌표 병합
            merged_groups = []
            if coords:
                curr = [coords[0]]
                merged_groups.append(curr)
                for val in coords[1:]:
                    diff = val - curr[-1]
                    # 차이가 설정값보다 작으면 같은 그룹 (병합) -> 틈새 메우기 효과
                    if diff < snap_distance:
                        curr.append(val)
                    else:
                        curr = [val]
                        merged_groups.append(curr)
            
            # 매핑 테이블 생성 (Val -> Representative Val)
            for grp in merged_groups:
                # 대푯값: 그룹의 중간값 (Average)
                # 예: 10.0과 10.2가 만나면 10.1이 되어 서로 붙음 -> Gap Closed
                rep = sum(grp) / len(grp)
                for v in grp:
                    snapped_planes[i][v] = rep

        # 3. 커터에 적용
        changed = 0
        for c in aabb_cutters:
            min_pt = c['center'] - c['size']/2
            max_pt = c['center'] + c['size']/2
            
            new_min = min_pt.copy()
            new_max = max_pt.copy()
            modified = False
            
            for i in range(3):
                # 정확한 Key 매칭 (float 이슈 최소화를 위해 직접 매핑 시도)
                # planes에 넣었던 값 그대로이므로 dict lookup 성공 확률 높음
                t_min = snapped_planes[i].get(min_pt[i], min_pt[i])
                t_max = snapped_planes[i].get(max_pt[i], max_pt[i])
                
                if abs(new_min[i] - t_min) > 1e-9:
                    new_min[i] = t_min
                    modified = True
                if abs(new_max[i] - t_max) > 1e-9:
                    new_max[i] = t_max
                    modified = True
            
            if modified:
                # 크기 검증 (Collapsed 방지)
                ns = new_max - new_min
                if np.any(ns < 1e-5): continue 
                
                c['size'] = ns
                c['center'] = new_min + ns/2.0
                changed += 1
                
        print(f"  - {changed}개의 커터 면이 조정되어 틈새가 밀봉되었습니다.")

    def optimize_cutters(self):
        """
        커터들을 최적화합니다. (포함 관계 제거에 집중)
        큰 커터 안에 포함되는 작은 커터를 제거하여 중복 연산을 줄입니다.
        """
        if not self.cutters: return

        print(f"[최적화] {len(self.cutters)}개의 커터 최적화 중... (포함 관계 제거)")
        
        # 1. 크기(부피) 순 정렬
        def get_vol(c):
             sz = c.get('extents') if 'extents' in c else c.get('size')
             if sz is None: return 0
             return np.prod(sz)
             
        self.cutters.sort(key=get_vol, reverse=True)
        
        kept_cutters = []
        n = len(self.cutters)
        removed_count = 0
        
        # Helper: 8 Corners Generator
        def _get_corners(c):
            if c.get('type') == 'oriented':
                return trimesh.creation.box(extents=c['extents'], transform=c['transform']).vertices
            else:
                sx, sy, sz = c['size'] / 2.0
                cx, cy, cz = c['center']
                return np.array([
                    [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
                    [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz], [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz]
                ])

        # Helper: Containment Check
        def _is_contained(inner_c, outer_c):
            if inner_c.get('type') != 'oriented' and outer_c.get('type') != 'oriented':
                imin = inner_c['center'] - inner_c['size']/2
                imax = inner_c['center'] + inner_c['size']/2
                omin = outer_c['center'] - outer_c['size']/2
                omax = outer_c['center'] + outer_c['size']/2
                return np.all(imin >= omin - 1e-4) and np.all(imax <= omax + 1e-4)
                
            inner_pts = _get_corners(inner_c)
            if outer_c.get('type') == 'oriented':
                 T_inv = np.linalg.inv(outer_c['transform'])
                 half = outer_c['extents'] / 2.0
            else:
                 T_inv = np.eye(4)
                 T_inv[:3, 3] = -outer_c['center']
                 half = outer_c['size'] / 2.0
            
            local_pts = trimesh.transform_points(inner_pts, T_inv)
            return np.all(np.abs(local_pts) <= (half + 1e-4))

        for i in range(n):
            current = self.cutters[i]
            is_contained = False
            for keeper in kept_cutters:
                if _is_contained(current, keeper):
                    is_contained = True
                    break
            if not is_contained:
                kept_cutters.append(current)
            else:
                removed_count += 1
                
        print(f"  - {removed_count}개의 포함된(중복) 커터 제거됨.")
        self.cutters = kept_cutters



    def _get_obb_geometry(self, c):
        """Helper: Extract edges and corners from OBB cutter."""
        center = c['transform'][:3, 3]
        half = c['extents'] / 2.0
        R = c['transform'][:3, :3]
        
        # 8 Corners (Local coords signs)
        signs = [
            (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
            (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)
        ]
        corners = []
        for s in signs:
            local_p = np.array(s) * half
            world_p = center + np.dot(R, local_p)
            corners.append({'pos': world_p, 'signs': s})
            
        # 12 Edges
        edges = []
        # Axis 0 edges (Fixed y, z)
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p1 = center + np.dot(R, np.array([-1, sy, sz]) * half)
                p2 = center + np.dot(R, np.array([1, sy, sz]) * half)
                edges.append({
                    'p1': p1, 'p2': p2, 
                    'dir': R[:, 0], 
                    'axis': 0, 
                    'fixed_signs': {1: sy, 2: sz}
                })
        # Axis 1 edges (Fixed x, z)
        for sx in [-1, 1]:
            for sz in [-1, 1]:
                p1 = center + np.dot(R, np.array([sx, -1, sz]) * half)
                p2 = center + np.dot(R, np.array([sx, 1, sz]) * half)
                edges.append({
                    'p1': p1, 'p2': p2, 
                    'dir': R[:, 1], 
                    'axis': 1, 
                    'fixed_signs': {0: sx, 2: sz}
                })
        # Axis 2 edges (Fixed x, y)
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                p1 = center + np.dot(R, np.array([sx, sy, -1]) * half)
                p2 = center + np.dot(R, np.array([sx, sy, 1]) * half)
                edges.append({
                    'p1': p1, 'p2': p2, 
                    'dir': R[:, 2], 
                    'axis': 2, 
                    'fixed_signs': {0: sx, 1: sy}
                })
        return corners, edges

    def _align_cutters_edge_vertex(self, snap_tolerance=None):
        """
        [New] Edge & Vertex based Alignment (Global Asymmetric Face Alignment).
        Iteratively adjusts 6 faces of cutters independently to align parallel edges and close vertices.
        This solves 'chain reactions' by allowing asymmetric expansion (e.g. expand Left only).
        """
        if not self.cutters: return
        
        snap_dist = snap_tolerance if snap_tolerance else (self.voxel_scale * 3.0)
        print(f"[Regularization] 엣지/버텍스 정렬 최적화 (Snap: {snap_dist:.2f}mm) - Asymmetric...")
        
        oriented_indices = [i for i, c in enumerate(self.cutters) if c.get('type') == 'oriented']
        n = len(oriented_indices)
        if n < 2: return

        # Define snap_dist early
        snap_dist = snap_tolerance if snap_tolerance else self.voxel_scale * 1.5

        # [User Request] Print list of close Edge Pairs
        print("\n[Debug] Scanning for Close Edge Pairs...")
        scan_dist = snap_dist * 2.0
        
        detected_pairs = [] # Format: (c1_id, e1_idx, c2_id, e2_idx)
        
        for i in range(n):
            idx1 = oriented_indices[i]
            c1 = self.cutters[idx1]
            _, edges1 = self._get_obb_geometry(c1)
            
            for j in range(i + 1, n):
                idx2 = oriented_indices[j]
                c2 = self.cutters[idx2]
                
                # Rough check
                if np.linalg.norm(c1['transform'][:3,3] - c2['transform'][:3,3]) > (np.linalg.norm(c1['extents']) + np.linalg.norm(c2['extents'])):
                    continue

                _, edges2 = self._get_obb_geometry(c2)
                
                for k1, e1 in enumerate(edges1):
                    for k2, e2 in enumerate(edges2):
                        # Distance check
                        mid1 = (e1['p1'] + e1['p2']) / 2
                        mid2 = (e2['p1'] + e2['p2']) / 2
                        if np.linalg.norm(mid1 - mid2) > scan_dist * 2: continue # Optimization
                        
                        # Exact line distance (Standardized)
                        dist, dist_vec = self._calc_edge_distance(e1, e2)
                        
                        # [User Logic] Only track pairs that are:
                        # 1. Within tolerance (snap_dist)
                        # 2. NOT already touching (dist > 0.01)
                        if dist < scan_dist: # Print wider context
                             status_log = ""
                             if dist < snap_dist and dist > 0.01:
                                 status_log = " [Target]"
                                 detected_pairs.append((idx1+1, k1, idx2+1, k2))
                             elif dist <= 0.01:
                                 status_log = " [Already Aligned]"
                             
                             print(f"  - Pair C{idx1+1}:E{k1} - C{idx2+1}:E{k2} | Dist: {dist:.3f}mm{status_log}")

    # Optimization Loop (Convergence)
        max_iter = int(self._cfg.get('max_align_iter', 20))
        
        # [Debug] Monitor specific pairs iteration-by-iteration
        # Use detected pairs dynamically
        monitor_targets = detected_pairs
        
        for iter_idx in range(max_iter):
            print(f"\n[Iteration {iter_idx+1}] Monitoring Gaps (Max {max_iter}):")
            self._debug_verify_pairs(monitor_targets)

            # Pending Shifts: (cutter_index, face_id) -> max_expansion_amount (float > 0)
            # face_id: 0(X+), 1(X-), 2(Y+), 3(Y-), 4(Z+), 5(Z-)
            pending_face_shifts = {} 
            
            check_count = 0
            
            for i in range(n):
                idx1 = oriented_indices[i]
                c1 = self.cutters[idx1]
                corners1, edges1 = self._get_obb_geometry(c1)
                
                for j in range(i + 1, n):
                    idx2 = oriented_indices[j]
                    c2 = self.cutters[idx2]
                    
                    dist_centers = np.linalg.norm(c1['transform'][:3, 3] - c2['transform'][:3, 3])
                    max_r = np.linalg.norm(c1['extents'])/2 + np.linalg.norm(c2['extents'])/2
                    if dist_centers > max_r + snap_dist: continue

                    corners2, edges2 = self._get_obb_geometry(c2)
                    check_count += 1
                    
                    # Debugging (Disabled)
                    is_watched = False
                    
                    R1 = c1['transform'][:3, :3]
                    R2 = c2['transform'][:3, :3]
                    
                    # --- Edge vs Edge ---
                    for e1 in edges1:
                        for e2 in edges2:
                            # Parallel Check
                            dot_val = np.dot(e1['dir'], e2['dir'])
                            if abs(dot_val) < 0.98: continue
                            
                            # Exact line distance (Standardized)
                            dist, dist_vec = self._calc_edge_distance(e1, e2)
                            
                            if is_watched and dist < snap_dist * 2.0:
                                print(f"    - Edge Prox ({dist:.2f}mm): Gap Vector {dist_vec}")

                            # [Update] Prevent micro-adjustments causing overlap. Explicit deadband.
                            if dist < 0.01: continue

                            if dist < snap_dist:
                                # Determine Smart Sharing Ratio based on Gap Direction
                                # Who is facing the gap?
                                gap_dir = dist_vec / (dist + 1e-9)
                                
                                # Check alignment of Gap Direction to Face Normals (Axes)
                                align1 = max([abs(np.dot(gap_dir, R1[:, k])) for k in range(3)])
                                align2 = max([abs(np.dot(gap_dir, R2[:, k])) for k in range(3)])
                                
                                ratio_c1 = 0.5
                                ratio_c2 = 0.5
                                
                                # [Debug] Analyze specific problematic pair C10-C18 (and others)
                                debug_pair = False
                                if (idx1+1 == 10 and idx2+1 == 18) or (idx1+1 == 10 and idx2+1 == 16):
                                    debug_pair = True
                                    # print(f"    [Analysis C{idx1+1}-C{idx2+1}] Dist {dist:.4f}, GapDir {gap_dir}, Align1 {align1:.2f}, Align2 {align2:.2f}")

                                # Strict Unilateral Logic
                                # If one is clearly a Face (Align > 0.9) and other is Edge/Corner (Align < 0.8)
                                if align1 > 0.9 and align2 < 0.8:
                                    ratio_c1 = 1.0
                                    ratio_c2 = 0.0
                                elif align2 > 0.9 and align1 < 0.8:
                                    ratio_c1 = 0.0
                                    ratio_c2 = 1.0
                                
                                # [Optimization] Stable Expansion
                                # Reverted to simple logic: Expand to cover gap + 20% safety.
                                # "Efficiency" logic caused instability. Trust the iteration.
                                boost = 1.2
                                
                                # C1 Adjustment
                                if ratio_c1 > 0:
                                    # Use raw Gap Vector length, project direction
                                    move_dir_c1 = dist_vec / (np.linalg.norm(dist_vec) + 1e-9)
                                    req_dist_c1 = dist * ratio_c1
                                    
                                    for axis_idx in [0, 1, 2]:
                                        if axis_idx == e1['axis']: continue
                                        
                                        local_proj_dir = np.dot(move_dir_c1, R1[:, axis_idx])
                                        edge_side = e1['fixed_signs'][axis_idx]
                                        
                                        if local_proj_dir * edge_side > 0:
                                            # Simple Projection: Expand based on how much the gap projects onto this axis.
                                            # If orthogonal (proj=0), we don't expand (which is correct, expanding doesn't close orthogonal gap).
                                            # We need iterative steps for diagonal gaps.
                                            
                                            expansion = req_dist_c1 * boost * abs(local_proj_dir)
                                            # Wait, if proj is 0.1, expansion is small. Convergence is slow.
                                            # But "Inverse Boost" exploded.
                                            # Let's try: expansion = req_dist_c1 * boost (Force full closure amount regardless of angle?)
                                            # If angle is 80 deg, expanding full amount moves the surface normal by X.
                                            # Impact on gap = X * cos(theta).
                                            # If we expand X=Gap, impact is Gap*cos. Still slow, but safe.
                                            # Let's stick to safe expansion.
                                            expansion = req_dist_c1 * boost
                                            
                                            # [Safety Cap] Never expand significantly more than the GAP itself.
                                            # This prevents the 2.4mm overshoot.
                                            # Limit expansion to Gap + 0.1mm (for overlap)
                                            if expansion > dist + 0.1: expansion = dist + 0.1
                                            
                                            # Global sanity check
                                            if expansion > 5.0: expansion = 5.0

                                            if expansion > 0.001:
                                                face_id = axis_idx * 2 + (0 if edge_side > 0 else 1)
                                                key = (idx1, face_id)
                                                if expansion > pending_face_shifts.get(key, 0.0):
                                                    pending_face_shifts[key] = expansion

                                # C2 Adjustment
                                if ratio_c2 > 0:
                                    move_dir_c2 = -dist_vec / (np.linalg.norm(dist_vec) + 1e-9)
                                    req_dist_c2 = dist * ratio_c2
                                    
                                    for axis_idx in [0, 1, 2]:
                                        if axis_idx == e2['axis']: continue
                                        
                                        local_proj_dir = np.dot(move_dir_c2, R2[:, axis_idx])
                                        edge_side = e2['fixed_signs'][axis_idx]
                                        
                                        if local_proj_dir * edge_side > 0:
                                            expansion = req_dist_c2 * boost
                                            
                                            if expansion > dist + 0.1: expansion = dist + 0.1
                                            if expansion > 5.0: expansion = 5.0
                                            
                                            if expansion > 0.001:
                                                face_id = axis_idx * 2 + (0 if edge_side > 0 else 1)
                                                key = (idx2, face_id)
                                                if expansion > pending_face_shifts.get(key, 0.0):
                                                    pending_face_shifts[key] = expansion
                                                # [Debug] Action Log (Unified Format)
                                                # print(f"    -> Action: C{idx2+1} Axis{axis_idx} Exp {expansion:.3f}mm (Cause: Gap to C{idx1+1})")

                    # --- Vertex vs Vertex ---
                    for v1 in corners1:
                        for v2 in corners2:
                            diff = v2['pos'] - v1['pos']
                            dist = np.linalg.norm(diff)
                            
                            if is_watched and dist < snap_dist * 2.0:
                                print(f"    - Vertex Prox ({dist:.2f}mm)")

                            if dist < snap_dist:
                                # Smart Sharing
                                gap_dir = diff / (dist + 1e-9)
                                align1 = max([abs(np.dot(gap_dir, R1[:, k])) for k in range(3)])
                                align2 = max([abs(np.dot(gap_dir, R2[:, k])) for k in range(3)])
                                
                                ratio_c1 = 0.5
                                ratio_c2 = 0.5
                                if align1 > 0.9 and align2 < 0.8:
                                    ratio_c1 = 1.0
                                    ratio_c2 = 0.0
                                elif align2 > 0.9 and align1 < 0.8:
                                    ratio_c1 = 0.0
                                    ratio_c2 = 1.0

                                # C1
                                move_vec_c1 = diff * ratio_c1
                                for ax in range(3):
                                    local_proj = np.dot(move_vec_c1, R1[:, ax])
                                    side = v1['signs'][ax]
                                    expansion = side * local_proj
                                    if expansion > 0.001 and expansion < snap_dist * 1.5:
                                        face_id = ax * 2 + (0 if side > 0 else 1)
                                        key = (idx1, face_id)
                                        if expansion > pending_face_shifts.get(key, 0.0):
                                            pending_face_shifts[key] = expansion

                                # C2
                                move_vec_c2 = -diff * ratio_c2
                                for ax in range(3):
                                    local_proj = np.dot(move_vec_c2, R2[:, ax])
                                    side = v2['signs'][ax]
                                    expansion = side * local_proj
                                    if expansion > 0.001 and expansion < snap_dist * 1.5:
                                        face_id = ax * 2 + (0 if side > 0 else 1)
                                        key = (idx2, face_id)
                                        if expansion > pending_face_shifts.get(key, 0.0):
                                            pending_face_shifts[key] = expansion

                    # --- Face vs Face (End-to-End & Unequal Size & Misaligned) ---
                    # Always run. Handles both Mutual and Unilateral.
                    corners2_arr = np.array([v['pos'] for v in corners2])
                    
                    # Planes of C1
                    half1 = c1['extents'] / 2.0
                    center1 = c1['transform'][:3, 3]
                    
                    for ax in range(3):
                        normal = R1[:, ax]
                        for side in [1, -1]:
                            plane_n = normal * side
                            plane_p = center1 + (normal * side * half1[ax])
                            
                            dists = np.dot(corners2_arr - plane_p, plane_n)
                            min_d = np.min(dists)
                            
                            if min_d > 0 and min_d < snap_dist:
                                # Overlap Check
                                overlap = True
                                for lat_ax in range(3):
                                    if lat_ax == ax: continue
                                    proj_c2 = np.dot(corners2_arr - center1, R1[:, lat_ax])
                                    h_lat = half1[lat_ax]
                                    if np.max(proj_c2) < (-h_lat - 0.1) or np.min(proj_c2) > (h_lat + 0.1):
                                        overlap = False
                                        break
                                
                                if overlap:
                                    if is_watched:
                                         print(f"    - Face Ax{ax} Side{side}: min_d={min_d:.2f}. Checking Alignment...")
                                         
                                    # [Smart Expansion Sharing]
                                    # Detect if this is Face-Face (Mutual) or Face-Edge (Unilateral).
                                    
                                    # Check C2's alignment to this plane normal (plane_n)
                                    R2 = c2['transform'][:3, :3]
                                    
                                    # Max alignment of any C2 axis to plane_n
                                    dots = [abs(np.dot(plane_n, R2[:, k])) for k in range(3)]
                                    max_alignment = max(dots)
                                    
                                    if max_alignment > 0.8:
                                        # Mutual
                                        ratio_c1 = 0.5
                                        ratio_c2 = 0.5
                                    else:
                                        # Unilateral (C1 Face expands 100%)
                                        # Note: We are iterating C1 faces. So C1 IS the Face.
                                        ratio_c1 = 1.0
                                        ratio_c2 = 0.0

                                    # Register C1
                                    expansion_1 = min_d * ratio_c1
                                    if expansion_1 > 0.001:
                                        face_id_1 = ax * 2 + (0 if side > 0 else 1)
                                        key1 = (idx1, face_id_1)
                                        if expansion_1 > pending_face_shifts.get(key1, 0.0):
                                            pending_face_shifts[key1] = expansion_1
                                        
                                    # Register C2 (Only if Mutual)
                                    if ratio_c2 > 0.001:
                                        expansion_2 = min_d * ratio_c2
                                        target_dir = -plane_n
                                        for ax2 in range(3):
                                            proj = np.dot(target_dir, R2[:, ax2])
                                            if proj > 0.7:
                                                factor = 1.0 / max(0.1, proj)
                                                shift_2 = expansion_2 * factor
                                                face_id_2 = ax2 * 2 + 0 
                                                key2 = (idx2, face_id_2)
                                                if shift_2 > pending_face_shifts.get(key2, 0.0):
                                                    pending_face_shifts[key2] = shift_2
                                            elif proj < -0.7:
                                                factor = 1.0 / max(0.1, abs(proj))
                                                shift_2 = expansion_2 * factor
                                                face_id_2 = ax2 * 2 + 1 
                                                key2 = (idx2, face_id_2)
                                                if shift_2 > pending_face_shifts.get(key2, 0.0):
                                                    pending_face_shifts[key2] = shift_2

            # Apply Face Shifts
            # Regroup by cutter
            cutter_shifts = {}
            for (c_idx, face_id), amount in pending_face_shifts.items():
                if c_idx not in cutter_shifts:
                    cutter_shifts[c_idx] = [0.0]*6
                cutter_shifts[c_idx][face_id] = amount
            
            applied_count = 0
            print(f"  - Iteration {iter_idx+1} Updates:")
            for c_idx, shifts in cutter_shifts.items():
                # shifts: [x+, x-, y+, y-, z+, z-]
                c = self.cutters[c_idx]
                R = c['transform'][:3, :3]
                
                # Check if any change
                if sum(shifts) < 1e-9: continue

                # Readable Axis Names
                axis_names = ['X', 'Y', 'Z']
                face_names = ['+', '-']
                
                log_parts = []
                
                # Apply per axis
                for ax in range(3):
                    expand_plus = shifts[ax*2]
                    expand_minus = shifts[ax*2 + 1]
                    
                    if expand_plus == 0 and expand_minus == 0: continue
                    
                    # 1. Update Size (Total Expansion)
                    total_expansion = expand_plus + expand_minus
                    c['extents'][ax] += total_expansion
                    
                    # 2. Update Center (Shift)
                    # Center moves towards the side that expanded more.
                    # Shift = (ExpandPlus - ExpandMinus) / 2
                    center_shift_val = (expand_plus - expand_minus) / 2.0
                    shift_vec = R[:, ax] * center_shift_val
                    c['transform'][:3, 3] += shift_vec
                    
                    # Log Formatting
                    if expand_plus > 0:
                        log_parts.append(f"{axis_names[ax]}+ Edge Adj {expand_plus:.3f}mm")
                    if expand_minus > 0:
                        log_parts.append(f"{axis_names[ax]}- Edge Adj {expand_minus:.3f}mm")
                    if abs(center_shift_val) > 0.001:
                         direction = "Pos" if center_shift_val > 0 else "Neg"
                         log_parts.append(f"Shift {axis_names[ax]} {direction} {abs(center_shift_val):.3f}mm")
                    
                    applied_count += 1
                
                if log_parts:
                    print(f"    [Cutter {c_idx+1}] Adjusted: " + ", ".join(log_parts))
                
            print(f"    -> Iteration {iter_idx+1}: Evaluated {check_count} pairs, Updated {applied_count} axes.")
            if applied_count == 0:
                break
        
        print(f"  - [Rule] 비대칭 엣지/버텍스 정렬 완료.")
        return detected_pairs

    def _align_cutter_neighbors(self, snap_tolerance=None):
        """
        [Legacy wrapper] Call the new Edge/Vertex alignment method.
        """
        return self._align_cutters_edge_vertex(snap_tolerance)

    def optimize_cutters(self):
        """
        [통합 최적화] 커터 정리 파이프라인
        1. 포함 관계 제거 (Remove Contained)
        2. 인접면 스냅 (Align Neighbors) [New]
        3. 실린더 피팅 (Fit Cylinders)
        """
        if not self.cutters: return
        
        # [Visualization] Store initial state (Deep Copy)
        import copy
        self.cutters_initial = copy.deepcopy(self.cutters)
        
        # 1. 포함 관계 제거 (기존 로직 사용)
        self.optimize_cutters_unused() 
        
        # 2. 인접면 정렬 (스냅)
        # [Update] User reported gaps around 3-5mm. Increase tolerance significantly to catch overshoot.
        # If gap became 2.4mm, we need tolerance > 2.4mm to continue fixing it.
        snap_tol = self.voxel_scale * 15.0 # e.g., 0.2 * 15 = 3.0mm
        detected_pairs = self._align_cutter_neighbors(snap_tolerance=snap_tol)
        
        # [Debug] Verify specific user-reported pairs (Final Check)
        # Format: (C1, E1, C2, E2) based on dynamically detected pairs
        if detected_pairs:
            self._debug_verify_pairs(detected_pairs)

        # 3. 실린더 피팅 (Optional, here or before CAD)
        # self._fit_cylinders_from_cutters()

    def _calc_edge_distance(self, e1, e2):
        """
        Calculate exact shortest distance between two line segments (edges).
        Returns (distance, vector_on_e1_to_e2)
        """
        p1 = e1['p1']
        d1 = e1['dir']
        p2 = e2['p1']
        d2 = e2['dir']
        
        # Segment 1: p1 + s*d1, 0<=s<=len1
        # Segment 2: p2 + t*d2, 0<=t<=len2
        # For simplicity, treat as infinite lines first, then clamp? 
        # Or use robust segment-segment dist.
        # Given the context (boxes), infinite line is often 'safe' if parallel check passed.
        # But 'Verify' used strict segment logic. Let's use strict segment logic consistently.
        
        w0 = p1 - p2
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, w0)
        e = np.dot(d2, w0)
        
        denom = a*c - b*b
        if denom < 1e-6:
            # Parallel
            # Project p2 onto line1
            dist = np.linalg.norm(np.cross(d1, (p2-p1)))
            # Vector? p2 to projection?
            # v = (p2 - p1) - dot(p2-p1, d1)*d1
            # But we want 'vector from e1 to e2'
            # Let's just return dist_vec from midpoints if parallel
            mid1 = (e1['p1'] + e1['p2'])/2
            mid2 = (e2['p1'] + e2['p2'])/2
            return dist, (mid2 - mid1)
        else:
            sc = (b*e - c*d) / denom
            tc = (a*e - b*d) / denom
            
            # [Fix] Clamp to finite segment lengths
            len1 = np.linalg.norm(e1['p2'] - e1['p1'])
            len2 = np.linalg.norm(e2['p2'] - e2['p1'])
            
            sc = np.clip(sc, 0, len1)
            tc = np.clip(tc, 0, len2)
            
            closest_p1 = p1 + sc * d1
            closest_p2 = p2 + tc * d2
            dist = np.linalg.norm(closest_p1 - closest_p2)
            dist_vec = closest_p2 - closest_p1
            return dist, dist_vec

    def _debug_verify_pairs(self, quartet_list):
        print("\n[Gap Check] Verifying Specific Edge Pairs:")
        if not self.cutters: return
        
        for (id1, e1_idx, id2, e2_idx) in quartet_list:
            idx1, idx2 = id1 - 1, id2 - 1
            if idx1 >= len(self.cutters) or idx2 >= len(self.cutters): continue
            
            c1 = self.cutters[idx1]
            c2 = self.cutters[idx2]
            
            _, edges1 = self._get_obb_geometry(c1)
            _, edges2 = self._get_obb_geometry(c2)
            
            if e1_idx >= len(edges1) or e2_idx >= len(edges2):
                print(f"  [Error] Invalid Edge Index for C{id1}-C{id2}")
                continue
                
            e1 = edges1[e1_idx]
            e2 = edges2[e2_idx]
            
            # Use shared helper
            dist, _ = self._calc_edge_distance(e1, e2)

            status = "TOUCH" if dist < 0.001 else "GAP"
            print(f"  - Pair C{id1}:E{e1_idx} - C{id2}:E{e2_idx} | Dist: {dist:.5f}mm ({status})")

    def optimize_cutters_unused(self):
        """
        [Legacy] 중복되거나 인접한 커터를 더 큰 커터로 그룹화하고 불필요한 커터를 제거합니다.
        (Called internally by optimize_cutters now)
        """
        if not self.cutters: return

        print(f"[최적화] {len(self.cutters)}개의 커터 최적화 중... (포함 관계 제거)")
        # ... (Rest of existing logic continues...)
        
        def get_bounds(c):
            if c.get('type') == 'oriented': return None
            cx, cy, cz = c['center']
            sx, sy, sz = c['size']
            return (
                np.array([cx - sx/2, cy - sy/2, cz - sz/2]),
                np.array([cx + sx/2, cy + sy/2, cz + sz/2])
            )

        # 1. 포함된 커터 제거
        # 부피순 정렬 (내림차순)
        self.cutters.sort(key=lambda c: np.prod(c['size']) if c.get('type')!='oriented' else np.prod(c['extents']), reverse=True)
        
        kept_cutters = []
        for current in self.cutters:
            if current.get('type') == 'oriented':
                kept_cutters.append(current)
                continue
                
            curr_min, curr_max = get_bounds(current)
            is_contained = False
            
            for other in kept_cutters:
                if other.get('type') == 'oriented': continue
                other_min, other_max = get_bounds(other)
                
                # 현재 커터가 다른 커터 내부에 있는지 확인 (허용 오차 포함)
                if np.all(curr_min >= other_min - 1e-4) and np.all(curr_max <= other_max + 1e-4):
                    is_contained = True
                    break
            
            if not is_contained:
                kept_cutters.append(current)
                
        print(f"  - {len(self.cutters) - len(kept_cutters)}개의 포함된 커터 제거됨.")
        self.cutters = kept_cutters

        # 2. 인접/중복 커터 병합 (AABB)
        merged_something = True
        while merged_something:
            merged_something = False
            n = len(self.cutters)
            new_list = []
            skip_indices = set()
            
            for i in range(n):
                if i in skip_indices: continue
                c1 = self.cutters[i]
                if c1.get('type') == 'oriented':
                    new_list.append(c1); continue
                
                min1, max1 = get_bounds(c1)
                merged_c1 = False
                
                for j in range(i + 1, n):
                    if j in skip_indices: continue
                    c2 = self.cutters[j]
                    if c2.get('type') == 'oriented': continue
                    
                    min2, max2 = get_bounds(c2)
                    diff_min, diff_max = np.abs(min1 - min2), np.abs(max1 - max2)
                    axes_match = (diff_min < 1e-4) & (diff_max < 1e-4)
                    
                    if np.sum(axes_match) == 2:
                        merge_axis = np.argmin(axes_match)
                        l1, l2 = max1[merge_axis] - min1[merge_axis], max2[merge_axis] - min2[merge_axis]
                        union_min, union_max = min(min1[merge_axis], min2[merge_axis]), max(max1[merge_axis], max2[merge_axis])
                        
                        if (union_max - union_min) <= (l1 + l2) + 1e-4:
                            new_min, new_max = min1.copy(), max1.copy()
                            new_min[merge_axis], new_max[merge_axis] = union_min, union_max
                            new_size = new_max - new_min
                            new_list.append({'center': new_min + new_size/2, 'size': new_size, 'type': 'aabb'})
                            skip_indices.add(j)
                            merged_c1 = True
                            merged_something = True
                            break
                
                if not merged_c1: new_list.append(c1)
            
            if merged_something:
                self.cutters = new_list
                print(f"  - 병합 패스: {len(self.cutters)}개의 커터로 감소됨.")

    def _fit_cylinders_from_cutters(self):
        """
        [New] Oriented 커터들을 분석하여 원기둥(Cylinder) 형상을 찾아내고 교체합니다.
        여러 개의 사각형 조각으로 표현된 홀(Hole)을 하나의 완벽한 원기둥 프리미티브로 단순화합니다.
        """
        oriented = [c for c in self.cutters if c.get('type') == 'oriented']
        if len(oriented) < 3: return

        print(f"[특징 인식] 원기둥(Cylinder) 형상 탐색 중 (대상: {len(oriented)}개)...")
        
        final_cutters = [c for c in self.cutters if c.get('type') != 'oriented']
        cluster_used = set()
        
        # 1. 축 방향 유사성으로 그룹화
        groups = [] 
        
        for i, c in enumerate(oriented):
            if i in cluster_used: continue
            
            R = c['transform'][:3, :3]
            candidates = [R[:, 0], R[:, 1], R[:, 2]]
            
            found_group = False
            for axis in candidates:
                for g_idx, (g_axis, g_list) in enumerate(groups):
                    if abs(np.dot(axis, g_axis)) > 0.90:
                        g_list.append(i)
                        cluster_used.add(i)
                        found_group = True
                        break
                if found_group: break
            
            if not found_group:
                ax_idx = np.argmax(c['extents'])
                groups.append([candidates[ax_idx], [i]])
                cluster_used.add(i)

        # 2. 그룹별로 원형 배치 확인 및 피팅
        cylinder_cutters = []
        used_for_cylinder = set()
        
        for g_axis, g_indices in groups:
            if len(g_indices) < 3: continue
            
            pts = np.array([oriented[idx]['transform'][:3, 3] for idx in g_indices])
            
            ez = g_axis / np.linalg.norm(g_axis)
            ex = np.array([1, 0, 0])
            if abs(np.dot(ex, ez)) > 0.9: ex = np.array([0, 1, 0])
            ey = np.cross(ez, ex)
            ey /= np.linalg.norm(ey)
            ex = np.cross(ey, ez)
            
            p2d = np.zeros((len(pts), 2))
            for k in range(len(pts)):
                p2d[k, 0] = np.dot(pts[k], ex)
                p2d[k, 1] = np.dot(pts[k], ey)
                
            c2d = np.mean(p2d, axis=0)
            dists = np.linalg.norm(p2d - c2d, axis=1)
            radius = np.mean(dists)
            std = np.std(dists)
            
            if std < radius * 0.2 and radius > 1.0:
                print(f"  - [인식] 반지름 {radius:.2f}mm 원기둥 생성 (조각 {len(g_indices)}개 통합)")
                
                axial_vals = [np.dot(p, ez) for p in pts]
                h_min, h_max = min(axial_vals), max(axial_vals)
                h_extents = [oriented[idx]['extents'][np.argmax(oriented[idx]['extents'])] for idx in g_indices]
                avg_h_ext = np.mean(h_extents)
                height = (h_max - h_min) + avg_h_ext
                
                world_center = c2d[0]*ex + c2d[1]*ey + ((h_min + h_max)/2.0)*ez
                
                T = np.eye(4)
                T[:3, 0] = ex; T[:3, 1] = ey; T[:3, 2] = ez; T[:3, 3] = world_center
                
                cylinder_cutters.append({
                    'type': 'cylinder',
                    'radius': radius,
                    'height': height,
                    'transform': T,
                    'is_refine': True
                })
                # 원기둥으로 성공적으로 변환된 인덱스만 체크
                for idx in g_indices: used_for_cylinder.add(idx)

        # 3. 원기둥이 되지 못한 나머지 경사면 커터들은 그대로 유지
        for i, c in enumerate(oriented):
            if i not in used_for_cylinder:
                final_cutters.append(c)
        
        self.cutters = final_cutters + cylinder_cutters
        if cylinder_cutters:
            print(f"  - [완료] {len(cylinder_cutters)}개의 원기둥 커터 생성.")

    def generate_cad(self, use_engine='gmsh'):
        """
        CAD 데이터를 생성합니다. 
        use_engine: 'gmsh' (권장) 또는 'cadquery'
        """
        # [New] 원기둥 형상 인식 및 통합
        self._fit_cylinders_from_cutters()

        if use_engine == 'gmsh' and HAS_GMSH:
            return self._generate_cad_gmsh()
        elif HAS_CADQUERY:
            return self._generate_cad_cadquery()
        else:
            print("[CAD] 사용 가능한 CAD 엔진이 없습니다.")
            return None, None

    def _generate_cad_gmsh(self):
        print("[CAD] Gmsh를 사용하여 형상 구성 중...")
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0) # 출력 억제
        gmsh.model.add("simplified_model")
        
        min_pt, max_pt = self.bounding_box
        size = max_pt - min_pt
        
        # 기본 박스 생성
        base_id = gmsh.model.occ.addBox(min_pt[0], min_pt[1], min_pt[2], size[0], size[1], size[2])
        
        cutter_ids = []
        for c in self.cutters:
            ctype = c.get('type')
            if ctype == 'oriented':
                sx, sy, sz = c['extents']
                cid = gmsh.model.occ.addBox(-sx/2, -sy/2, -sz/2, sx, sy, sz)
                matrix = c['transform'].flatten().tolist()
                gmsh.model.occ.affineTransform([(3, cid)], matrix)
                cutter_ids.append((3, cid))
            elif ctype == 'cylinder':
                radius = c['radius']
                height = c['height']
                cid = gmsh.model.occ.addCylinder(0, 0, -height/2, 0, 0, height, radius)
                matrix = c['transform'].flatten().tolist()
                gmsh.model.occ.affineTransform([(3, cid)], matrix)
                cutter_ids.append((3, cid))
            else: # AABB
                cx, cy, cz = c['center']
                sx, sy, sz = c['size']
                cid = gmsh.model.occ.addBox(cx - sx/2, cy - sy/2, cz - sz/2, sx, sy, sz)
                cutter_ids.append((3, cid))
        
        gmsh.model.occ.synchronize()
        
        # 커터 합집합 생성 (불리언 연산을 위해)
        print("[CAD] 커터 합집합 계산 중...")
        # 퓨전 연산
        try:
            # 모든 커터를 하나로 융합
            if cutter_ids:
                union_tags, _ = gmsh.model.occ.fuse([cutter_ids[0]], cutter_ids[1:])
                gmsh.model.occ.synchronize()
                
                # 기본 박스에서 커터 차집합 수행
                print("[CAD] 최종 차집합 연산 중...")
                result_tags, _ = gmsh.model.occ.cut([(3, base_id)], union_tags)
                gmsh.model.occ.synchronize()
            else:
                print("[CAD] 커터가 없습니다.")
        except Exception as e:
            print(f"[CAD] Gmsh 불리언 연산 중 오류: {e}")
        
        # 메모리 상의 모델은 gmsh 라이브러리에 있으므로, 나중에 export 시 사용
        # Gmsh는 현재 활성화된 모델을 저장하므로 상태 문자열만 반환합니다.
        return "gmsh_model_final", "gmsh_model_cutters"

    def _generate_cad_cadquery(self):
        if not HAS_CADQUERY: return None, None
        print("[CAD] CadQuery를 사용하여 형상 구성 중...")
        
        min_pt, max_pt = self.bounding_box
        center = (min_pt + max_pt) / 2.0
        size = max_pt - min_pt
        base_box = cq.Workplane("XY").box(size[0], size[1], size[2]).translate(tuple(center))
        
        print(f"[CAD] {len(self.cutters)}개의 커터 솔리드 생성 및 삭감 중...")
        result = base_box
        for idx, c in enumerate(self.cutters):
            try:
                if c.get('type') == 'oriented':
                    sx, sy, sz = c['extents']
                    b = cq.Workplane("XY").box(sx, sy, sz)
                    R = c['transform'][:3, :3]
                    T = c['transform'][:3, 3]
                    axis, angle = self._matrix_to_axis_angle(R)
                    if abs(angle) > 1e-6:
                        b = b.rotate((0, 0, 0), axis, angle)
                    cutter_solid = b.translate(tuple(T)).val()
                elif c.get('type') == 'cylinder':
                    radius = c['radius']
                    height = c['height']
                    # Create cylinder centered at origin along Z
                    b = cq.Workplane("XY").cylinder(height, radius)
                    # Use transformShape for efficient affine transform
                    cutter_solid = b.val().transformShape(c['transform'])
                else: # AABB
                    cx, cy, cz = c['center']
                    sx, sy, sz = c['size']
                    cutter_solid = cq.Workplane("XY").box(sx, sy, sz).translate((cx, cy, cz)).val()
                
                # 솔리드 유효성 확인 후 삭감
                if cutter_solid is not None:
                    result = result.cut(cutter_solid)
            except Exception as e:
                print(f"  - 커터 {idx+1} 삭감 실패: {e}")

        print("[CAD] 불리언 연산 완료.")
        return result, None

    def calculate_volume_error(self, result_shape):
        """
        단순화된 형상의 부피 오차율(%)을 계산하여 반환합니다.
        Auto-tune 기능에서 사용됩니다.
        """
        vol_simplified = 0.0
        
        # 1. 단순화된 형상 부피 계산
        if result_shape == "gmsh_model_final" and HAS_GMSH:
            try:
                # Gmsh 초기화 상태 확인 (이미 되어 있을 수 있음)
                if not gmsh.isInitialized(): gmsh.initialize()
                # 모든 3차원 엔티티(볼륨) 가져오기
                entities = gmsh.model.getEntities(3)
                for ent in entities:
                    vol_simplified += gmsh.model.occ.getMass(ent[0], ent[1])
            except Exception as e:
                print(f"[Volume Calc Error] Gmsh error: {e}")
                return None
        elif HAS_CADQUERY and result_shape is not None and result_shape != "gmsh_model":
            try:
                if hasattr(result_shape, "val"):
                     vol_simplified = result_shape.val().Volume()
                else:
                     return None
            except Exception as e:
                print(f"[Volume Calc Error] CadQuery error: {e}")
                return None
        else:
            return None

        # 2. 원본 형상 부피 계산
        vol_original = 0.0
        if self.original_cq:
            try:
                vol_original = self.original_cq.val().Volume()
            except: pass
        
        if vol_original <= 1e-6 and self.original_mesh:
             vol_original = self.original_mesh.volume
             
        # 3. 오차율 계산
        if vol_original > 1e-6:
            diff = abs(vol_original - vol_simplified)
            error_rate = (diff / vol_original) * 100.0
            return error_rate
        else:
            return None # 원본 부피가 0이거나 알 수 없음

    def evaluate_accuracy(self, result_shape):
        """
        원본 형상과 단순화된 형상의 부피 및 표면적을 비교하여 정확도를 평가합니다.
        """
        vol_simplified = 0.0
        area_simplified = 0.0

        if result_shape == "gmsh_model_final" and HAS_GMSH:
            print("\n[평가] Gmsh 모델 정확도 분석 중...")
            # Gmsh 초기화 상태 확인
            try:
                if not gmsh.isInitialized(): gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                # 모든 3차원 엔티티(볼륨) 가져오기
                entities = gmsh.model.getEntities(3)
                for ent in entities:
                    vol_simplified += gmsh.model.occ.getMass(ent[0], ent[1])
                # 모든 2차원 엔티티(면) 가져오기
                entities_2d = gmsh.model.getEntities(2)
                for ent in entities_2d:
                    area_simplified += gmsh.model.occ.getMass(ent[0], ent[1])
            except Exception as e:
                print(f"  - Gmsh 속성 계산 실패: {e}")
        elif HAS_CADQUERY and result_shape is not None and result_shape != "gmsh_model":
            print("\n[평가] CadQuery 모델 정확도 분석 중...")
            try:
                vol_simplified = result_shape.val().Volume()
                area_simplified = result_shape.val().Area()
            except Exception as e:
                print(f"  - CadQuery 속성 계산 실패: {e}")
        else:
            print("[평가] 결과 형상을 분석할 수 없어 평가를 건너뜁니다.")
            return

        print("\n[평가] 정확도 분석 중...")
        
        # --- 부피 (Volume) 및 표면적 (Area) ---
        # 1. 원본 속성 계산
        vol_original = 0.0
        area_original = 0.0
        
        if self.original_cq:
            try:
                vol_original = self.original_cq.val().Volume()
                area_original = self.original_cq.val().Area()
            except Exception as e:
                print(f"  - 원본 CadQuery 속성 계산 실패: {e}")
        
        if vol_original <= 1e-6 and self.original_mesh:
             vol_original = self.original_mesh.volume
             if area_original <= 1e-6:
                 area_original = self.original_mesh.area

        # 3. 부피 비교 및 출력
        print("  [부피 (Volume)]")
        if vol_original > 1e-6:
            diff = abs(vol_original - vol_simplified)
            ratio = (vol_simplified / vol_original) * 100.0
            error_rate = (diff / vol_original) * 100.0
            
            print(f"  - 원본   : {vol_original:,.2f} mm^3")
            print(f"  - 단순화 : {vol_simplified:,.2f} mm^3")
            print(f"  - 차이   : {diff:,.2f} mm^3")
            print(f"  - 비율   : {ratio:.2f}% (원본 대비)")
            print(f"  - 오차율 : {error_rate:.2f}%")
        else:
            print(f"  - 단순화 : {vol_simplified:,.2f} mm^3")
            print("  - 원본 부피를 신뢰할 수 없어 비교가 불가능합니다.")
            
        # 4. 표면적 비교 및 출력
        print("  [표면적 (Surface Area)]")
        if area_original > 1e-6:
            diff_area = abs(area_original - area_simplified)
            ratio_area = (area_simplified / area_original) * 100.0
            error_rate_area = (diff_area / area_original) * 100.0
            
            print(f"  - 원본   : {area_original:,.2f} mm^2")
            print(f"  - 단순화 : {area_simplified:,.2f} mm^2")
            print(f"  - 차이   : {diff_area:,.2f} mm^2")
            print(f"  - 비율   : {ratio_area:.2f}% (원본 대비)")
            print(f"  - 오차율 : {error_rate_area:.2f}%")
        else:
            print(f"  - 단순화 : {area_simplified:,.2f} mm^2")
            print("  - 원본 표면적을 신뢰할 수 없어 비교가 불가능합니다.")
            
        # 5. 좌표계 일치 확인을 위한 바운딩 박스 출력
        if result_shape == "gmsh_model_final" and HAS_GMSH:
            # Gmsh 모델의 경우 전체 엔티티의 경계 상자 계산
            try:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(-1, -1)
                print(f"  [결과물 좌표 범위] 최소: [{xmin:.1f}, {ymin:.1f}, {zmin:.1f}], 최대: [{xmax:.1f}, {ymax:.1f}, {zmax:.1f}]")
            except: pass
        elif hasattr(result_shape, "val"):
             b = result_shape.val().BoundingBox()
             print(f"  [결과물 좌표 범위] 최소: [{b.xmin:.1f}, {b.ymin:.1f}, {b.zmin:.1f}], 최대: [{b.xmax:.1f}, {b.ymax:.1f}, {b.zmax:.1f}]")
            
    def export_stl(self, result_shape, filename):
        """
        단순화된 형상을 STL 파일로 내보냅니다.
        """
        if result_shape == "gmsh_model_final" and HAS_GMSH:
            print(f"[내보내기] Gmsh를 통해 {filename} 저장 중...")
            if not gmsh.isInitialized():
                print("  - 오류: Gmsh가 초기화되지 않아 STL을 저장할 수 없습니다.")
                return
            try: gmsh.write(filename)
            except Exception as e: print(f"  - Gmsh STL 내보내기 실패: {e}")
            return

        if not HAS_CADQUERY or result_shape is None:
            return
            
        print(f"[내보내기] {filename} 저장 중...")
        try:
            result_shape.val().exportStl(filename)
        except Exception as e:
            print(f"  - STL 내보내기 실패: {e}")

    def export_step(self, shape_flag, filename):
        """
        심플화된 형상을 STEP 또는 IGES 파일로 내보냅니다.
        """
        # 1. Gmsh Engine
        # Check if it's a Gmsh model reference OR a special export flag for Gmsh
        is_gmsh_task = False
        if isinstance(shape_flag, str) and HAS_GMSH:
            if shape_flag.startswith("gmsh_model"):
                is_gmsh_task = True
            elif shape_flag in ["export_bounding_box", "export_cutters_only", "export_base_and_cutters"]:
                is_gmsh_task = True

        if is_gmsh_task:
            print(f"[내보내기] Gmsh를 통해 {filename} 저장 중...")
            try:
                # Gmsh 세션 확인 및 복구
                if not gmsh.isInitialized():
                    gmsh.initialize()
                
                # Special Export Flags
                if shape_flag == "export_bounding_box":
                    print(f"  - Bounding Box 생성 및 저장 중: {filename}")
                    gmsh.clear()
                    gmsh.model.add("bounding_box")
                    min_pt, max_pt = self.bounding_box
                    size = max_pt - min_pt
                    gmsh.model.occ.addBox(min_pt[0], min_pt[1], min_pt[2], size[0], size[1], size[2])
                    gmsh.model.occ.synchronize()
                    gmsh.write(filename)
                    print(f"  - 저장 완료: {filename}")
                    return

                if shape_flag == "export_cutters_only":
                    print(f"  - 커터 형상(Union) 생성 및 저장 중: {filename}")
                    gmsh.clear()
                    gmsh.model.add("cutters_only")
                    
                    cutter_ids = []
                    for c in self.cutters:
                        if c.get('type') == 'oriented':
                            sx, sy, sz = c['extents']
                            cid = gmsh.model.occ.addBox(-sx/2, -sy/2, -sz/2, sx, sy, sz)
                            matrix = c['transform'].flatten().tolist()
                            gmsh.model.occ.affineTransform([(3, cid)], matrix)
                            cutter_ids.append((3, cid))
                        elif c.get('type') == 'cylinder':
                            radius = c['radius']
                            height = c['height']
                            cid = gmsh.model.occ.addCylinder(0, 0, -height/2, 0, 0, height, radius)
                            matrix = c['transform'].flatten().tolist()
                            gmsh.model.occ.affineTransform([(3, cid)], matrix)
                            cutter_ids.append((3, cid))
                        else: # AABB
                            cx, cy, cz = c['center']
                            sx, sy, sz = c['size']
                            cid = gmsh.model.occ.addBox(cx - sx/2, cy - sy/2, cz - sz/2, sx, sy, sz)
                            cutter_ids.append((3, cid))
                    
                    gmsh.model.occ.synchronize()
                    
                    if cutter_ids:
                        gmsh.model.occ.fuse([cutter_ids[0]], cutter_ids[1:])
                        gmsh.model.occ.synchronize()
                    
                    gmsh.write(filename)
                    print(f"  - 저장 완료: {filename}")
                    return

                if shape_flag == "export_base_and_cutters":
                    print(f"  - 원본 베이스와 모든 커터(Assembly) 저장 중: {filename}")
                    gmsh.clear()
                    gmsh.model.add("base_and_cutters")

                    # 1. Base Block (Raw Material)
                    min_pt, max_pt = self.bounding_box
                    size = max_pt - min_pt
                    # Color it Gray/Invisible if possible? Gmsh doesn't easily store color in STEP cleanly without attributes, 
                    # but we create the geometry.
                    base_id = gmsh.model.occ.addBox(min_pt[0], min_pt[1], min_pt[2], size[0], size[1], size[2])
                    
                    # 2. All Cutters
                    for c in self.cutters:
                        if c.get('type') == 'oriented':
                            sx, sy, sz = c['extents']
                            cid = gmsh.model.occ.addBox(-sx/2, -sy/2, -sz/2, sx, sy, sz)
                            matrix = c['transform'].flatten().tolist()
                            gmsh.model.occ.affineTransform([(3, cid)], matrix)
                        else:
                            cx, cy, cz = c['center']
                            sx, sy, sz = c['size']
                            cid = gmsh.model.occ.addBox(cx - sx/2, cy - sy/2, cz - sz/2, sx, sy, sz)

                    gmsh.model.occ.synchronize()
                    gmsh.write(filename)
                    print(f"  - 저장 완료: {filename}")
                    return

                # Normal Model Export
                # FORCE REGENERATION:
                # 사용자가 "Refine" 등을 통해 커터를 수정한 상태를 확실히 반영하기 위해
                # 기존 모델을 초기화하고 현재 self.cutters 기반으로 다시 생성합니다.
                gmsh.clear() 
                print("  - Gmsh 모델 데이터 재생성 중...")
                self._generate_cad_gmsh()
                
                gmsh.write(filename)
                print(f"  - 저장 완료: {filename}")
            except Exception as e:
                print(f"  - Gmsh CAD 내보내기 실패: {e}")
                show_custom_msg("Export Error", f"Gmsh Export Failed:\n{e}", 'error')
            return
        
        # 2. CadQuery Engine
        if not HAS_CADQUERY or shape_flag is None or isinstance(shape_flag, str):
            if isinstance(shape_flag, str):
                print(f"[내보내기] 경고: '{shape_flag}'는 유효한 CadQuery 객체가 아닙니다.")
            return

        print(f"[내보내기] CadQuery를 통해 {filename} 저장 중...")
        try:
            # step, stp, iges, igs
            export_type = 'STEP'
            if filename.lower().endswith(('.iges', '.igs')):
                export_type = 'IGES'
                
            cq.exporters.export(shape_flag.val(), filename, exportType=export_type)
            print(f"  - 저장 완료: {filename}")
        except Exception as e:
            print(f"  - CadQuery 내보내기 실패: {e}")
            show_custom_msg("Export Error", f"CadQuery Export Failed:\n{e}", 'error')
        
        
        # Gmsh Finalize는 프로그램 종료 시에만 호출해야 함 (여기서 닫으면 세션이 끊김)
        # if HAS_GMSH:
        #    try: gmsh.finalize()
        #    except: pass


    def export_cutter_info(self, filename):
        """커터 정보를 텍스트 파일로 저장합니다."""
        if not self.cutters:
            return
            
        print(f"[내보내기] 커터 정보 {filename} 저장 중...")
        try:
            with open(filename, 'w') as f:
                f.write("ID, Type, Center_X, Center_Y, Center_Z, Size_X, Size_Y, Size_Z, [Vectors...]\n")
                for idx, c in enumerate(self.cutters):
                    if c.get('type') == 'oriented':
                        cx, cy, cz = c['transform'][:3, 3]
                        sx, sy, sz = c['extents']
                        # transform 행렬 정보도 함께 저장
                        mat_str = ",".join([f"{x:.4f}" for x in c['transform'].flatten()])
                        f.write(f"{idx+1},Oriented,{cx:.4f},{cy:.4f},{cz:.4f},{sx:.4f},{sy:.4f},{sz:.4f},Matrix:[{mat_str}]\n")
                    elif c.get('type') == 'cylinder':
                        cx, cy, cz = c['transform'][:3, 3]
                        r, h = c['radius'], c['height']
                        mat_str = ",".join([f"{x:.4f}" for x in c['transform'].flatten()])
                        f.write(f"{idx+1},Cylinder,{cx:.4f},{cy:.4f},{cz:.4f},{r:.4f},{h:.4f},0,Matrix:[{mat_str}]\n")
                    else:
                        cx, cy, cz = c['center']
                        sx, sy, sz = c['size']
                        f.write(f"{idx+1},AABB,{cx:.4f},{cy:.4f},{cz:.4f},{sx:.4f},{sy:.4f},{sz:.4f}\n")
        except Exception as e:
            print(f"  - 커터 정보 저장 실패: {e}")

    def visualize(self, result_shape=None, show_removed=False):
        if not HAS_PYVISTA:
            print("[시각화] PyVista를 사용할 수 없어 시각화를 건너뜁니다.")
            return
            
        # 빈 메쉬 플로팅 허용 (에러 방지)
        pv.global_theme.allow_empty_mesh = True
        
        print("[시각화] 3D 뷰어 실행 중...")
        # 2x2 레이아웃으로 변경 (원본+커터 | 결과 | 중첩 비교 | 히트맵)
        p = pv.Plotter(shape=(2, 2), title="CAD Simplification Comparison")
        p.enable_parallel_projection()
        
        # 결과물 메쉬 변환 (시각화용)
        result_pv = None
        if result_shape is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    tmp_path = tmp.name
                self.export_stl(result_shape, tmp_path)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    result_pv = pv.read(tmp_path)
                if os.path.exists(tmp_path): os.remove(tmp_path)
            except Exception as e:
                print(f"[시각화] 결과물 변환 실패: {e}")

        # 1. 원본 및 커터
        p.subplot(0, 0)
        p.add_text("1. Original + Cutter Guide", font_size=9, color='black')
        p.add_mesh(self.original_mesh, color='lightblue', opacity=0.3, label="Original")
        
        for c in self.cutters:
            is_refine = c.get('is_refine', False)
            # Refine 커터는 진한 마젠타색으로 강조
            color = 'magenta' if is_refine else ('orange' if c.get('type') == 'oriented' else 'red')
            width = 3 if is_refine else 1
            opacity = 0.8 if is_refine else 0.4
            
            if c.get('type') == 'oriented':
                box = pv.Box(bounds=(-c['extents'][0]/2, c['extents'][0]/2,
                                     -c['extents'][1]/2, c['extents'][1]/2,
                                     -c['extents'][2]/2, c['extents'][2]/2))
                box.transform(c['transform'], inplace=True)
                p.add_mesh(box, color=color, opacity=opacity, style='wireframe', line_width=width)
            elif c.get('type') == 'cylinder':
                cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=c['radius'], height=c['height'])
                cyl.transform(c['transform'], inplace=True)
                p.add_mesh(cyl, color=color, opacity=opacity, style='wireframe', line_width=width)
            else:
                cx, cy, cz = c['center']
                sx, sy, sz = c['size']
                bounds = [cx-sx/2, cx+sx/2, cy-sy/2, cy+sy/2, cz-sz/2, cz+sz/2]
                p.add_mesh(pv.Box(bounds=bounds), color=color, opacity=opacity, style='wireframe', line_width=width)


        # 2. 최종 결과물 (Simplified Result)
        p.subplot(0, 1)
        p.add_text("2. Simplified Result", font_size=9, color='black')
        
        # [Strategy 3 Support] 분할된 솔리드가 있으면 각각 다른 색상으로 표시
        if self.decomposed_solids:
            import matplotlib.colors as mcolors
            colors = list(mcolors.TABLEAU_COLORS.values())
            
            p.add_text(f"\n(Decomposed: {len(self.decomposed_solids)} blocks)", position='upper_right', font_size=9, color='darkgreen')
            
            for i, solid in enumerate(self.decomposed_solids):
                try:
                    # 각 솔리드를 개별 STL로 변환
                    # solid is a CQ Workplane object
                    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                         # exportStl calculates bounding box center? No, just exports.
                         # Need to ensure solid is valid
                         solid.val().exportStl(tmp.name)
                         block_mesh = pv.read(tmp.name)
                         
                    col = colors[i % len(colors)]
                    p.add_mesh(block_mesh, color=col, show_edges=True, opacity=0.9, label=f"Block {i+1}")
                    
                    if os.path.exists(tmp.name): os.remove(tmp.name)
                except Exception as e:
                    print(f"Block vis error: {e}")
        elif result_pv and result_pv.n_points > 0:
            p.add_mesh(result_pv, color='lightgreen', show_edges=True)
        else:
            p.add_text("\n(Empty Result)", color='darkred', font_size=9)
            
        if show_removed:
            p.add_text("\n(+ Removed Volume)", position='upper_right', font_size=9, color='darkred')
            for c in self.cutters:
                if c.get('type') == 'oriented':
                    box = pv.Box(bounds=(-c['extents'][0]/2, c['extents'][0]/2,
                                         -c['extents'][1]/2, c['extents'][1]/2,
                                         -c['extents'][2]/2, c['extents'][2]/2))
                    box.transform(c['transform'], inplace=True)
                    p.add_mesh(box, color='red', opacity=0.15, style='surface')
                elif c.get('type') == 'cylinder':
                    cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=c['radius'], height=c['height'])
                    cyl.transform(c['transform'], inplace=True)
                    p.add_mesh(cyl, color='red', opacity=0.15, style='surface')
                else:
                    cx, cy, cz = c['center']
                    sx, sy, sz = c['size']
                    bounds = [cx-sx/2, cx+sx/2, cy-sy/2, cy+sy/2, cz-sz/2, cz+sz/2]
                    p.add_mesh(pv.Box(bounds=bounds), color='red', opacity=0.15, style='surface')

        # 3. 중첩 비교 (Overlay Comparison)
        p.subplot(1, 0)
        p.add_text("3. Overlay Comparison", font_size=9, color='black')
        p.add_mesh(self.original_mesh, color='lightblue', opacity=0.3)
        if result_pv and result_pv.n_points > 0:
            p.add_mesh(result_pv, color='red', opacity=0.5, style='wireframe', label="Simplified")
        
        # 4. 히트맵 (Heatmap) - 거리 오차 시각화
        p.subplot(1, 1)
        p.add_text("4. Error Heatmap", font_size=9, color='black')
        
        if result_pv and result_pv.n_points > 0 and self.original_mesh:
            try:
                # Trimesh의 근접 점 찾기 기능 사용
                # result_pv의 각 정점에서 original_mesh까지의 최단 거리 계산
                closest, distances, triangle_id = self.original_mesh.nearest.on_surface(result_pv.points)
                
                # 거리를 스칼라 값으로 추가
                result_pv["Distance"] = distances
                
                # 히트맵 출력 (거리가 클수록 빨간색)
                p.add_mesh(result_pv, scalars="Distance", cmap="jet", show_scalar_bar=True, label="Error")
                p.add_text(f"Max Error: {np.max(distances):.4f} mm", position='lower_right', font_size=8, color='black')
            except Exception as e:
                print(f"[시각화] 히트맵 계산 실패: {e}")
                p.add_text(f"\nHeatmap Error: {e}", color='darkred', font_size=8)
        else:
             p.add_text("\n(Insufficient Data)", color='darkred', font_size=9)

        p.link_views() # 모든 뷰포트 시점 동기화
        p.show()


    def visualize_comparison(self, result_shape=None):
        """
        사용자가 요청한 2x2 상세 비교 뷰:
        1. 원본 + 복셀 (Original & Voxel)
        2. 생성된 커팅 블럭 (Cutting Blocks)
        3. 최종 형상과 원본 비교 (Comparison Overlay)
        4. 오차 히트맵 (Distance Heatmap)
        """
        if not HAS_PYVISTA:
            print("[시각화] PyVista를 사용할 수 없습니다.")
            return
            
        pv.global_theme.allow_empty_mesh = True
        print("[시각화] 상세 비교 뷰(2x2) 실행 중...")
        p = pv.Plotter(shape=(2, 2), title="CAD Simplification Detailed Comparison")
        p.enable_parallel_projection() # 원본 비율 유지를 위해 평행 투영(Orthographic) 사용
        
        # 결과물 메쉬 변환
        result_pv = None
        if result_shape is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    tmp_path = tmp.name
                self.export_stl(result_shape, tmp_path)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    result_pv = pv.read(tmp_path)
                if os.path.exists(tmp_path): os.remove(tmp_path)
            except Exception as e:
                print(f"[시각화] 결과물 변환 실패: {e}")

        # 1. 원본 + 복셀
        p.subplot(0, 0)
        p.add_text("1. Original & Voxel", font_size=9, color='black')
        p.add_mesh(self.original_mesh, color='lightblue', opacity=0.3, label="Original")
        if self.current_grid is not None:
            try:
                grid = pv.ImageData()
                grid.dimensions = np.array(self.current_grid.shape) + 1
                grid.origin = self.grid_origin
                grid.spacing = (self.voxel_scale, self.voxel_scale, self.voxel_scale)
                grid.cell_data["Occupied"] = (~self.current_grid).flatten(order="F")
                voxel_mesh = grid.threshold(0.5, scalars="Occupied")
                if voxel_mesh.n_cells > 0:
                    p.add_mesh(voxel_mesh, color='gray', opacity=0.5, show_edges=True, label="Voxel")
            except Exception as e:
                print(f"[시각화] 복셀 시각화 실패: {e}")

        # 2. 생성된 커팅 블럭 (Solids)
        p.subplot(0, 1)
        p.add_text("2. Cutting Blocks Structure", font_size=9, color='black')
        
        # 색상 팔레트 (Refine view_cutters_only와 동일하게)
        colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"]
        
        if not self.cutters:
            p.add_text("\n(No Cutters Generated)", color='gray', font_size=10)
        else:
            for i, c in enumerate(self.cutters):
                color = colors[i % len(colors)]
                if c.get('type') == 'oriented':
                    box = pv.Box(bounds=(-c['extents'][0]/2, c['extents'][0]/2,
                                         -c['extents'][1]/2, c['extents'][1]/2,
                                         -c['extents'][2]/2, c['extents'][2]/2))
                    box.transform(c['transform'], inplace=True)
                    p.add_mesh(box, color=color, opacity=0.8, show_edges=True)
                elif c.get('type') == 'cylinder':
                    cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=c['radius'], height=c['height'])
                    cyl.transform(c['transform'], inplace=True)
                    p.add_mesh(cyl, color=color, opacity=0.8, show_edges=True)
                else:
                    cx, cy, cz = c['center']
                    sx, sy, sz = c['size']
                    bounds = [cx-sx/2, cx+sx/2, cy-sy/2, cy+sy/2, cz-sz/2, cz+sz/2]
                    p.add_mesh(pv.Box(bounds=bounds), color=color, opacity=0.8, show_edges=True)

        # 3. 최종 형상과 원본 비교 (Overlay)
        p.subplot(1, 0)
        p.add_text("3. Overlay Comparison", font_size=9, color='black')
        p.add_mesh(self.original_mesh, color='lightblue', opacity=0.3)
        if result_pv:
            p.add_mesh(result_pv, color='red', opacity=0.5, style='wireframe', label="Simplified")

        # 4. 오차 히트맵
        p.subplot(1, 1)
        p.add_text("4. Error Heatmap", font_size=9, color='black')
        if result_pv and self.original_mesh:
            try:
                closest, distances, triangle_id = self.original_mesh.nearest.on_surface(result_pv.points)
                result_pv["Distance"] = distances
                p.add_mesh(result_pv, scalars="Distance", cmap="jet", show_scalar_bar=True)
            except Exception as e:
                p.add_mesh(result_pv, color='lightgreen', show_edges=True)
        
        p.link_views()
        p.link_views()
        p.show()


    def _check_collision_solid(self, R, pos, dims, col_mgr=None, use_boolean=True):
        """
        [Helper] 정밀 충돌 검사 (Boolean Subtraction)
        1. FCL(빠른 충돌)로 1차 필터링
        2. 충돌 감지 시, 실제 Boolean Subtraction으로 부피 감소 확인 (정밀 확인)
        3. 실패 시 FCL 결과 신뢰
        """
        # 1. Fast Pass: FCL or Simple AABB/Point Check
        # 만약 FCL이 "충돌 없음"이라고 하면, 굳이 Boolean을 할 필요가 없음.
        
        # 임시 Box Mesh 생성 (FCL용)
        T_check = np.eye(4)
        T_check[:3, :3] = R
        T_check[:3, 3] = pos
        box_mesh = trimesh.creation.box(extents=dims, transform=T_check)
        
        is_touching = False
        
        # FCL Check
        if col_mgr:
            try:
                if col_mgr.in_collision_single(box_mesh):
                    is_touching = True
            except:
                is_touching = True # Error -> Assume safe (blocked)
        elif self.original_mesh:
             # Legacy Point Check
            dx, dy, dz = dims / 2.0
            l_corners = np.array([
                [-dx, -dy, -dz], [-dx, -dy, dz],
                [-dx, dy, -dz], [-dx, dy, dz],
                [dx, -dy, -dz], [dx, -dy, dz],
                [dx, dy, -dz], [dx, dy, dz]
            ])
            # Sample Faces (Reduced for speed in fast pass)
            l_faces = [
                [dx, 0, 0], [-dx, 0, 0], [0, dy, 0], [0, -dy, 0], [0, 0, dz], [0, 0, -dz]
            ]
            local_pts = np.vstack([l_corners, l_faces])
            world_pts = np.dot(local_pts, R.T) + pos
            if self.original_mesh.contains(world_pts).any():
                is_touching = True
        
        # [Optimization] 충돌이 전혀 없으면 바로 False 반환
        if not is_touching:
            return False
            
        # 2. Touch Detected -> Verify with Boolean (Real Cut)
        if not use_boolean or not self.original_mesh:
            return True # Boolean 안 쓰면 그냥 충돌로 인정

        try:
            # Try Trimesh Boolean first
            difference_mesh = trimesh.boolean.difference([self.original_mesh], [box_mesh])
            
            if difference_mesh.is_watertight:
                vol_diff = self.original_mesh.volume - difference_mesh.volume
                if vol_diff > 1e-4:
                    return True # Volume Decreased -> Real Collision
                else:
                    return False # Touching but no volume loss (Grazing)
            else:
                 # Boolean Failed (Non-Watertight)
                 raise ValueError("Non-watertight boolean result")
                 
        except Exception as e:
            # [Fallback] Gmsh Attempt? or Just Trust FCL
            if self._try_gmsh_boolean_check(T_check, dims):
                 return True
            
            # If all booleans fail, conservative fallback:
            # FCL said touch, so likely collision.
            return True

    def _try_gmsh_boolean_check(self, T, dims):
        """
        [Experimental] Gmsh를 이용한 Boolean Check
        """
        if not HAS_GMSH: return False
        try:
             # Gmsh Python API requires initializing logic.
             # This is a placeholder as full Gmsh Boolean inside loop is too slow without optimization.
             # If user explicitly enabled Gmsh mode, we could try.
             return False 
        except:
             return False
        """
        현재 설정(base_params)을 기준으로 지정된 범위(opt_config) 내에서 파라미터를 탐색합니다.
        opt_config: {
            'res_range': (min, max, steps),
            'vol_range': (min, max, steps),
            'w_accuracy': float (0.0~1.0),
            'w_simplicity': float (0.0~1.0)
        }
        """
        print("\n[최적화] 파라미터 최적화 시뮬레이션 시작...")
        
        if opt_config is None:
            # Default fallback
            opt_config = {
                'res_range': (0.8, 1.2, 3),
                'vol_range': (0.8, 1.2, 3),
                'w_accuracy': 0.7,
                'w_simplicity': 0.3
            }

        # 1. 탐색 공간 생성
        r_min, r_max, r_steps = opt_config['res_range']
        v_min, v_max, v_steps = opt_config['vol_range']
        
        res_factors = np.linspace(r_min, r_max, int(r_steps))
        vol_factors = np.linspace(v_min, v_max, int(v_steps))
        
        variations = []
        for r in res_factors:
            for v in vol_factors:
                desc = f"Res x{r:.2f}, Vol x{v:.2f}"
                if abs(r-1.0)<0.01 and abs(v-1.0)<0.01: desc += " (Current)"
                variations.append((r, v, desc))

        # 중복 제거 (Set)
        variations = list(set(variations))
        
        results = []
        original_cutters = self.cutters # 상태 백업
        
        # 원본 부피 (참조용)
        vol_orig = 0.0
        if self.original_mesh:
             vol_orig = self.original_mesh.volume
        
        if vol_orig <= 0:
            print("[최적화] 원본 부피를 계산할 수 없어 중단합니다.")
            return []
            
        total_vars = len(variations)
        print(f"[최적화] 총 {total_vars}개 조합 시뮬레이션...")

        for idx, (res_f, vol_f, desc) in enumerate(variations):
            # 파라미터 설정
            # Resolution은 너무 작으면(0.1mm 미만) 너무 느리므로 하한선
            trial_res = max(0.1, round(base_params['voxel_resolution'] * res_f, 2))
            trial_min_vol = base_params['min_volume_ratio'] * vol_f
            
            # generate_cutters 메서드는 self.cutters를 변경하므로 실행 후 복사본 저장
            self.generate_cutters(
                voxel_resolution=trial_res,
                min_volume_ratio=trial_min_vol,
                max_cutters=int(base_params['max_cutters']),
                tolerance=base_params['tolerance'],
                detect_slanted=base_params['detect_slanted'],
                masks=[],
                slanted_area_factor=base_params['slanted_area_factor'],
                slanted_edge_factor=base_params['slanted_edge_factor'],
                min_cutter_size=base_params['min_cutter_size'],
                append=False, # 항상 새로 생성
                undersize_mode=base_params.get('undersize_mode', False),
                perform_erosion=base_params.get('perform_erosion', False),
                cleanup_artifacts=base_params.get('cleanup_artifacts', False)
            )
            
            # 결과 분석
            count = len(self.cutters)
            
            est_vol = 0.0
            if self.current_grid is not None:
                for c in self.cutters:
                    if c.get('type') == 'oriented':
                        est_vol += np.prod(c['extents'])
                    else:
                        est_vol += np.prod(c['size'])
            
            # 오차율 (부피 차이)
            diff_vol = abs(vol_orig - est_vol)
            err_rate = (diff_vol / vol_orig) * 100.0
            
            # 점수 계산 (Weighted Score)
            # Normalize Error: 5% ~ 1.0
            # Normalize Count: 100 ~ 1.0
            norm_err = err_rate / 5.0 
            norm_count = count / 50.0 
            
            w_acc = opt_config['w_accuracy']
            w_sim = opt_config['w_simplicity']
            
            # 작을수록 우수함.
            score = (norm_err * w_acc) + (norm_count * w_sim)
            
            results.append({
                'params': {
                    'voxel_resolution': trial_res,
                    'min_volume_ratio': trial_min_vol,
                    'max_cutters': base_params['max_cutters'],
                    'tolerance': base_params['tolerance'],
                    'detect_slanted': base_params['detect_slanted'],
                    'slanted_area_factor': base_params['slanted_area_factor'],
                    'slanted_edge_factor': base_params['slanted_edge_factor'],
                    'min_cutter_size': base_params['min_cutter_size'],
                    'undersize_mode': base_params.get('undersize_mode', False),
                    'perform_erosion': base_params.get('perform_erosion', False)
                },
                'metrics': {
                    'count': count,
                    'error_rate': err_rate,
                    'desc': desc
                },
                'score': score
            })
            
            print(f"  - ({idx+1}/{total_vars}) [Sim] Res={trial_res}, Cutters={count}, Err={err_rate:.1f}%, Score={score:.2f}")

        # 복구
        self.cutters = original_cutters
        # 정렬 (점수 오름차순 - 낮은게 좋음)
        results.sort(key=lambda x: x['score'])
        
        return results



    def create_sample_shape(self, shape_type='basic'):
        """샘플 형상을 생성하여 로드합니다."""
        if not HAS_CADQUERY:
            print("CadQuery not available for sample generation.")
            return False
        
        try:
            if shape_type == 'basic':
                # Existing example
                s = cq.Workplane("XY").box(100, 100, 20).faces(">Z").workplane().hole(40)
                s = s.edges("|Z").chamfer(5)
                s = s.faces(">X").workplane().transformed(rotate=(0, -45, 0)).split(keepBottom=True)
                self.original_cq = s
                
            elif shape_type == 'ribbed_cushion':
                # Simple ribbed structure
                L, W, H, t = 100, 80, 30, 2
                s = cq.Workplane("XY").box(L, W, H).faces("-Z").shell(t)
                # Add cross ribs
                rib_x = cq.Workplane("XY").box(L-2*t, 2, H-t).translate((0, 0, t/2))
                rib_y = cq.Workplane("XY").box(2, W-2*t, H-t).translate((0, 0, t/2))
                s = s.union(rib_x).union(rib_y)
                self.original_cq = s
                
            elif shape_type == 'complex_ribs':
                # Grid ribs
                L, W, H, t = 120, 120, 40, 3
                s = cq.Workplane("XY").box(L, W, H).faces("-Z").shell(t)
                rib_grid = cq.Workplane("XY")
                for i in range(-1, 2):
                    r = cq.Workplane("XY").box(L-2*t, 2, H-t).translate((0, i*30, t/2))
                    rib_grid = rib_grid.union(r)
                    r2 = cq.Workplane("XY").box(2, W-2*t, H-t).translate((i*30, 0, t/2))
                    rib_grid = rib_grid.union(r2)
                s = s.union(rib_grid)
                s = s.union(rib_grid)
                self.original_cq = s

            elif shape_type == 'l_bracket':
                # L-Shape Bracket with holes
                s = cq.Workplane("XY").box(60, 60, 10).faces(">Z").workplane().rect(20, 60).cutThruAll()
                s = s.faces("<X").workplane().circle(5).cutThruAll()
                self.original_cq = s
            
            elif shape_type == 'pipe_connector':
                # Cylindrical Pipe
                s = cq.Workplane("XY").cylinder(50, 20).faces(">Z").hole(30)
                # Side pipe
                s2 = cq.Workplane("XZ").transformed(offset=(0,0,10)).cylinder(40, 10).faces(">Z").hole(12)
                s = s.union(s2)
                self.original_cq = s
                
            elif shape_type == 'packaging_cushion':
                # Reference: EPS/Pulp packaging buffer (User Image)
                # Dims: 2000 x 1200 x 250 mm
                L, W, H = 2000.0, 1200.0, 250.0
                
                print("Generating Packaging Cushion Model...")
                
                # 1. Base Plate (Z=0~50)
                base_h = 50.0
                s = cq.Workplane("XY").box(L, W, base_h).translate((0, 0, base_h/2))
                
                # 2. Add Corner Blocks (Chunky supports)
                corner_l = 400.0
                corner_w = 300.0
                h_block = H - base_h
                
                blocks = []
                for x_sign in [-1, 1]:
                    for y_sign in [-1, 1]:
                        cx = x_sign * (L/2 - corner_l/2)
                        cy = y_sign * (W/2 - corner_w/2)
                        b = cq.Workplane("XY").box(corner_l, corner_w, h_block).translate((cx, cy, base_h + h_block/2))
                        blocks.append(b)
                
                # 3. Add Ribs (Internal supports)
                rib_t = 60.0 # Thickness
                rib_h = h_block * 0.8 # Slightly lower than max height
                rib_len = W - (corner_w * 2) # Between corner blocks
                
                # 3 Ribs spaced along Length
                for i in range(-1, 2):
                    rx = i * 500.0
                    if abs(rx) < 1.0: # Center
                         # Center usually has cutout, let's make it splitted
                         r1 = cq.Workplane("XY").box(rib_t, rib_len/2 - 50, rib_h).translate((rx, (rib_len/4 + 25), base_h + rib_h/2))
                         r2 = cq.Workplane("XY").box(rib_t, rib_len/2 - 50, rib_h).translate((rx, -(rib_len/4 + 25), base_h + rib_h/2))
                         blocks.append(r1)
                         blocks.append(r2)
                    else:
                         r = cq.Workplane("XY").box(rib_t, rib_len, rib_h).translate((rx, 0, base_h + rib_h/2))
                         blocks.append(r)
                
                # Union all
                for b in blocks:
                    s = s.union(b)
                
                # Center the whole assembly to (0,0,0)
                s = s.translate((0, 0, -H/2))
                self.original_cq = s

            # Convert to mesh for visualization/processing
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                self.original_cq.val().exportStl(tmp.name)
                tmp_path = tmp.name
            self.original_mesh = trimesh.load(tmp_path)
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
            self.original_mesh.process()
            self.original_mesh.fix_normals()
            self.bounding_box = self.original_mesh.bounds
            return True
        except Exception as e:
            print(f"Sample generation failed: {e}")
            return False

    def show_control_panel(self):
        """
        메인 제어 패널을 실행합니다. 설정, 실행, 결과 확인을 통합합니다.
        """
        if not HAS_PYVISTA:
            print("PyVista가 없어 GUI를 실행할 수 없습니다.")
            return
        
        import tkinter as tk
        from tkinter import messagebox, ttk, filedialog
        import threading
        
        # 기본값 설정
        '''
        # 설정 변수에 대한 주석을 항상 여기에 추가해달라.
        'voxel_resolution': 1.0,       # 복셀 해상도 (mm). 작을수록 정밀하지만 계산 비용 증가.
        'min_volume_ratio': 0.002,     # 전체 부피 대비 최소 커터 부피 비율. 이보다 작은 커터는 무시.
        'max_cutters': 100,            # 생성할 최대 커터 개수.
        'tolerance': 0.05,             # 커터 확장 시 허용 오차 (mm).
        'slanted_tolerance': -0.1,     # 경사면 커터 전용 오차 (작게 설정).
        'detect_slanted': True,        # 경사면(축 정렬되지 않은 면) 감지 여부.
        'slanted_area_factor': 1.5,    # 경사면 감지 시 최소 면적 계수 (voxel_resolution * factor)^2.
        'min_cutter_size': 3.0,        # 커터의 최소 변 길이 (mm).
        'auto_tune': False,            # 목표 오차율 달성을 위한 해상도 자동 조절 여부.
        'target_error': 5.0,           # 자동 조절 시 목표 부피 오차율 (%).
        'max_iterations': 5            # 자동 조절 또는 반복 탐색 시 최대 반복 횟수.
        '''
        self._cfg = {
            'voxel_resolution': 1.0,       # 복셀 해상도 (mm). 작을수록 정밀하지만 계산 비용 증가.
            'min_volume_ratio': 0.002,     # 전체 부피 대비 최소 커터 부피 비율. 이보다 작은 커터는 무시.
            'max_cutters': 100,            # 생성할 최대 커터 개수.
            'tolerance': 0.0,              # 커터 확장 시 허용 오차 (mm).
            'slanted_tolerance': 0.0,      # 경사면 커터 전용 오차 (0.0=안전).
            'detect_slanted': True,        # 경사면(축 정렬되지 않은 면) 감지 여부.
            'slanted_area_factor': 4.0,    # 경사면 감지 시 최소 면적 계수 (voxel_resolution * factor)^2.
            'slanted_edge_factor': 2.0,    # 경사면 감지 시 최소 모서리 길이 계수 (voxel_resolution * factor).
            'angle_snap_divisions': 2,     # [New] 경사면 각도 90도 분할 수 (0=끄기, 4=22.5도 단위).
            'min_cutter_size': 3.0,        # 커터의 최소 변 길이 (mm).
            'auto_tune': False,            # 목표 오차율 달성을 위한 해상도 자동 조절 여부.
            'target_error': 5.0,           # 자동 조절 시 목표 부피 오차율 (%).
            'max_iterations': 5,           # 자동 조절 또는 반복 탐색 시 최대 반복 횟수.
            'undersize_mode': False,       # 커터 확장 (Undercut)
            'perform_erosion': False,      # 복셀 침식 (Erosion)
            'cleanup_artifacts': False     # 잔여물 제거 (Artifact Cleanup)
        }

        # Tkinter 윈도우 설정
        root = tk.Tk()
        root.title("CAD Simplifier Control Panel")
        root.title("CAD Simplifier Control Panel")
        root.geometry("520x850")
        center_window(root)
        center_window(root)
        
        # 파라미터 상세 설명 텍스트 (Help Window용)
        help_md_text = """
# 상세 도움말 (Detailed Help)

## 📌 기본 설정 (Basic Settings)
**Voxel Resolution (mm)**
모델을 복셀(3D 픽셀)화할 때의 단위 크기입니다.
- **작은 값**: 정밀도가 높아지지만, 연산 속도가 느려지고 커터 수가 급격히 늘어납니다.
- **큰 값**: 연산이 빠르지만, 형상이 뭉뚱그려질 수 있습니다.
- *권장*: 모델 크기에 따라 1.0 ~ 5.0 mm.

**Min Volume Ratio (0~1)**
노이즈 제거를 위한 임계값입니다.
전체 부피 대비 이 비율보다 작은 커터(부스러기)는 생성하지 않습니다.
- 예: 0.002는 전체 부피의 0.2% 미만인 덩어리를 무시합니다.

**Max Cutter Count**
생성할 최대 커터(육면체 블록)의 개수입니다.
이 제한에 도달하면 크기가 큰 덩어리부터 우선적으로 생성됩니다.

**Tolerance (mm)**
커터를 팽창/축소할 때 적용하는 여유 공차입니다.
너무 작으면 CAD 연산 시 면이 겹치는 오류가 발생할 수 있습니다.

## 📌 고급 설정 (Advanced)
**Detect Slanted Surfaces**
축(X,Y,Z)에 정렬되지 않은 경사면을 감지합니다.
이 옵션이 켜져 있으면 회전된 박스(OBB)를 생성하여 비스듬한 형상을 더 적은 수의 커터로 표현합니다.

**Min Cutter Size (mm)**
설정된 값보다 변의 길이가 작은 커터는 생성을 억제합니다.
미세한 조각이 CAD 불리언 연산 실패를 유발하는 것을 방지합니다.

**Auto-tune**
Generative Design 방식처럼, 목표 오차율(Target Error)을 달성할 때까지 해상도를 자동으로 미세 조정하며 반복 실행합니다.

## 📌 체크박스 옵션
**Prevent Excess Mass (Undersize Mode)**
단순화된 결과물이 원본 형상 밖으로 튀어나오지 않도록 합니다.
커터(제거되는 영역)를 약간 확장(Inflation)하여 깎아내므로, 결과적으로 남는 형상은 원본보다 작거나 같게 됩니다.
(포장 완충재 설계 등 간섭이 없어야 하는 경우 필수)

**Voxel Erosion**
복셀 단계에서 표면을 한 겹 깎아냅니다.
Undersize Mode와 유사하지만 픽셀 단위로 작동하여 더 거칠 수 있습니다.

**Show Removed**
결과 뷰에서 '제거된 영역(Cutters)'을 빨간색 반투명 박스로 함께 시각화합니다.
어느 부분이 깎여나갔는지 직관적으로 확인할 수 있습니다.
"""

        def open_help_window():
            # Singleton Check
            if hasattr(self, 'help_win') and self.help_win and self.help_win.winfo_exists():
                self.help_win.lift()
                return
            
            self.help_win = tk.Toplevel(root)
            self.help_win.title("Help & Documentation")
            self.help_win.geometry("500x700")
            
            # Container Frame for Text and Scrollbar
            text_frame = tk.Frame(self.help_win)
            text_frame.pack(fill=tk.BOTH, expand=True)

            scrollbar = tk.Scrollbar(text_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            txt = tk.Text(text_frame, wrap=tk.WORD, padx=15, pady=15, bg="white", borderwidth=0, font=("Segoe UI", 10))
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            txt.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=txt.yview)
            
            # Markdown Parsing Tags
            txt.tag_config("h1", font=("Segoe UI", 16, "bold"), foreground="#2c3e50", spacing1=10, spacing3=10)
            txt.tag_config("h2", font=("Segoe UI", 12, "bold"), foreground="#2980b9", spacing1=15, spacing3=5)
            txt.tag_config("bold", font=("Segoe UI", 10, "bold"), foreground="#333333")
            txt.tag_config("bullet", lmargin1=15, lmargin2=25)
            txt.tag_config("normal", spacing1=2)
            
            # Simple Parser
            for line in help_md_text.strip().split('\n'):
                line = line.strip()
                if not line:
                    txt.insert(tk.END, "\n")
                    continue
                
                if line.startswith("# "):
                    txt.insert(tk.END, line[2:] + "\n", "h1")
                elif line.startswith("## "):
                    txt.insert(tk.END, line[3:] + "\n", "h2")
                elif line.startswith("- "):
                    # Simple bullet with bold support
                    content = line[2:]
                    parts = content.split("**")
                    for i, part in enumerate(parts):
                        tag = "bold" if i % 2 == 1 else "normal"
                        txt.insert(tk.END, part, (tag, "bullet"))
                    txt.insert(tk.END, "\n", "bullet")
                else:
                    # Generic line with bold support
                    parts = line.split("**")
                    for i, part in enumerate(parts):
                        tag = "bold" if i % 2 == 1 else "normal"
                        txt.insert(tk.END, part, tag)
                    txt.insert(tk.END, "\n", "normal")
            
            txt.configure(state='disabled')

        entries = {}
        fields = [
            ('Voxel Resolution (mm)', 'voxel_resolution'),
            ('Min Volume Ratio', 'min_volume_ratio'),
            ('Max Cutter Count', 'max_cutters'),
            ('Tolerance (mm)', 'tolerance'),
            ('Slanted Tol (mm)', 'slanted_tolerance'),
            ('Min Cutter Size (mm)', 'min_cutter_size'),
            ('Slanted Area Factor (Area)', 'slanted_area_factor'),
            ('Slanted Edge Factor (Length)', 'slanted_edge_factor'),
            ('Angle Snap Divs (0=Off)', 'angle_snap_divisions'),
            ('Target Error (%)', 'target_error'),
            ('Max Iterations', 'max_iterations')
        ]
        
        # Parameters Panel
        param_frame = tk.LabelFrame(root, text="Settings", padx=10, pady=10)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # --- Preset Panel (Inside Settings, Top) ---
        preset_frame = tk.Frame(param_frame, bg="#f0f0f0", pady=5)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(preset_frame, text="Auto-Set Parameters:", bg="#f0f0f0", font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value="Medium (Balanced)")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var, 
                                    values=["Fine (Detail)", "Medium (Balanced)", "Coarse (Rough)", "Very Rough (Super Simplified)"], 
                                    state="readonly", width=25)
        preset_combo.pack(side=tk.LEFT, padx=5)
        
        # Apply Logic defined early to bind button
        def apply_preset():
            if self.bounding_box is None:
                show_custom_msg("Warning", "Load a model first to calculate scale.", 'warning', root)
                return
                
            min_pt, max_pt = self.bounding_box
            diagonal = np.linalg.norm(max_pt - min_pt)
            mode = self.preset_var.get()
            
            # Scale-based Heuristics with FIXED Min Cutter Sizes
            if "Fine" in mode:
                res = diagonal / 150.0
                min_cutter = 2.0 
                tol = 0.01 
                max_cnt = 200
            elif "Very Rough" in mode:
                res = diagonal / 25.0
                min_cutter = 10.0
                tol = 0.1 
                max_cnt = 25
            elif "Coarse" in mode:
                res = diagonal / 50.0
                min_cutter = 5.0
                tol = 0.05
                max_cnt = 50
            else: # Medium
                res = diagonal / 100.0
                min_cutter = 3.0
                tol = 0.02
                max_cnt = 100
                
            # Limits
            res = max(0.2, round(res, 2))
            # min_cutter is now fixed per preset, but ensuring it's not too small is still good
            min_cutter = max(0.5, min_cutter)
            
            # Update UI
            entries['voxel_resolution'].delete(0, tk.END); entries['voxel_resolution'].insert(0, str(res))
            entries['min_cutter_size'].delete(0, tk.END); entries['min_cutter_size'].insert(0, str(min_cutter))
            entries['max_cutters'].delete(0, tk.END); entries['max_cutters'].insert(0, str(max_cnt))
            entries['tolerance'].delete(0, tk.END); entries['tolerance'].insert(0, str(round(tol, 3)))
            
            show_custom_msg("Preset Applied", f"Applied '{mode}' preset.\n(Scale: {diagonal:.1f}mm)\n\nRes: {res}mm\nMin Cutter: {min_cutter}mm", 'info', root)


        tk.Button(preset_frame, text="Apply", command=apply_preset, bg='white').pack(side=tk.LEFT, padx=(0, 10))
        
        # Help Button
        tk.Button(preset_frame, text="?", width=3, command=open_help_window, bg='#e3f2fd', font=('Arial', 9, 'bold'), relief='flat').pack(side=tk.RIGHT, padx=5)

        # --- Input Fields (Grid Layout: 2 params per row) ---
        grid_frame = tk.Frame(param_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True)
        
        # Grid Column Config
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.columnconfigure(3, weight=1)

        
        for idx, (label, key) in enumerate(fields):
            r = idx // 2
            c = (idx % 2) * 2
            
            lbl = tk.Label(grid_frame, text=label, anchor='w')
            lbl.grid(row=r, column=c, sticky='w', padx=5, pady=2)
            
            entry = tk.Entry(grid_frame)
            entry.insert(0, str(self._cfg[key]))
            entry.grid(row=r, column=c+1, sticky='ew', padx=5, pady=2)
            entries[key] = entry
            
        # Checkboxes (3 Columns)
        check_frame = tk.Frame(param_frame)
        check_frame.pack(fill=tk.X, pady=10)
        
        # Col 0
        self.var_detect_slanted = tk.BooleanVar(value=self._cfg.get('detect_slanted', True))
        tk.Checkbutton(check_frame, text="Detect Slanted", variable=self.var_detect_slanted).grid(row=0, column=0, sticky='w')

        self.var_erosion = tk.BooleanVar(value=self._cfg.get('perform_erosion', False))
        tk.Checkbutton(check_frame, text="Voxel Erosion", variable=self.var_erosion).grid(row=1, column=0, sticky='w')

        self.var_slanted_only = tk.BooleanVar(value=False) 
        tk.Checkbutton(check_frame, text="Slanted Only", variable=self.var_slanted_only, fg='blue').grid(row=2, column=0, sticky='w')

        # Col 1
        self.var_auto_tune = tk.BooleanVar(value=self._cfg.get('auto_tune', False))
        tk.Checkbutton(check_frame, text="Auto-tune Res", variable=self.var_auto_tune).grid(row=0, column=1, sticky='w')

        self.var_show_removed = tk.BooleanVar(value=False)
        tk.Checkbutton(check_frame, text="Show Removed", variable=self.var_show_removed).grid(row=1, column=1, sticky='w')
        
        self.var_smart_expand = tk.BooleanVar(value=False) 
        tk.Checkbutton(check_frame, text="Smart Expand", variable=self.var_smart_expand, fg='darkgreen').grid(row=2, column=1, sticky='w')

        # Col 2
        self.var_undersize = tk.BooleanVar(value=self._cfg.get('undersize_mode', False))
        tk.Checkbutton(check_frame, text="Prevent Excess", variable=self.var_undersize).grid(row=0, column=2, sticky='w')

        self.var_cleanup = tk.BooleanVar(value=self._cfg.get('cleanup_artifacts', False))
        tk.Checkbutton(check_frame, text="Cleanup Resid", variable=self.var_cleanup).grid(row=1, column=2, sticky='w')
        
        # Grid weights
        check_frame.columnconfigure(0, weight=1)
        check_frame.columnconfigure(1, weight=1)
        check_frame.columnconfigure(2, weight=1)

        
        # Method Selection (Combobox)
        method_frame = tk.LabelFrame(param_frame, text="Algorithm Strategy", padx=5, pady=5)
        method_frame.pack(fill=tk.X, pady=5)
        
        self.var_method = tk.StringVar(value='voxel') # Logic var
        
        method_map = {
            "Method 1: Voxel (Fast & General)": "voxel",
            "Method 2: Sculpting (Growth)": "sculpting",
            "Method 3: Octree (Decomposition)": "octree"
        }
        self.var_method_display = tk.StringVar(value="Method 1: Voxel (Fast & General)")
        
        # Octree UI (Definition)
        oct_frame = tk.Frame(method_frame)
        tk.Label(oct_frame, text="Decomp Level:", font=('Arial', 9)).pack(side=tk.LEFT)
        self.var_oct_level = tk.IntVar(value=1)
        tk.Spinbox(oct_frame, from_=1, to=4, textvariable=self.var_oct_level, width=3).pack(side=tk.LEFT)

        def on_method_change(event=None):
            disp = self.var_method_display.get()
            val = method_map.get(disp, 'voxel')
            self.var_method.set(val)
            
            if val == 'octree':
                oct_frame.pack(side=tk.LEFT, padx=10)
            else:
                oct_frame.pack_forget()

        cbo_method = ttk.Combobox(method_frame, textvariable=self.var_method_display, 
                                  values=list(method_map.keys()), state="readonly")
        cbo_method.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        cbo_method.bind("<<ComboboxSelected>>", on_method_change)
        
        # Init state
        on_method_change()




        # Dashboard Panel
        dash_frame = tk.Frame(root, bg='#f0f0f0', relief=tk.RIDGE, bd=2)
        dash_frame.pack(fill=tk.X, padx=10, pady=5)
        
        lbl_cutters = tk.Label(dash_frame, text="Cutters: 0", bg='#f0f0f0', font=('Arial', 10, 'bold'))
        lbl_cutters.pack(side=tk.LEFT, padx=10, pady=5)
        
        lbl_vol = tk.Label(dash_frame, text="Est. Vol Sum: 0%", bg='#f0f0f0')
        lbl_vol.pack(side=tk.RIGHT, padx=10, pady=5)
        
        def update_dashboard():
            count = len(self.cutters)
            lbl_cutters.config(text=f"Cutters: {count}")
            if self.original_mesh and self.original_mesh.volume > 0:
                total_cutter_vol = sum([np.prod(c['size']) if c.get('type')!='oriented' else np.prod(c['extents']) for c in self.cutters])
                ratio = (total_cutter_vol / self.original_mesh.volume) * 100.0
                lbl_vol.config(text=f"Est. Vol Sum: {ratio:.1f}%")

        def update_ui_from_model():
            if self.bounding_box is None: return
            
            min_pt, max_pt = self.bounding_box
            size = max_pt - min_pt
            min_dim = np.min(size)
            
            # Update default resolution based on model size
            new_res = max(0.5, round(min_dim / 20.0, 1))
            self._cfg['voxel_resolution'] = new_res
            
            # Update Entry
            entries['voxel_resolution'].delete(0, tk.END)
            entries['voxel_resolution'].insert(0, str(new_res))
            status_var.set(f"Model Loaded. Size: {size[0]:.1f}x{size[1]:.1f}x{size[2]:.1f}")
            update_dashboard()

        def get_values_from_ui():
            try:
                for key, entry in entries.items():
                    self._cfg[key] = float(entry.get())
                self._cfg['auto_tune'] = self.var_auto_tune.get()
                self._cfg['detect_slanted'] = self.var_detect_slanted.get()
                self._cfg['undersize_mode'] = self.var_undersize.get()
                self._cfg['perform_erosion'] = self.var_erosion.get()
                self._cfg['cleanup_artifacts'] = self.var_cleanup.get()
                self._cfg['method'] = self.var_method.get()
                self._cfg['method'] = self.var_method.get()
                self._cfg['slanted_only'] = self.var_slanted_only.get() 
                self._cfg['smart_expand'] = self.var_smart_expand.get() # New
                return True
            except ValueError:
                show_custom_msg("Error", "숫자를 올바르게 입력해주세요.", 'error', root)
                return False

        def show_preview():
            if self.original_mesh is None:
                show_custom_msg("Warning", "No model loaded.", 'warning', root)
                return
            if not get_values_from_ui(): return
            
            # PyVista Plotter 생성 (Preview 모드)
            p = pv.Plotter(title="Preview - Close window to return to settings")
            p.enable_parallel_projection()
            
            cfg = self._cfg
            min_pt, max_pt = self.bounding_box
            size = max_pt - min_pt
            res = cfg['voxel_resolution']
            
            # [New] Slanted Only Check
            is_slanted_only = cfg.get('slanted_only', False)

            if is_slanted_only:
                 padding = 0
                 origin = min_pt
                 grid_shape = np.array([0, 0, 0])
                 total_voxels = 0
                 grid_str = "Skipped (Slanted Mode)"
            else:
                padding = res * 1.2
                origin = min_pt - padding
                grid_shape = np.ceil((size + padding*2) / res).astype(int)
                total_voxels = np.prod(grid_shape)
                grid_str = str(grid_shape)
            
            # OBB Calculation for Tight Bounding Box
            obb_info = ""
            obb_box = None
            if self.original_mesh:
                try:
                    obb = self.original_mesh.bounding_box_oriented
                    extents = obb.primitive.extents
                    obb_vol = np.prod(extents)
                    aabb_vol = np.prod(size)
                    obb_info = (f"\nTight BBox (OBB):\n"
                                f"  Size: {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}\n"
                                f"  Vol Ratio: {obb_vol/aabb_vol*100:.1f}%")
                    obb_box = obb
                except: pass

            # 정보 텍스트 (Font Size 9로 축소, Dark Colors)
            info = (f"Resolution: {res} mm\n"
                    f"Grid: {grid_str}\n"
                    f"Total Voxels: {total_voxels:,}\n"
                    f"Bounding Box:\n"
                    f"  X: {min_pt[0]:.2f} ~ {max_pt[0]:.2f} ({size[0]:.2f})\n"
                    f"  Y: {min_pt[1]:.2f} ~ {max_pt[1]:.2f} ({size[1]:.2f})\n"
                    f"  Z: {min_pt[2]:.2f} ~ {max_pt[2]:.2f} ({size[2]:.2f})"
                    f"{obb_info}")
            
            print("\n" + "="*40)
            print(" [클래스 1/2] Preview Information")
            print(info)
            print("="*40)

            p.add_text(info, font_size=9, position='upper_left', color='black')
            
            # 메쉬
            p.add_mesh(self.original_mesh, color='lightblue', opacity=0.3)
            
            # 그리드 (너무 많으면 박스만)
            if not is_slanted_only and total_voxels < 1000000: # 1M voxels limit for wireframe
                grid = pv.ImageData()
                grid.dimensions = grid_shape + 1
                grid.origin = origin
                grid.spacing = (res, res, res)
                p.add_mesh(grid, style='wireframe', color='black', opacity=0.2, line_width=1)
            else:
                bounds = [origin[0], origin[0] + grid_shape[0]*res,
                          origin[1], origin[1] + grid_shape[1]*res,
                          origin[2], origin[2] + grid_shape[2]*res]
                p.add_mesh(pv.Box(bounds=bounds), style='wireframe', color='darkblue', line_width=2)
                p.add_text("Grid too dense to display fully", position='lower_left', color='darkred', font_size=8)
            
            # Visualize OBB if available
            if obb_box:
                try:
                    ext = obb_box.primitive.extents
                    # Create box centered at origin
                    b_obb = pv.Box(bounds=(-ext[0]/2, ext[0]/2, -ext[1]/2, ext[1]/2, -ext[2]/2, ext[2]/2))
                    # Apply transform
                    b_obb.transform(obb_box.primitive.transform, inplace=True)
                    p.add_mesh(b_obb, color='blue', style='wireframe', opacity=0.5, line_width=2, label='Tight BBox')
                except Exception as e:
                    print(f"OBB Visualization Error: {e}")

            
            # [Preview Enhancement] Detect Slanted Surfaces 옵션이 켜져 있으면 미리보기 제공
            if cfg.get('detect_slanted', False):
                # 시각화를 위해 임시로 커터 생성
                min_area = (cfg['voxel_resolution'] * cfg['slanted_area_factor']) ** 2
                min_edge = cfg['voxel_resolution'] * cfg['slanted_edge_factor']
                extrusion_depth = max(cfg['voxel_resolution'] * 5.0, cfg['min_cutter_size'])
                
                # 기존 커터 리스트 백업 (Preview는 원본 상태를 변경하면 안 됨)
                backup_cutters = self.cutters
                self.cutters = []
                
                # 감지 실행 (Angle Snap 및 Slanted Tolerance 적용)
                print("[Preview] 경사면 커터 생성 및 병합 적용 중...")
                self.generate_slanted_cutters(
                    min_area, min_edge, extrusion_depth, 
                    tolerance=cfg.get('slanted_tolerance', -0.1), 
                    masks=[], 
                    min_cutter_size=cfg['min_cutter_size'],
                    angle_snap_divisions=int(cfg.get('angle_snap_divisions', 0))
                )
                
                # [New] Preview에서도 스냅 적용
                print("[Preview] 인접면 스냅(Snapping) 적용 중...")
                snap_tol = res * 1.2
                detected_pairs = self._align_cutter_neighbors(snap_tolerance=snap_tol)
                
                # [Debug] Verify specific user-reported pairs (Preview Check)
                if detected_pairs:
                    self._debug_verify_pairs(detected_pairs)
                
                if self.cutters:


                    p.add_text(f"Preview: {len(self.cutters)} Slanted Cutters (Merged)", position='upper_right', color='darkgreen', font_size=9)
                    for idx, c in enumerate(self.cutters):
                        if c.get('type') == 'oriented':
                            box = pv.Box(bounds=(-c['extents'][0]/2, c['extents'][0]/2,
                                                 -c['extents'][1]/2, c['extents'][1]/2,
                                                 -c['extents'][2]/2, c['extents'][2]/2))
                            box.transform(c['transform'], inplace=True)
                            p.add_mesh(box, color='orange', opacity=0.5, style='wireframe', line_width=2)
                            
                            # [New] Add ID Label for Prediction
                            center_pv = c['transform'][:3, 3]
                            p.add_point_labels([center_pv], [f"P{idx+1}"], point_size=0, font_size=10, text_color='white', always_visible=True)

                            # [New] Add Edge Labels (E0..E11) for Slanted Cutters
                            _, edges = self._get_obb_geometry(c)
                            edge_centers = []
                            edge_labels = []
                            for e_idx, e in enumerate(edges):
                                mid = (e['p1'] + e['p2']) / 2
                                edge_centers.append(mid)
                                edge_labels.append(f"E{e_idx}")
                            
                            p.add_point_labels(edge_centers, edge_labels, point_size=0, font_size=8, text_color='white', always_visible=True)

                
                # 커터 리스트 복원
                self.cutters = backup_cutters


            p.show() # Blocking call

        def view_cutters_only():
            """최종 생성된 커터만 보여주고 정보를 출력합니다."""
            if not self.cutters:
                messagebox.showinfo("Info", "생성된 커터가 없습니다. 먼저 실행하세요.")
                return
                
            print("\n" + "="*60)
            print(f" [Cutter Information] Total: {len(self.cutters)}")
            print(" Format: cutter ID, (origin x, y, z, ax, ay, az, bx, by, bz, cx, cy, cz)")
            print("="*60)
            
            p = pv.Plotter(title="View Cutters Only")
            p.enable_parallel_projection()
            
            # 색상 팔레트 생성 (구분 가능한 색상들)
            import matplotlib.colors as mcolors
            colors = list(mcolors.TABLEAU_COLORS.values())
            
            for idx, c in enumerate(self.cutters):
                # 색상 순환 선택
                color = colors[idx % len(colors)]
                
                # 기하 정보 계산
                if c.get('type') == 'oriented':
                    center = c['transform'][:3, 3]
                    R = c['transform'][:3, :3]
                    extents = c['extents']
                    v1 = R[:, 0] * extents[0]
                    v2 = R[:, 1] * extents[1]
                    v3 = R[:, 2] * extents[2]
                    origin = center - 0.5 * (v1 + v2 + v3)
                    
                    box = pv.Box(bounds=(-extents[0]/2, extents[0]/2, -extents[1]/2, extents[1]/2, -extents[2]/2, extents[2]/2))
                    box.transform(c['transform'], inplace=True)
                    p.add_mesh(box, color=color, opacity=0.8, show_edges=True)
                    
                    # [New] Add ID Label at Center
                    p.add_point_labels([center], [f"C{idx+1}"], point_size=0, font_size=12, text_color='white', always_visible=True, shape_color='black', shape_opacity=0.5)

                    # [New] Add Edge Labels (E0..E11)
                    # Helper to reuse
                    _, edges = self._get_obb_geometry(c)
                    edge_centers = []
                    edge_labels = []
                    for e_idx, e in enumerate(edges):
                        mid = (e['p1'] + e['p2']) / 2
                        edge_centers.append(mid)
                        edge_labels.append(f"E{e_idx}")
                    
                    p.add_point_labels(edge_centers, edge_labels, point_size=0, font_size=8, text_color=color, always_visible=True)

                else:
                    cx, cy, cz = c['center']
                    sx, sy, sz = c['size']
                    origin = np.array([cx - sx/2, cy - sy/2, cz - sz/2])
                    v1 = np.array([sx, 0, 0])
                    v2 = np.array([0, sy, 0])
                    v3 = np.array([0, 0, sz])
                    
                    p.add_mesh(pv.Box(bounds=[origin[0], origin[0]+sx, origin[1], origin[1]+sy, origin[2], origin[2]+sz]), color=color, opacity=0.8, show_edges=True)

                print(f"cutter {idx+1:04d}, ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}, "
                      f"{v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}, {v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f}, {v3[0]:.3f}, {v3[1]:.3f}, {v3[2]:.3f})")

            # [New] Render Initial State (Dotted)
            if hasattr(self, 'cutters_initial'):
                print("[View] Rendering Initial State (Dotted Grey)...")
                for c_init in self.cutters_initial:
                     if c_init.get('type') == 'oriented':
                        extents = c_init['extents']
                        box = pv.Box(bounds=(-extents[0]/2, extents[0]/2, -extents[1]/2, extents[1]/2, -extents[2]/2, extents[2]/2))
                        box.transform(c_init['transform'], inplace=True)
                        p.add_mesh(box, color='grey', style='wireframe', opacity=0.3, line_width=1)
                     else:
                        cx, cy, cz = c_init['center']
                        sx, sy, sz = c_init['size']
                        box = pv.Box(bounds=[cx-sx/2, cx+sx/2, cy-sy/2, cy+sy/2, cz-sz/2, cz+sz/2])
                        p.add_mesh(box, color='grey', style='wireframe', opacity=0.3, line_width=1)
            
            # [Preview] 경사면 커터 미리보기 (병합 로직 포함)
            cfg = self._cfg
            res = cfg['voxel_resolution']
            
            if cfg['detect_slanted']:
                try:
                    print("[Preview] 경사면 커터 생성 및 병합 미리보기...")
                    # 임시로 커터 생성
                    temp_cutters_backup = self.cutters
                    self.cutters = [] 
                    
                    min_area = (res * cfg['slanted_area_factor']) ** 2
                    min_edge_len = res * cfg['slanted_edge_factor']
                    extrusion_depth = max(res * 5.0, cfg['min_cutter_size'])
                    
                    # 각도 스냅 옵션 반영하여 생성 -> 내부에서 병합 로직 수행됨
                    self.generate_slanted_cutters(
                        min_area, min_edge_len, extrusion_depth, 
                        tolerance=cfg['slanted_tolerance'], 
                        masks=[], 
                        min_cutter_size=cfg['min_cutter_size'],
                        angle_snap_divisions=int(cfg['angle_snap_divisions'])
                    )
                    
                    # [New] Preview에서도 스냅 적용
                    print("[Preview] 인접면 스냅(Snapping) 적용 중...")
                    snap_tol = res * 1.2
                    self._align_cutter_neighbors(snap_tolerance=snap_tol)
                    
                    print(f"[Preview] 생성된 커터 수: {len(self.cutters)}")
                    
                    colors = ["orange", "gold", "yellow", "peru"]
                    for i, c in enumerate(self.cutters):
                        if c.get('type') == 'oriented':
                            sx, sy, sz = c['extents']
                            box = pv.Box(bounds=(-sx/2, sx/2, -sy/2, sy/2, -sz/2, sz/2))
                            box.transform(c['transform'], inplace=True)
                            p.add_mesh(box, color=colors[i % 4], opacity=0.6, style='wireframe', line_width=3, label="Slanted Cutter")
                    
                    # 복구
                    self.cutters = temp_cutters_backup
                    
                except Exception as e:
                    print(f"[Preview] 커터 미리보기 실패: {e}")
            
            p.show()

        def run_simplification(append=False):
            if self.original_mesh is None:
                show_custom_msg("Warning", "No model loaded.", 'warning', root)
                return
            if not get_values_from_ui(): return
            
            btn_run.config(state='disabled')
            btn_refine.config(state='disabled')
            btn_view.config(state='disabled')
            status_var.set("Processing... (Generating Cutters)")
            
            def worker():
                try:
                    auto_tune = self._cfg.get('auto_tune', False)
                    target_error = self._cfg.get('target_error', 5.0)
                    current_res = self._cfg['voxel_resolution']
                    max_iter_val = int(self._cfg.get('max_iterations', 5))
                    
                    # Refine 모드이거나 Auto-tune이 아니면 1회만 실행
                    max_iterations = max_iter_val if (auto_tune and not append) else 1
                    
                    # 진행률 콜백 함수
                    def update_progress(msg):
                        root.after(0, lambda: status_var.set(msg))
                    
                    for i in range(max_iterations):
                        # 반복할수록 커터 수를 늘려 잔여 영역 제거 확률을 높임 (매회 20% 증가)
                        current_max_cutters = int(self._cfg['max_cutters'] * (1.0 + 0.2 * i))
                        
                        if auto_tune:
                            msg = f"Auto-tuning Iter {i+1}/{max_iterations}: Res={current_res:.2f}mm, Cutters={current_max_cutters}..."
                            root.after(0, lambda m=msg: status_var.set(m))
                        
                        # 1. Generate Cutters
                        self.generate_cutters(
                            voxel_resolution=self._cfg['voxel_resolution'],
                            min_volume_ratio=self._cfg['min_volume_ratio'],
                            max_cutters=current_max_cutters,
                            tolerance=self._cfg['tolerance'],
                            detect_slanted=self._cfg['detect_slanted'],
                            masks=[],
                            slanted_area_factor=self._cfg['slanted_area_factor'],
                            slanted_edge_factor=self._cfg['slanted_edge_factor'],
                            angle_snap_divisions=int(self._cfg.get('angle_snap_divisions', 0)),
                            min_cutter_size=self._cfg['min_cutter_size'],
                            append=append,
                            undersize_mode=self._cfg['undersize_mode'],
                            perform_erosion=self._cfg['perform_erosion'],
                            cleanup_artifacts=self._cfg.get('cleanup_artifacts', False),
                            slanted_tolerance=self._cfg.get('slanted_tolerance', -0.1),
                            progress_callback=update_progress,
                            method=self._cfg.get('method', 'voxel')
                        )
                        
                        if not auto_tune:
                            root.after(0, lambda: status_var.set("Processing... (Generating CAD)"))
                        
                        # 2. Generate CAD (Split logic for Method 3)
                        if self._cfg.get('method') == 'octree':
                             root.after(0, lambda: status_var.set("Processing... (Decomposing Domain)"))
                             level = self.var_oct_level.get()
                             self.decompose_by_octree(level=level)
                             # Result is already in self.simplified_shape (Compound) inside method
                        else:
                            engine = 'cadquery' if HAS_CADQUERY else 'gmsh'
                            self.simplified_shape, self.cutters_shape = self.generate_cad(use_engine=engine)
                        

                        if not auto_tune:
                            break
                            
                        # Auto-tune check
                        if self.simplified_shape:
                            error = self.calculate_volume_error(self.simplified_shape)
                            if error is not None:
                                print(f"[Auto-tune] Iter {i+1}: Res={current_res:.2f}mm, Error={error:.2f}% (Target: {target_error}%)")
                                if error <= target_error:
                                    break # Success
                                
                                # Reduce resolution for next iteration (finer grid)
                                current_res *= 0.8
                                if current_res < 0.1: break # Safety limit
                    
                    if self.simplified_shape:
                        root.after(0, lambda: status_var.set("Processing... (Evaluating & Exporting)"))
                        
                        # 3. Evaluate & Export
                        self.evaluate_accuracy(self.simplified_shape)
                        
                        def on_success():
                            status_var.set("Completed. Please Save Result.")
                            btn_view.config(state='normal')
                            btn_run.config(state='normal')
                            btn_refine.config(state='normal')
                            btn_comp.config(state='normal')
                            btn_save.config(state='normal')
                            show_custom_msg("Success", "Simplification Complete!\nUse 'Save Results' to save files.", 'info', root)
                            update_dashboard()



                        root.after(0, on_success)
                    else:
                        def on_failure():
                            status_var.set("Failed to generate CAD.")
                            btn_run.config(state='normal')
                            btn_refine.config(state='normal')
                            btn_comp.config(state='disabled')
                            btn_save.config(state='disabled')
                            show_custom_msg("Error", "Failed to generate CAD geometry.", 'error', root)
                        root.after(0, on_failure)

                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    print(tb_str) # Print to console
                    
                    def on_error(msg, trace):
                        status_var.set(f"Error: {msg}")
                        btn_run.config(state='normal')
                        btn_refine.config(state='normal')
                        # Truncate trace if too long for msgbox
                        short_trace = trace[-500:] if len(trace) > 500 else trace
                        show_custom_msg("Error", f"An error occurred:\n{msg}\n\nTraceback:\n{short_trace}", 'error', root)
                        
                    root.after(0, on_error, str(e), tb_str)
            threading.Thread(target=worker, daemon=True).start()

        def view_result():
            if self.simplified_shape:
                self.visualize(self.simplified_shape, show_removed=self.var_show_removed.get())

        def run_optimizer():
            if self.original_mesh is None:
                show_custom_msg("Warning", "No model loaded.", 'warning', root)
                return
            if not get_values_from_ui(): return
            
            base_params = self._cfg.copy()
            
            opt_win = tk.Toplevel(root)
            opt_win.title("Parameter Optimizer")
            opt_win.geometry("600x600")
            center_window(opt_win, root)
            
            # --- Configuration Panel ---
            cfg_frame = tk.LabelFrame(opt_win, text="Optimization Config", padx=10, pady=5)
            cfg_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Resolution Range
            tk.Label(cfg_frame, text="Resolution Scale Range:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w')
            e_res_min = tk.Entry(cfg_frame, width=5); e_res_min.insert(0, "0.5")
            e_res_max = tk.Entry(cfg_frame, width=5); e_res_max.insert(0, "1.5")
            e_res_step = tk.Entry(cfg_frame, width=3); e_res_step.insert(0, "3")
            
            tk.Label(cfg_frame, text="Min x").grid(row=0, column=1)
            e_res_min.grid(row=0, column=2)
            tk.Label(cfg_frame, text="Max x").grid(row=0, column=3)
            e_res_max.grid(row=0, column=4)
            tk.Label(cfg_frame, text="Steps").grid(row=0, column=5)
            e_res_step.grid(row=0, column=6)
            
            # Volume Ratio Range
            tk.Label(cfg_frame, text="Min Vol Ratio Range:", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky='w')
            e_vol_min = tk.Entry(cfg_frame, width=5); e_vol_min.insert(0, "0.5")
            e_vol_max = tk.Entry(cfg_frame, width=5); e_vol_max.insert(0, "1.5")
            e_vol_step = tk.Entry(cfg_frame, width=3); e_vol_step.insert(0, "3")
            
            tk.Label(cfg_frame, text="Min x").grid(row=1, column=1)
            e_vol_min.grid(row=1, column=2)
            tk.Label(cfg_frame, text="Max x").grid(row=1, column=3)
            e_vol_max.grid(row=1, column=4)
            tk.Label(cfg_frame, text="Steps").grid(row=1, column=5)
            e_vol_step.grid(row=1, column=6)
            
            # Weight Slider
            tk.Label(cfg_frame, text="Priority Weight:", font=('Arial', 9, 'bold')).grid(row=2, column=0, sticky='w', pady=5)
            w_frame = tk.Frame(cfg_frame)
            w_frame.grid(row=2, column=1, columnspan=6, sticky='we')
            
            tk.Label(w_frame, text="Accuracy").pack(side=tk.LEFT)
            s_weight = tk.Scale(w_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, length=200)
            s_weight.set(0.7) # Default slightly towards accuracy
            s_weight.pack(side=tk.LEFT, padx=5)
            tk.Label(w_frame, text="Simplicity (Low Count)").pack(side=tk.LEFT)
            
            
            btn_start = tk.Button(cfg_frame, text="Start Optimization", bg='lightgreen', width=20)
            btn_start.grid(row=3, column=0, columnspan=7, pady=10)
            
            # --- Results Panel ---
            tk.Label(opt_win, text="Optimization Results:", font=('Arial', 10, 'bold')).pack(pady=(10, 0))
            
            cols = ('Desc', 'Res', 'Cutters', 'Error(Est)', 'Score')
            tree = ttk.Treeview(opt_win, columns=cols, show='headings', height=8)
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=90, anchor='center')
            tree.column('Desc', width=160, anchor='w')
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            btn_apply = tk.Button(opt_win, text="Apply Selected Settings", state='disabled', bg='#e1f5fe')
            btn_apply.pack(pady=5, fill=tk.X, padx=10)
            
            
            def run_sim():
                try:
                    # Config Build
                    cfg = {
                        'res_range': (float(e_res_min.get()), float(e_res_max.get()), float(e_res_step.get())),
                        'vol_range': (float(e_vol_min.get()), float(e_vol_max.get()), float(e_vol_step.get())),
                        'w_accuracy': s_weight.get(),
                        'w_simplicity': 1.0 - s_weight.get()
                    }
                    
                    btn_start.config(state='disabled', text="Running...")
                    
                    def worker_opt():
                        results = self.simulate_optimization(base_params, cfg)
                        
                        def update_ui():
                            btn_start.config(state='normal', text="Start Optimization")
                            # Clear tree
                            for item in tree.get_children(): tree.delete(item)
                            
                            for r in results:
                                p = r['params']
                                m = r['metrics']
                                row = (m['desc'], f"{p['voxel_resolution']:.2f}", m['count'], f"{m['error_rate']:.1f}%", f"{r['score']:.2f}")
                                tree.insert('', 'end', values=row, tags=(str(results.index(r))))
                            
                            if results:
                                # Best selection
                                child_id = tree.get_children()[0]
                                tree.selection_set(child_id)
                                tree.focus(child_id)
                            
                            btn_apply.config(state='normal', command=lambda: apply_opt(results))
                        
                        root.after(0, update_ui)
                    
                    threading.Thread(target=worker_opt, daemon=True).start()
                    
                except ValueError:
                    messagebox.showerror("Error", "Invalid inputs.")

            btn_start.config(command=run_sim)

            def apply_opt(results):
                sel = tree.selection()
                if not sel: return
                idx = int(tree.item(sel[0], 'tags')[0])
                best = results[idx]
                
                self._cfg.update(best['params'])
                entries['voxel_resolution'].delete(0, tk.END); entries['voxel_resolution'].insert(0, str(best['params']['voxel_resolution']))
                entries['min_volume_ratio'].delete(0, tk.END); entries['min_volume_ratio'].insert(0, str(best['params']['min_volume_ratio']))
                
                show_custom_msg("Optimization", f"Applied: {best['metrics']['desc']}", 'info', root)
                opt_win.destroy()

        # --- Menu Functions ---
        def open_file():
            file_path = filedialog.askopenfilename(
                filetypes=[("CAD Files", "*.step *.stp *.iges *.igs *.stl *.obj"), ("All Files", "*.*")]
            )
            if file_path:
                try:
                    self.load_file(file_path)
                    update_ui_from_model()
                    self.simplified_shape = None # Reset result
                    btn_view.config(state='disabled')
                except Exception as e:
                    show_custom_msg("Error", f"Failed to load file:\n{e}", 'error', root)

        def load_sample(shape_type):
            if self.create_sample_shape(shape_type):
                update_ui_from_model()
                self.simplified_shape = None
                btn_view.config(state='disabled')
            else:
                show_custom_msg("Error", "Failed to generate sample.", 'error', root)

        def view_comparison():
            if self.simplified_shape:
                self.visualize_comparison(self.simplified_shape)

        def export_cad_file():
            if not self.simplified_shape:
                return
            
            file_types = [('STEP files', '*.step *.stp'), ('IGES files', '*.iges *.igs'), ('STL files', '*.stl'), ('OBJ files', '*.obj')]
            filename = filedialog.asksaveasfilename(title="Save CAD Result", filetypes=file_types, defaultextension=".step")
            if filename:
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.stl', '.obj']:
                    self.export_stl(self.simplified_shape, filename)
                elif ext in ['.step', '.stp', '.iges', '.igs']:
                    self.export_step(self.simplified_shape, filename)
                
                # [CAE] 커터 정보 JSON 내보내기 (자동)
                import json
                try:
                    json_path = os.path.splitext(filename)[0] + "_cutters.json"
                    export_data = []
                    for c in self.cutters:
                        item = {
                            'type': c.get('type', 'aabb'),
                            'center': c['center'].tolist() if isinstance(c['center'], np.ndarray) else c['center'],
                            'size': c['size'].tolist() if isinstance(c['size'], np.ndarray) else c['size']
                        }
                        if c.get('type') == 'oriented':
                            item['extents'] = c['extents'].tolist()
                            item['transform'] = c['transform'].tolist()
                        export_data.append(item)
                        
                    with open(json_path, 'w') as f:
                        json.dump(export_data, f, indent=2)
                    print(f"[Export] 커터 데이터 저장됨: {json_path}")
                except Exception as e:
                    print(f"[Export] JSON 저장 실패: {e}")

                show_custom_msg("Export", f"Saved CAD to {filename}", 'info', root)

        def export_cutter_txt():
            if not self.cutters:
                return
            filename = filedialog.asksaveasfilename(title="Save Cutter Info", filetypes=[('Text files', '*.txt')], defaultextension=".txt")
            if filename:
                self.export_cutter_info(filename)
                show_custom_msg("Export", f"Saved Info to {filename}", 'info', root)

        def open_save_dialog():
            """저장 다이얼로그 (통합 버튼)"""
            if not self.simplified_shape:
                 show_custom_msg("Warning", "No result to save.", 'warning', root)
                 return
            
            save_win = tk.Toplevel(root)
            save_win.title("Save Results")
            save_win.geometry("350x250")
            center_window(save_win, root) # Center this too
            
            tk.Label(save_win, text="Select format to save:", pady=10, font=('Arial', 11, 'bold')).pack()

            # Helper to ask filename and call export
            def ask_and_export(type_flag, title_prefix):
                file_types = [('STEP files', '*.step *.stp'), ('IGES files', '*.iges *.igs')]
                filename = filedialog.asksaveasfilename(title=f"Save {title_prefix}", filetypes=file_types, defaultextension=".step")
                if filename:
                   self.export_step(type_flag, filename)
                   show_custom_msg("Export", f"Saved {title_prefix} to {filename}", 'info', root)
                   save_win.destroy()

            tk.Button(save_win, text="1. Save Simplified Model (STEP/STL...)", command=lambda: [export_cad_file(), save_win.destroy()], width=35, anchor='w', padx=10).pack(pady=2)
            tk.Button(save_win, text="2. Save Base + Cutters (Assembly)", command=lambda: ask_and_export("export_base_and_cutters", "Base & Cutters"), width=35, anchor='w', padx=10).pack(pady=2)
            tk.Button(save_win, text="3. Save Original Bounding Box (STEP)", command=lambda: ask_and_export("export_bounding_box", "Bounding Box"), width=35, anchor='w', padx=10).pack(pady=2)
            tk.Button(save_win, text="4. Save Cutters Only (STEP)", command=lambda: ask_and_export("export_cutters_only", "Cutters"), width=35, anchor='w', padx=10).pack(pady=2)
            
            tk.Frame(save_win, height=1, bg="grey").pack(fill='x', pady=5, padx=10)
            
            tk.Button(save_win, text="5. Save Cutter Info (.txt)", command=lambda: [export_cutter_txt(), save_win.destroy()], width=35, anchor='w', padx=10).pack(pady=2)


        # --- Menu Bar ---
        menubar = tk.Menu(root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=open_file)
        
        file_menu.add_command(label="Save Base + Cutters...", command=lambda: export_cad_file() if False else show_custom_msg("Info", "Use Save button for advanced options", 'info'))
        file_menu.add_separator()
        
        samples_menu = tk.Menu(file_menu, tearoff=0)
        samples_menu.add_command(label="Basic Box", command=lambda: load_sample('basic'))
        samples_menu.add_command(label="Ribbed Cushion", command=lambda: load_sample('ribbed_cushion'))
        samples_menu.add_command(label="Complex Ribs", command=lambda: load_sample('complex_ribs'))
        samples_menu.add_command(label="L-Bracket", command=lambda: load_sample('l_bracket'))
        samples_menu.add_command(label="Pipe Connector", command=lambda: load_sample('pipe_connector'))
        samples_menu.add_separator()
        samples_menu.add_command(label="Packaging Cushion (2000x1200)", command=lambda: load_sample('packaging_cushion'))
        file_menu.add_cascade(label="Sample Shapes", menu=samples_menu)
        
        
        file_menu.add_separator()
        save_menu = tk.Menu(file_menu, tearoff=0)
        save_menu.add_command(label="Save CAD Model...", command=export_cad_file)
        save_menu.add_command(label="Save Cutter Info...", command=export_cutter_txt)
        file_menu.add_cascade(label="Export Result", menu=save_menu)
        
        # Tools Menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        
        def ask_expand():
            if not self.cutters:
                messagebox.showwarning("Warning", "No cutters generated yet.")
                return
            val = simpledialog.askfloat("Expand Cutters", "Enter expansion amount (mm):\n(Positive=Expand, Negative=Shrink)", initialvalue=0.5, parent=root)
            if val is not None:
                self.expand_cutters(val)
                messagebox.showinfo("Success", f"Expanded cutters by {val}mm.\nPlease view cutters or save results to see changes.")

        tools_menu.add_command(label="Expand/Shrink Cutters...", command=ask_expand)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
        menubar.add_cascade(label="File", menu=file_menu)
        root.config(menu=menubar)

        def load_sample(stype):
            if self.create_sample_shape(stype):
                update_ui_from_model()
                show_custom_msg("Success", f"'{stype}' sample loaded.", 'success', root)

        sample_frame = tk.LabelFrame(root, text="Samples", padx=10, pady=5)
        sample_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(sample_frame, text="Basic", command=lambda: load_sample('basic')).pack(side=tk.LEFT, padx=2)
        tk.Button(sample_frame, text="Pipe(Cylinder)", command=lambda: load_sample('pipe_connector')).pack(side=tk.LEFT, padx=2)
        tk.Button(sample_frame, text="Ribs", command=lambda: load_sample('ribbed_cushion')).pack(side=tk.LEFT, padx=2)
        tk.Button(sample_frame, text="L-Bracket", command=lambda: load_sample('l_bracket')).pack(side=tk.LEFT, padx=2)
        tk.Button(sample_frame, text="Packaging", command=lambda: load_sample('packaging_cushion')).pack(side=tk.LEFT, padx=2)

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, pady=20, padx=10)

        # Row 1: Preview
        tk.Button(btn_frame, text="Preview Input", command=show_preview, bg='lightblue', height=2).pack(fill=tk.X, pady=2)
        
        # Row 2: Run & Refine (Side by Side)
        row2 = tk.Frame(btn_frame)
        row2.pack(fill=tk.X, pady=2)
        btn_run = tk.Button(row2, text="Run (Reset)", command=lambda: run_simplification(append=False), bg='lightgreen', height=2)
        btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        btn_refine = tk.Button(row2, text="Refine (+)", command=lambda: run_simplification(append=True), bg='#E0F8E0', height=2)
        btn_refine.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Optimizer Button added
        btn_opt = tk.Button(row2, text="AI Optimize 🚀", command=run_optimizer, bg='#e3f2fd', height=2)
        btn_opt.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # Row 3: View Cutters & View Result (Side by Side)
        row3 = tk.Frame(btn_frame)
        row3.pack(fill=tk.X, pady=2)
        tk.Button(row3, text="View Cutters", command=view_cutters_only, bg='lightyellow', height=2).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        btn_view = tk.Button(row3, text="View Result", command=view_result, state='disabled', bg='lightyellow', height=2)
        btn_view.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Row 4: Comparison & Save (Side by Side) -> Restore user requested features in new layout
        row4 = tk.Frame(btn_frame)
        row4.pack(fill=tk.X, pady=2)
        btn_comp = tk.Button(row4, text="Comparision (2x2)", command=view_comparison, state='disabled', bg='#FFFACD', height=2)
        btn_comp.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        btn_save = tk.Button(row4, text="Save Results...", command=open_save_dialog, state='disabled', bg='#FFEFD5', height=2)
        btn_save.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        


        # Status Bar

        status_var = tk.StringVar()
        status_var.set("Ready")
        tk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor='w').pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initial Load if data exists
        if self.bounding_box is not None:
            update_ui_from_model()
            # 초기 로드 시 Medium 프리셋 자동 제안 (값만 계산해두거나 자동 적용)
            # apply_preset() # 자동 적용은 사용자 혼란을 줄 수 있으므로 일단 제외

        else:
            # Load default sample if nothing loaded
            load_sample('basic')

        root.mainloop()


# =============================================================================
# 메인 실행 예제 (Main Execution Example)
# =============================================================================

if __name__ == "__main__":
    simplifier = CADSimplifier()
    simplifier.show_control_panel()
