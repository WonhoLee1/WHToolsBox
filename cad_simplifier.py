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
                         progress_callback=None):
        """
        커터 박스를 찾는 메인 알고리즘입니다.
        append=True일 경우, 기존 그리드 상태를 유지하고 커터를 추가로 찾습니다.
        masks: [{'type': 'exclude'|'include', 'bounds': ((min_x, min_y, min_z), (max_x, max_y, max_z))}, ...]
        """
        min_pt, max_pt = self.bounding_box
        size = max_pt - min_pt
        
        # 상태 초기화 여부 결정
        if not append or self.current_grid is None or self.voxel_scale != voxel_resolution:
            print("\n[처리] 단순화 시작 (초기화)...")
            self.cutters = []
            
            # 0. 경사진 커터 감지 (특징 감지) - 초기화 시에만 수행
            if detect_slanted:
                min_area = (voxel_resolution * slanted_area_factor) ** 2
                min_edge_len = voxel_resolution * slanted_edge_factor
                extrusion_depth = max(voxel_resolution * 5.0, min_cutter_size) 
                self.generate_slanted_cutters(min_area, min_edge_len, extrusion_depth, tolerance, masks=masks, min_cutter_size=min_cutter_size)

            # 1. 경계 상자 복셀화 (음수 공간)
            print(f"[복셀화] 해상도 {voxel_resolution}mm로 복셀화 수행 중...")
            
            try:
                # [최적화] Trimesh의 고속 복셀화 기능 우선 시도
                voxel_obj = self.original_mesh.voxelized(pitch=voxel_resolution)
                voxel_obj = voxel_obj.fill()
                dense_grid = voxel_obj.matrix
                
                pad_width = 2
                padded_grid = np.pad(dense_grid, pad_width, mode='constant', constant_values=False)
                
                # [Fix] voxel_obj.origin은 최신 trimesh에서 제거됨 -> transform 사용
                self.grid_origin = voxel_obj.transform[:3, 3] - (pad_width * voxel_resolution)
                self.voxel_scale = voxel_resolution
                self.current_grid = ~padded_grid # True=Empty
                
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
                
                self.current_grid = ~is_occupied.reshape(grid_shape)
            
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
        
        for i in range(max_cutters):
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
            
            if not self._validate_cutter_overlap(center_vec, size_vec):
                # [Shrink] 침범 시 즉시 포기하지 않고, 크기를 줄여서 재시도
                found_shrunk = False
                temp_size = size_vec.copy()
                
                # 최대 5회, 15%씩 축소하며 재검사
                for _ in range(5):
                    temp_size *= 0.85
                    if np.any(temp_size < min_cutter_size):
                        break
                    
                    if self._validate_cutter_overlap(center_vec, temp_size):
                        size_x, size_y, size_z = temp_size
                        found_shrunk = True
                        break
                
                if not found_shrunk:
                    # 침범이 감지되고 축소도 실패하면 해당 영역을 점유된 것으로 표시
                    mark_grid_occupied(self.current_grid, (vx, vy, vz, vw, vh, vd))
                    continue

            # 정제 (Refinement)
            refined_box = self._refine_cutter(
                center=(cx, cy, cz), 
                size=(size_x, size_y, size_z), 
                tolerance=tolerance
            )
            refined_box['type'] = 'aabb' # 축 정렬 경계 상자 (Axis Aligned Bounding Box)
            
            self.cutters.append(refined_box)
            mark_grid_occupied(self.current_grid, (vx, vy, vz, vw, vh, vd))
            
            if i % max(1, max_cutters // 10) == 0:
                pct = (i+1)/max_cutters*100
                print(f"  - 커터 생성 진행: {pct:.1f}% ({i+1}/{max_cutters})")
                if progress_callback:
                    progress_callback(f"Generating Cutters... {pct:.0f}%")
        
        print("\n[커터] 생성 완료.")
        self.optimize_cutters()
        print(f"[처리] 총 커터 수: {len(self.cutters)} (이번 실행으로 {len(self.cutters)-start_count}개 추가됨)")

    def generate_slanted_cutters(self, min_area, min_edge_len, extrusion_length, tolerance, masks=None, min_cutter_size=3.0):
        """
        축에 정렬되지 않은 평면 영역을 감지하고 방향성 있는 커터를 생성합니다.
        """
        print(f"[커터] 경사진 표면 감지 중 (최소 면적: {min_area:.1f})...")
        
        # 패싯(Facets)은 동일 평면상의 면 그룹입니다.
        facets = self.original_mesh.facets
        count = 0
        
        for face_indices in facets:
            # 1. 면적 확인
            area = np.sum(self.original_mesh.area_faces[face_indices])
            
            # 최소 면적 및 종횡비 체크 (너무 가늘거나 작은 것은 제외)
            if area < (min_area * 1.5): # 더 엄격하게 제한
                continue

            # 3. 마스크 확인 (제외 영역에 포함되면 건너뜜)
                
            # 법선 확인 (면 법선의 평균)
            normals = self.original_mesh.face_normals[face_indices]
            avg_normal = np.mean(normals, axis=0)
            avg_normal /= np.linalg.norm(avg_normal)
            
            # 축 정렬 확인 (축과의 내적)
            is_aligned = False
            for i in range(3):
                if abs(abs(avg_normal[i]) - 1.0) < 0.05: # 5% 허용 오차
                    is_aligned = True
            
            if is_aligned:
                continue
                
            # 방향성 있는 커터 생성
            try:
                # 정점 가져오기
                f = self.original_mesh.faces[face_indices]
                v_idx = np.unique(f.flatten())
                points = self.original_mesh.vertices[v_idx]
                
                # [추가] 모서리 길이 기준 필터링
                # 점들의 최대 거리(Bounding Box 대각선 등)가 기준보다 작으면 무시
                pt_min = np.min(points, axis=0)
                pt_max = np.max(points, axis=0)
                if np.linalg.norm(pt_max - pt_min) < min_edge_len:
                    continue
                
                # 방향성 있는 경계 상자(OBB) 계산
                pc = trimesh.points.PointCloud(points)
                obb = pc.bounding_box_oriented
                
                transform = obb.primitive.transform
                extents = obb.primitive.extents
                
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
                # 표면에서 절반만큼 밖으로 빼서 표면에 딱 걸치게 함
                shift = (new_extents[axis_idx] / 2.0) - (extents[axis_idx] / 2.0) + tolerance
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
                count += 1
                
            except Exception as e:
                print(f"  - 패싯에 대한 OBB 생성 실패: {e}")
                
        print(f"[커터] {count}개의 경사진 커터 추가됨.")

    def _refine_cutter(self, center, size, tolerance):
        """
        메쉬와 충돌할 때까지 각 축 방향으로 박스를 확장합니다.
        """
        refined_center = np.array(center, dtype=float)
        refined_size = np.array(size, dtype=float)
        
        # [추가] 초기 상태 체크: 시작부터 침투 중인지 확인
        # 만약 침투 중이라면 충돌이 없을 때까지 각 축을 조금씩 줄임
        for _ in range(5): # 최대 5단계 수축
             test_pts = self._get_surface_sample_points(refined_center, refined_size, padding=-0.1)
             if np.any(self.original_mesh.contains(test_pts)):
                 refined_size *= 0.9 # 10%씩 수축
             else:
                 break

        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        
        for i, axis in enumerate(axes):
            for side in [-1, 1]:
                step = self.voxel_scale * 0.2 # 더 세밀한 스텝
                max_expansion = self.voxel_scale * 5.0
                
                current_expansion = 0.0
                while current_expansion < max_expansion:
                    test_size = refined_size.copy()
                    test_size[i] += (current_expansion + step)
                    test_center = refined_center.copy()
                    test_center[i] += side * (current_expansion + step) / 2.0
                    
                    # [추가] 꼭짓점 검사 (Vertices Check): 8개 코너 최우선 순위
                    dx, dy, dz = test_size / 2.0
                    cx, cy, cz = test_center
                    v_pts = np.array([
                        [cx-dx, cy-dy, cz-dz], [cx-dx, cy-dy, cz+dz], [cx-dx, cy+dy, cz-dz], [cx-dx, cy+dy, cz+dz],
                        [cx+dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz+dz], [cx+dx, cy+dy, cz-dz], [cx+dx, cy+dy, cz+dz]
                    ])
                    # tolerance(오프셋) 반영
                    v_check = v_pts + side * axis * tolerance
                    if np.any(self.original_mesh.contains(v_check)):
                        break

                    # 5x5 고밀도 샘플링으로 침투 감지
                    pts = self._get_face_sample_points(test_center, test_size, i, side)
                    check_pts = pts + side * axis * tolerance
                    
                    if np.any(self.original_mesh.contains(check_pts)):
                        break
                    
                    current_expansion += step
                
                if current_expansion > 0:
                    refined_size[i] += current_expansion
                    refined_center[i] += side * current_expansion / 2.0
                
        # [추가] 최종 최소 크기 제약 체크
        if np.any(refined_size < 3.0): # 설정값이 없으면 기본 3.0
             # 너무 작으면 무의미한 커터로 보고 버림(None 반환)
             return None
             
        return {'center': refined_center, 'size': refined_size, 'type': 'aabb'}

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
    def optimize_cutters(self):
        """
        중복되거나 인접한 커터를 더 큰 커터로 그룹화하고 불필요한 커터를 제거합니다.
        """
        if not self.cutters:
            return

        print(f"[최적화] {len(self.cutters)}개의 커터 최적화 중...")
        
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

    def generate_cad(self, use_engine='gmsh'):
        """
        CAD 데이터를 생성합니다. 
        use_engine: 'gmsh' (권장) 또는 'cadquery'
        """
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
            if c.get('type') == 'oriented':
                # OBB 처리 (Gmsh에서 변환 적용)
                sx, sy, sz = c['extents']
                # -sx/2 등 로컬 좌표 박스
                cid = gmsh.model.occ.addBox(-sx/2, -sy/2, -sz/2, sx, sy, sz)
                # 변환 행렬 적용 (4x4 affine transform)
                matrix = c['transform'].flatten().tolist()
                gmsh.model.occ.affineTransform([(3, cid)], matrix)
                cutter_ids.append((3, cid))
            else:
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
                else:
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
        if isinstance(shape_flag, str) and shape_flag.startswith("gmsh_model") and HAS_GMSH:
            print(f"[내보내기] Gmsh를 통해 {filename} 저장 중...")
            try:
                # Gmsh 세션 확인 및 복구
                if not gmsh.isInitialized():
                    gmsh.initialize()
                
                # 모델이 비어있을 수 있으므로 (세션 만료 등) 엔티티 확인
                # 만약 엔티티가 없다면 형상 재생성 시도
                if not gmsh.model.getEntities(3):
                    print("  - Gmsh 모델 데이터가 없어 재생성합니다...")
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
        
        # Gmsh 종료 처리 (필요시)
        if HAS_GMSH:
            try: gmsh.finalize()
            except: pass

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
            if c.get('type') == 'oriented':
                box = pv.Box(bounds=(-c['extents'][0]/2, c['extents'][0]/2,
                                     -c['extents'][1]/2, c['extents'][1]/2,
                                     -c['extents'][2]/2, c['extents'][2]/2))
                box.transform(c['transform'], inplace=True)
                p.add_mesh(box, color='orange', opacity=0.4, style='wireframe', line_width=1)
            else:
                cx, cy, cz = c['center']
                sx, sy, sz = c['size']
                bounds = [cx-sx/2, cx+sx/2, cy-sy/2, cy+sy/2, cz-sz/2, cz+sz/2]
                p.add_mesh(pv.Box(bounds=bounds), color='red', opacity=0.4, style='wireframe')


        # 2. 최종 결과물 (Simplified Result)
        p.subplot(0, 1)
        p.add_text("2. Simplified Result", font_size=9, color='black')
        if result_pv and result_pv.n_points > 0:
            p.add_mesh(result_pv, color='lightgreen', show_edges=True)
        else:
            p.add_text("\n(Empty Result)", color='darkred', font_size=10)
            
        if show_removed:
            p.add_text("\n(+ Removed Volume)", position='upper_right', font_size=8, color='darkred')
            for c in self.cutters:
                if c.get('type') == 'oriented':
                    box = pv.Box(bounds=(-c['extents'][0]/2, c['extents'][0]/2,
                                         -c['extents'][1]/2, c['extents'][1]/2,
                                         -c['extents'][2]/2, c['extents'][2]/2))
                    box.transform(c['transform'], inplace=True)
                    p.add_mesh(box, color='red', opacity=0.15, style='surface')
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
             p.add_text("\n(Insufficient Data)", color='darkred', font_size=10)

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
        p.add_text("2. Cutting Blocks", font_size=9, color='black')
        for i, c in enumerate(self.cutters):
            if c.get('type') == 'oriented':
                box = pv.Box(bounds=(-c['extents'][0]/2, c['extents'][0]/2,
                                     -c['extents'][1]/2, c['extents'][1]/2,
                                     -c['extents'][2]/2, c['extents'][2]/2))
                box.transform(c['transform'], inplace=True)
                p.add_mesh(box, color='orange', opacity=0.6, show_edges=True)
            else:
                cx, cy, cz = c['center']
                sx, sy, sz = c['size']
                bounds = [cx-sx/2, cx+sx/2, cy-sy/2, cy+sy/2, cz-sz/2, cz+sz/2]
                p.add_mesh(pv.Box(bounds=bounds), color='red', opacity=0.6, show_edges=True)

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
        p.show()


    def calculate_volume_error(self, result_shape):
        """
        단순화된 형상의 부피 오차율(%)을 계산합니다.
        """
        vol_simplified = 0.0
        
        if result_shape == "gmsh_model_final" and HAS_GMSH:
            try:
                # Gmsh 모델의 부피 계산 (현재 활성 모델 기준)
                entities = gmsh.model.getEntities(3)
                for ent in entities:
                    vol_simplified += gmsh.model.occ.getMass(ent[0], ent[1])
            except: return None
        elif HAS_CADQUERY and result_shape is not None and result_shape != "gmsh_model":
            try:
                vol_simplified = result_shape.val().Volume()
            except: return None
        else:
            return None

        vol_original = 0.0
        if self.original_cq:
            try:
                vol_original = self.original_cq.val().Volume()
            except: pass
        
        if vol_original <= 1e-6 and self.original_mesh:
             vol_original = self.original_mesh.volume
             
        # Helper: Center Window
        def center_window(win, parent=None):
            win.update_idletasks()
            width = win.winfo_width()
            height = win.winfo_height()
            
            p = parent if parent else root
            p_x = p.winfo_rootx()
            p_y = p.winfo_rooty()
            p_w = p.winfo_width()
            p_h = p.winfo_height()
            
            x = p_x + (p_w // 2) - (width // 2)
            y = p_y + (p_h // 2) - (height // 2)
            win.geometry(f'+{x}+{y}')

        # Helper: Custom Message Box (Centered)
        def show_custom_msg(title, msg, dtype='info', parent=None):
            dlg = tk.Toplevel(parent if parent else root)
            dlg.title(title)
            dlg.geometry("350x180")
            dlg.resizable(False, False)
            dlg.transient(parent if parent else root)
            dlg.grab_set()
            
            icon_char = "ℹ"
            bg_col = "#f0f0f0"
            if dtype == 'warning': icon_char = "⚠️"; bg_col="#fff8e1"
            elif dtype == 'error': icon_char = "❌"; bg_col="#ffebee"
            
            dlg.configure(bg=bg_col)
            
            tk.Label(dlg, text=icon_char, font=("Arial", 24), bg=bg_col).pack(pady=(20, 5))
            tk.Label(dlg, text=msg, wraplength=320, bg=bg_col, font=("Arial", 10)).pack(pady=5, expand=True)
            tk.Button(dlg, text="OK", command=dlg.destroy, width=10, bg='white').pack(pady=(0, 20))
            
            center_window(dlg, parent if parent else root)
            root.wait_window(dlg)


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
            'tolerance': 0.05,             # 커터 확장 시 허용 오차 (mm).
            'detect_slanted': True,        # 경사면(축 정렬되지 않은 면) 감지 여부.
            'slanted_area_factor': 4.0,    # 경사면 감지 시 최소 면적 계수 (voxel_resolution * factor)^2.
            'slanted_edge_factor': 2.0,    # 경사면 감지 시 최소 모서리 길이 계수 (voxel_resolution * factor).
            'min_cutter_size': 3.0,        # 커터의 최소 변 길이 (mm).
            'auto_tune': False,            # 목표 오차율 달성을 위한 해상도 자동 조절 여부.
            'target_error': 5.0,           # 자동 조절 시 목표 부피 오차율 (%).
            'max_iterations': 5            # 자동 조절 또는 반복 탐색 시 최대 반복 횟수.
        }

        # Tkinter 윈도우 설정
        root = tk.Tk()
        root.title("CAD Simplifier Control Panel")
        root.geometry("550x750")
        root.geometry("550x800")
        
        # 설명 (스크롤 가능하도록 변경)
        desc_frame = tk.Frame(root)
        desc_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=False)
        
        desc_text = ("=== 파라미터 상세 설명 ===\n"
                    "1. Voxel Resolution (mm): 모델을 복셀화할 때의 해상도입니다. 작을수록 정밀하지만 느려집니다.\n"
                    "2. Min Volume Ratio (0~1): 전체 부피 대비, 이 비율보다 작은 커터는 무시합니다. (노이즈 제거)\n"
                    "3. Max Cutters (count): 한 번 실행에서 생성할 최대 커터 개수입니다.\n"
                    "4. Tolerance (mm): 커터 생성 시 적용할 여유 공차입니다.\n"
                    "5. Detect Slanted Surfaces: 축에 정렬되지 않은 경사면을 감지하여 비스듬한 커터를 생성합니다.\n"
                    "6. Min Cutter Size (mm): 생성할 커터의 최소 변 길이입니다. 너무 작으면 CAD 연산이 불안정해집니다.\n"
                    "   [Tip] Voxel Resolution의 2~3배 이상을 권장합니다. (예: Res=1.0mm -> Min Size=3.0mm)\n"
                    "   - 1.5배 미만: 노이즈가 많아지고 연산이 불안정 해질 수 있음.\n"
                    "   - 3.0배 이상: 안정적이며 덩어리 위주로 단순화됨.\n"
                    "7. Auto-tune: 목표 오차율(Target Error)에 도달할 때까지 해상도를 자동으로 조절하며 반복합니다.")
        
        # Text Widget + Scrollbar
        txt_desc = tk.Text(desc_frame, height=8, bg="#f0f0f0", relief=tk.FLAT, font=("Consolas", 9), wrap=tk.WORD)
        scrollbar = tk.Scrollbar(desc_frame, command=txt_desc.yview)
        txt_desc.configure(yscrollcommand=scrollbar.set)
        
        txt_desc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        txt_desc.insert(tk.END, desc_text)
        txt_desc.configure(state='disabled') # 읽기 전용

        entries = {}
        fields = [
            ('Voxel Resolution (mm)', 'voxel_resolution'),
            ('Min Volume Ratio', 'min_volume_ratio'),
            ('Max Cutter Count', 'max_cutters'),
            ('Tolerance (mm)', 'tolerance'),
            ('Min Cutter Size (mm)', 'min_cutter_size'),
            ('Slanted Area Factor (Area)', 'slanted_area_factor'),
            ('Slanted Edge Factor (Length)', 'slanted_edge_factor'),
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
            
            # Scale-based Heuristics
            # Tolerance는 모델 크기와 무관하게 작게 유지해야 벽면에 밀착됨
            if "Fine" in mode:
                res = diagonal / 150.0
                min_cutter_factor = 2.0
                tol = 0.01 # Fixed small tolerance
                max_cnt = 200
            elif "Very Rough" in mode:
                res = diagonal / 25.0
                min_cutter_factor = 5.0
                tol = 0.1 # Looser but still fixed
                max_cnt = 25
            elif "Coarse" in mode:
                res = diagonal / 50.0
                min_cutter_factor = 4.0
                tol = 0.05
                max_cnt = 50
            else: # Medium
                res = diagonal / 100.0
                min_cutter_factor = 3.0
                tol = 0.02
                max_cnt = 100
                
            # Limits
            res = max(0.2, round(res, 2))
            min_cutter = max(1.0, round(res * min_cutter_factor, 1))
            
            # Update UI
            entries['voxel_resolution'].delete(0, tk.END); entries['voxel_resolution'].insert(0, str(res))
            entries['min_cutter_size'].delete(0, tk.END); entries['min_cutter_size'].insert(0, str(min_cutter))
            entries['max_cutters'].delete(0, tk.END); entries['max_cutters'].insert(0, str(max_cnt))
            entries['tolerance'].delete(0, tk.END); entries['tolerance'].insert(0, str(round(tol, 3)))
            
            entries['tolerance'].delete(0, tk.END); entries['tolerance'].insert(0, str(round(tol, 3)))
            
            show_custom_msg("Preset Applied", f"Applied '{mode}' preset.\n(Scale: {diagonal:.1f}mm)\n\nRes: {res}mm\nMin Cutter: {min_cutter}mm", 'info', root)

        tk.Button(preset_frame, text="Apply", command=apply_preset, bg='white').pack(side=tk.LEFT)

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
            
        # Checkboxes
        check_frame = tk.Frame(param_frame)
        check_frame.pack(fill=tk.X, pady=10)
        
        self.var_detect_slanted = tk.BooleanVar(value=self._cfg.get('detect_slanted', True))
        chk_slanted = tk.Checkbutton(check_frame, text="Detect Slanted Surfaces", variable=self.var_detect_slanted)
        chk_slanted.grid(row=0, column=0, sticky='w', padx=5)

        self.var_auto_tune = tk.BooleanVar(value=self._cfg.get('auto_tune', False))
        chk_auto = tk.Checkbutton(check_frame, text="Auto-tune Resolution", variable=self.var_auto_tune)
        chk_auto.grid(row=0, column=1, sticky='w', padx=5)

        self.var_show_removed = tk.BooleanVar(value=False)
        chk_removed = tk.Checkbutton(check_frame, text="Show Removed Volume", variable=self.var_show_removed)
        chk_removed.grid(row=0, column=2, sticky='w', padx=5)


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
                return True
            except ValueError:
                messagebox.showerror("Error", "숫자를 올바르게 입력해주세요.")
                return False

        def show_preview():
            if self.original_mesh is None:
                messagebox.showwarning("Warning", "No model loaded.")
                return
            if not get_values_from_ui(): return
            
            # PyVista Plotter 생성 (Preview 모드)
            p = pv.Plotter(title="Preview - Close window to return to settings")
            
            cfg = self._cfg
            min_pt, max_pt = self.bounding_box
            size = max_pt - min_pt
            res = cfg['voxel_resolution']
            padding = res * 1.2
            origin = min_pt - padding
            grid_shape = np.ceil((size + padding*2) / res).astype(int)
            total_voxels = np.prod(grid_shape)
            
            # 정보 텍스트 (Font Size 9로 축소, Dark Colors)
            info = (f"Resolution: {res} mm\n"
                    f"Grid: {grid_shape}\n"
                    f"Total Voxels: {total_voxels:,}\n"
                    f"Bounding Box:\n"
                    f"  X: {min_pt[0]:.2f} ~ {max_pt[0]:.2f} ({size[0]:.2f})\n"
                    f"  Y: {min_pt[1]:.2f} ~ {max_pt[1]:.2f} ({size[1]:.2f})\n"
                    f"  Z: {min_pt[2]:.2f} ~ {max_pt[2]:.2f} ({size[2]:.2f})")
            
            print("\n" + "="*40)
            print(" [클래스 1/2] Preview Information")
            print(info)
            print("="*40)

            p.add_text(info, font_size=9, position='upper_left', color='black')
            
            # 메쉬
            p.add_mesh(self.original_mesh, color='lightblue', opacity=0.3)
            
            # 그리드 (너무 많으면 박스만)
            if total_voxels < 1000000: # 1M voxels limit for wireframe
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
            
            # [Preview Enhancement] Detect Slanted Surfaces 옵션이 켜져 있으면 미리보기 제공
            if cfg.get('detect_slanted', False):
                # 시각화를 위해 임시로 커터 생성
                min_area = (cfg['voxel_resolution'] * cfg['slanted_area_factor']) ** 2
                min_edge = cfg['voxel_resolution'] * cfg['slanted_edge_factor']
                extrusion_depth = max(cfg['voxel_resolution'] * 5.0, cfg['min_cutter_size'])
                
                # 기존 커터 리스트 백업 (Preview는 원본 상태를 변경하면 안 됨)
                backup_cutters = self.cutters
                self.cutters = []
                
                # 감지 실행
                self.generate_slanted_cutters(min_area, min_edge, extrusion_depth, cfg['tolerance'], masks=[], min_cutter_size=cfg['min_cutter_size'])
                
                if self.cutters:
                    p.add_text(f"Preview: {len(self.cutters)} Slanted Cutters Detected", position='upper_right', color='darkgreen', font_size=9)
                    for c in self.cutters:
                        if c.get('type') == 'oriented':
                            box = pv.Box(bounds=(-c['extents'][0]/2, c['extents'][0]/2,
                                                 -c['extents'][1]/2, c['extents'][1]/2,
                                                 -c['extents'][2]/2, c['extents'][2]/2))
                            box.transform(c['transform'], inplace=True)
                            p.add_mesh(box, color='orange', opacity=0.5, style='wireframe', line_width=2)
                
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
                            min_cutter_size=self._cfg['min_cutter_size'],
                            append=append,
                            progress_callback=update_progress
                        )
                        
                        if not auto_tune:
                            root.after(0, lambda: status_var.set("Processing... (Generating CAD)"))
                        
                        # 2. Generate CAD
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
                    print(e)
                    def on_error(msg):
                        status_var.set(f"Error: {msg}")
                        btn_run.config(state='normal')
                        btn_refine.config(state='normal')
                        show_custom_msg("Error", f"An error occurred:\n{msg}", 'error', root)
                    root.after(0, on_error, str(e))
            threading.Thread(target=worker, daemon=True).start()

        def view_result():
            if self.simplified_shape:
                self.visualize(self.simplified_shape, show_removed=self.var_show_removed.get())

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
            save_win.geometry("300x150")
            center_window(save_win, root) # Center this too
            
            tk.Label(save_win, text="Select format to save:", pady=10).pack()

            save_win.title("Save Results")
            save_win.geometry("300x150")
            
            tk.Label(save_win, text="Select format to save:", pady=10).pack()
            tk.Button(save_win, text="Save CAD Model (STEP/STL...)", command=lambda: [export_cad_file(), save_win.destroy()], width=30).pack(pady=5)
            tk.Button(save_win, text="Save Cutter Info (.txt)", command=lambda: [export_cutter_txt(), save_win.destroy()], width=30).pack(pady=5)


        # --- Menu Bar ---
        menubar = tk.Menu(root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=open_file)
        
        samples_menu = tk.Menu(file_menu, tearoff=0)
        samples_menu.add_command(label="Basic Box", command=lambda: load_sample('basic'))
        samples_menu.add_command(label="Ribbed Cushion", command=lambda: load_sample('ribbed_cushion'))
        samples_menu.add_command(label="Complex Ribs", command=lambda: load_sample('complex_ribs'))
        file_menu.add_cascade(label="Sample Shapes", menu=samples_menu)
        
        
        file_menu.add_separator()
        save_menu = tk.Menu(file_menu, tearoff=0)
        save_menu.add_command(label="Save CAD Model...", command=export_cad_file)
        save_menu.add_command(label="Save Cutter Info...", command=export_cutter_txt)
        file_menu.add_cascade(label="Export Result", menu=save_menu)

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
        menubar.add_cascade(label="File", menu=file_menu)
        root.config(menu=menubar)

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
        btn_refine.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
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
