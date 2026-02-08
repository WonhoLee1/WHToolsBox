import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import copy
import csv
import tkinter as tk
from tkinter import ttk, scrolledtext
from dataclasses import dataclass

try:
    from box_mesh_generator import BoxMeshByGmsh
    HAS_GMSH_GEN = True
except ImportError:
    HAS_GMSH_GEN = False
    print("Warning: box_mesh_generator not found or Gmsh not installed.")

# 선택적 PyVista 임포트
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

class BoxGeometry:
    """
    박스의 기하학적 정보를 정의하는 클래스.
    기본적으로 박스의 중심은 (0,0,0)에 위치하며 축 정렬되어 있다고 가정합니다.
    """
    def __init__(self, width, depth, height):
        """
        BoxGeometry 초기화.

        Args:
            width (float): 박스의 가로 길이 (x축 방향)
            depth (float): 박스의 세로 길이 (y축 방향)
            height (float): 박스의 높이 (z축 방향)
        """
        self.width = width   # x축 크기
        self.depth = depth   # y축 크기
        self.height = height # z축 크기
        
        # 반폭, 반깊이, 반높이
        self.dx = width / 2.0
        self.dy = depth / 2.0
        self.dz = height / 2.0

        # 면 정의 (Normal vector, Distance from origin)
        # ax + by + cz = d 형태의 평면 방정식에서 (a,b,c)는 법선, d는 원점 거리
        # 각 면은 고유의 키('F', 'B', 'L', 'R', 'T', 'D')를 가짐
        self.faces = {
            'F': {'normal': np.array([0, -1, 0]), 'dist': self.dy, 'axis_idx': 1, 'val': -self.dy}, # Front (-Y)
            'B': {'normal': np.array([0, 1, 0]),  'dist': self.dy, 'axis_idx': 1, 'val': self.dy},  # Back (+Y)
            'L': {'normal': np.array([-1, 0, 0]), 'dist': self.dx, 'axis_idx': 0, 'val': -self.dx}, # Left (-X)
            'R': {'normal': np.array([1, 0, 0]),  'dist': self.dx, 'axis_idx': 0, 'val': self.dx},  # Right (+X)
            'T': {'normal': np.array([0, 0, 1]),  'dist': self.dz, 'axis_idx': 2, 'val': self.dz},  # Top (+Z)
            'D': {'normal': np.array([0, 0, -1]), 'dist': self.dz, 'axis_idx': 2, 'val': -self.dz}, # Down (-Z)
        }

        # 코너 정의 (8개)
        # 이름 규칙: C_{Top/Down}{Front/Back}{Left/Right} -> 예: C_TFL
        # 각 코너의 로컬 좌표를 계산하여 저장
        self.corners = {}
        for z_key, z_val in [('T', self.dz), ('D', -self.dz)]:
            for y_key, y_val in [('B', self.dy), ('F', -self.dy)]: # 참고: 보통 스크린 좌표계에서 Back은 +Y, Front는 -Y이나, 여기서는 오른손 좌표계 표준을 따름
                for x_key, x_val in [('R', self.dx), ('L', -self.dx)]:
                    name = f"C_{z_key}{y_key}{x_key}"
                    self.corners[name] = np.array([x_val, y_val, z_val])

    def get_vertices(self):
        """
        박스의 8개 꼭짓점 좌표를 배열로 반환합니다. (시각화 및 계산용)
        
        Returns:
            np.array: (8, 3) 크기의 꼭짓점 좌표 배열
        """
        return np.array(list(self.corners.values()))

    def get_ista_face_name(self, face_key):
        """ISTA 6A Face Mapping (Amazon)"""
        mapping = {'T': '1 (Top)', 'D': '2 (Bottom)', 'F': '3 (Front)', 'B': '4 (Back)', 'R': '5 (Right)', 'L': '6 (Left)'}
        return mapping.get(face_key, face_key)

class BoxMotionEstimator:
    """
    주어진 3D 점들을 이용하여 박스의 최적 위치(회전, 병진)를 추정하는 클래스.
    비선형 최소자승법(Non-linear Least Squares)을 사용하여 관측된 점들과 박스 모델 간의 오차를 최소화합니다.
    """
    def __init__(self, geometry: BoxGeometry, outlier_threshold=50.0, track_deformation=False, estimate_dims=False):
        """
        BoxMotionEstimator 초기화.

        Args:
            geometry (BoxGeometry): 박스의 기하학적 정보 객체
            outlier_threshold (float): 이상치(Outlier)로 판단할 거리 임계값 (mm 단위). 기본값 50.0mm.
            track_deformation (bool): 지면 침투(변형) 추적 기능 활성화 여부.
            estimate_dims (bool): 박스의 크기(가로, 세로, 높이)도 함께 추정할지 여부.
        """
        self.geo = geometry
        self.outlier_threshold = outlier_threshold
        self.track_deformation = track_deformation
        self.estimate_dims = estimate_dims

        # 상태 변수: [tx, ty, tz, rx, ry, rz] (rx,ry,rz는 회전 벡터)
        self.current_pose = np.zeros(6) 
        self.velocity_linear = np.zeros(3)  # 선속도 벡터 [vx, vy, vz]
        self.velocity_angular = np.zeros(3) # 각속도 벡터 [wx, wy, wz]
        self.last_time = None               # 마지막 프레임 시간
        self.history = []                   # 시간별 상태 저장 리스트 [{'t', 'pose', 'v_lin', 'v_ang', 'a_lin', 'a_ang'}, ...]
        self.corner_cuts = {name: {'x': 0.0, 'y': 0.0, 'z': 0.0} for name in geometry.corners} # 코너별 3축 절단 길이 (변형량)

    def _transform_points_inverse(self, points, pose):
        """
        월드 좌표계의 점들을 박스 로컬 좌표계로 변환 (Inverse Transform).
        T_box_to_world의 역변환을 적용.
        
        Args:
            points (np.array): 월드 좌표계 점들 (N, 3)
            pose (np.array): 박스 포즈 [tx, ty, tz, rx, ry, rz]
            
        Returns:
            np.array: 로컬 좌표계로 변환된 점들 (N, 3)
        """
        t = pose[:3]
        r_vec = pose[3:]
        rot = R.from_rotvec(r_vec)
        
        # P_local = R^T * (P_world - T)
        return rot.inv().apply(points - t)

    def _transform_points_forward(self, points_local, pose):
        """
        박스 로컬 좌표계의 점들을 월드 좌표계로 변환 (Forward Transform).
        
        Args:
            points_local (np.array): 로컬 좌표계 점들 (N, 3)
            pose (np.array): 박스 포즈 [tx, ty, tz, rx, ry, rz]
            
        Returns:
            np.array: 월드 좌표계로 변환된 점들 (N, 3)
        """
        t = pose[:3]
        r_vec = pose[3:]
        rot = R.from_rotvec(r_vec)
        return rot.apply(points_local) + t

    def _cost_function(self, params, points_data):
        """
        최적화 목적 함수 (Residual 계산).
        관측된 점들과 박스 모델 사이의 거리를 계산하여 반환합니다.
        
        Args:
            params (np.array): 최적화 변수. [tx, ty, tz, rx, ry, rz] 또는 [..., w, d, h]
            points_data (list): 점 데이터 리스트 [{'coord': np.array, 'type': str, 'id': str}, ...]
            
        Returns:
            np.array: 각 점에 대한 잔차(Residual) 배열
        """
        residuals = []
        
        # 1. 모든 점을 현재 추정된 박스 로컬 좌표계로 변환
        if self.estimate_dims:
            pose_params = params[:6]
            dims = params[6:]
            # 반복을 위한 임시 기하학 객체 생성
            temp_geo = BoxGeometry(dims[0], dims[1], dims[2])
        else:
            pose_params = params
            temp_geo = self.geo

        coords_world = np.array([p['coord'] for p in points_data])
        coords_local = self._transform_points_inverse(coords_world, pose_params)
        
        for i, p_info in enumerate(points_data):
            pt_local = coords_local[i]
            p_type = p_info['type']
            
            if p_type == 'N': # Unknown surface
                # 가장 가까운 면까지의 거리
                dists = []
                dists.append(np.abs(pt_local[0] - temp_geo.dx)) # R
                dists.append(np.abs(pt_local[0] + temp_geo.dx)) # L
                dists.append(np.abs(pt_local[1] - temp_geo.dy)) # B
                dists.append(np.abs(pt_local[1] + temp_geo.dy)) # F
                dists.append(np.abs(pt_local[2] - temp_geo.dz)) # T
                dists.append(np.abs(pt_local[2] + temp_geo.dz)) # D
                residuals.append(min(dists))
                
            elif p_type in self.geo.faces: # Specific Face (F, B, L, R, T, D)
                # Use the face definition from the temporary geometry
                face = temp_geo.faces[p_type]
                # 해당 면의 평면까지의 거리: |coord[axis] - val|
                dist = pt_local[face['axis_idx']] - face['val']
                residuals.append(dist) # 부호 있는 거리는 솔버가 방향을 파악하도록 함
                
            elif p_type.startswith('C_'): # Specific Corner
                if p_type in temp_geo.corners:
                    target = temp_geo.corners[p_type]
                    diff = pt_local - target
                    residuals.extend(diff) # 3D distance vector
                else:
                    # 정의되지 않은 코너 이름이면 무시하거나 N처럼 처리
                    pass
            
            elif p_type == 'C': # Unknown Corner
                # 가장 가까운 코너 찾기
                min_d = 1e9
                for c_pos in temp_geo.corners.values():
                    d = np.linalg.norm(pt_local - c_pos)
                    if d < min_d: min_d = d
                residuals.append(min_d)

        return np.array(residuals).flatten()

    def fit(self, points_input, time_stamp=None, continuous=True):
        """
        점 데이터를 기반으로 박스 위치 추정.
        2단계 최적화(Robust -> Precise)를 통해 이상치를 제거하고 정밀한 포즈를 계산합니다.
        
        Args:
            points_input (list): 입력 점 데이터 리스트 [{'id': '#1', 'type': 'F', 'coord': [x,y,z]}, ...]
            time_stamp (float, optional): 현재 프레임의 시간 (초). 속도 및 가속도 계산에 사용됨.
            continuous (bool): True일 경우 이전 프레임의 추정 결과를 초기값으로 사용하여 연속성을 보장함.
        
        Returns:
            tuple: (pose, valid_points, rejected_points)
                - pose: 추정된 포즈 [tx, ty, tz, rx, ry, rz]
                - valid_points: 추정에 사용된 유효한 점 리스트
                - rejected_points: 이상치로 판단되어 제외된 점 리스트
        """
        # 초기값 설정
        if self.estimate_dims:
            # [tx, ty, tz, rx, ry, rz, w, d, h]
            pose_guess = self.current_pose if continuous and np.any(self.current_pose) else np.zeros(6)
            dims_guess = np.array([self.geo.width, self.geo.depth, self.geo.height])
            x0 = np.concatenate([pose_guess, dims_guess])
        else:
            if continuous and np.any(self.current_pose):
                x0 = self.current_pose
            else:
                # 간단한 초기화: 점들의 중심을 박스 중심으로 가정
                coords = np.array([p['coord'] for p in points_input])
                center = np.mean(coords, axis=0)
                x0 = np.array([center[0], center[1], center[2], 0, 0, 0])


        # 1차 최적화
        res = least_squares(self._cost_function, x0, args=(points_input,), loss='soft_l1')
        pose_est = res.x
        
        # Outlier 제거 (Smart Rejection)
        valid_points = []
        rejected_points = []
        
        # 최종 Residual 계산
        if self.estimate_dims:
            final_pose = pose_est[:6]
            final_dims = pose_est[6:]
            final_geo = BoxGeometry(final_dims[0], final_dims[1], final_dims[2])
        else:
            final_pose = pose_est
            final_geo = self.geo

        coords_world = np.array([p['coord'] for p in points_input])
        coords_local = self._transform_points_inverse(coords_world, final_pose)
        
        for i, p in enumerate(points_input):
            pt_local = coords_local[i]
            error = 0
            reason = ""
            
            if p['type'] == 'N':
                # 가장 가까운 면과의 거리
                dists = [np.abs(pt_local[0]-final_geo.dx), np.abs(pt_local[0]+final_geo.dx),
                         np.abs(pt_local[1]-final_geo.dy), np.abs(pt_local[1]+final_geo.dy),
                         np.abs(pt_local[2]-final_geo.dz), np.abs(pt_local[2]+final_geo.dz)]
                error = min(dists)
            elif p['type'] in final_geo.faces:
                face = final_geo.faces[p['type']]
                error = np.abs(pt_local[face['axis_idx']] - face['val'])
            elif p['type'].startswith('C_'):
                if p['type'] in final_geo.corners:
                    error = np.linalg.norm(pt_local - final_geo.corners[p['type']])
            
            if error > self.outlier_threshold:
                p_copy = p.copy()
                p_copy['reject_reason'] = f"Error {error:.4f} > {self.outlier_threshold}"
                rejected_points.append(p_copy)
            else:
                valid_points.append(p)

        # 2차 최적화 (Outlier 제거 후 정밀 피팅)
        if len(valid_points) >= 3: # 최소 3점 필요
            res_final = least_squares(self._cost_function, pose_est, args=(valid_points,), loss='linear')
            final_params = res_final.x
        else:
            print("Warning: Not enough valid points for fitting.")
            final_params = pose_est

        if self.estimate_dims:
            self.current_pose = final_params[:6]
            # 추정된 치수로 기하학 정보 업데이트
            self.geo = BoxGeometry(final_params[6], final_params[7], final_params[8])
        else:
            self.current_pose = final_params

        # 속도 계산 (Kinematics)
        accel_linear = np.zeros(3)
        accel_angular = np.zeros(3)
        
        if time_stamp is not None and self.last_time is not None:
            dt = time_stamp - self.last_time
            if dt > 0:
                if self.estimate_dims:
                    prev_pose = x0[:6]
                else:
                    prev_pose = x0
                # Linear Velocity
                new_v_lin = (self.current_pose[:3] - prev_pose[:3]) / dt
                
                # 각속도 (회전 벡터 차이로부터 근사)
                # R_diff = R_curr * R_prev^T
                r_curr = R.from_rotvec(self.current_pose[3:])
                r_prev = R.from_rotvec(x0[3:])
                r_diff = r_curr * r_prev.inv()
                rot_vec_diff = r_diff.as_rotvec()
                new_v_ang = rot_vec_diff / dt # 이는 근사값임
                
                # 가속도 (유한 차분)
                # 이전 프레임의 속도가 history에 있다면 사용
                if self.history:
                    last_state = self.history[-1]
                    accel_linear = (new_v_lin - last_state['v_lin']) / dt
                    accel_angular = (new_v_ang - last_state['v_ang']) / dt
                
                self.velocity_linear = new_v_lin
                self.velocity_angular = new_v_ang
        
        # 이력 저장
        if time_stamp is not None:
            self.history.append({
                't': time_stamp,
                'pose': self.current_pose.copy(),
                'v_lin': self.velocity_linear.copy(),
                'v_ang': self.velocity_angular.copy(),
                'a_lin': accel_linear,
                'a_ang': accel_angular
            })
        
        # 변형(지면 침투) 추적 업데이트
        if self.track_deformation:
            current_defs = self._update_deformation()
            if self.history:
                self.history[-1]['deformations'] = current_defs
        
        self.last_time = time_stamp
        return self.current_pose, valid_points, rejected_points

    def _update_deformation(self):
        """
        현재 포즈에서 코너의 지면(Z=0) 침투 여부를 확인하고 최대 침투량(절단 길이)을 갱신합니다.
        
        Returns:
            dict: 현재 프레임에서의 코너별 절단 길이 {'C_...': max_cut_length}
        """
        t = self.current_pose[:3]
        r_obj = R.from_rotvec(self.current_pose[3:])
        rot_mat = r_obj.as_matrix() # (3,3)
        
        # 현재 프레임의 변형량 (리턴용, 여기서는 각 코너의 최대 절단 길이로 요약)
        current_frame_defs = {}
        
        # 각 축별 최대 길이 (Box Dimensions)
        max_lengths = [self.geo.width, self.geo.depth, self.geo.height]
        
        for name, local_pos in self.geo.corners.items():
            # 코너의 월드 좌표 계산
            c_world = r_obj.apply(local_pos) + t
            
            # 지면(Z=0) 침투 확인 (Z < 0)
            if c_world[2] < 0:
                # 코너가 지면 아래에 있음 -> 연결된 3개의 엣지에 대해 지면과의 교차점 계산
                # 엣지 방향 벡터 (Local Axis가 회전된 것)
                # 코너 이름에서 방향 유추 (BoxGeometry 생성 규칙 기반)
                # C_TFL -> T(Z+), F(Y-), L(X-) 인데, 
                # 박스 중심에서 코너로 가는 벡터의 각 성분 부호가 엣지 방향(코너 -> 중심 반대? 아니면 코너 -> 이웃)
                # 여기서는 "코너에서 이웃 코너로 가는 방향"을 엣지로 정의.
                # 직육면체이므로 로컬 축(X,Y,Z)과 평행함.
                # 코너가 (x,y,z)일 때 이웃은 (-x, y, z), (x, -y, z), (x, y, -z)
                # 즉, 엣지 방향은 로컬 좌표계에서 -sign(corner_local) * axis_vector
                
                signs = np.sign(local_pos) # [sx, sy, sz]
                
                # 3개 축에 대해 반복 (0:X, 1:Y, 2:Z)
                axes_keys = ['x', 'y', 'z']
                for i in range(3):
                    # 엣지 방향 벡터 (월드 좌표계)
                    # 로컬에서 -sign * axis_i
                    edge_dir_local = np.zeros(3)
                    edge_dir_local[i] = -signs[i]
                    edge_dir_world = rot_mat @ edge_dir_local
                    
                    # 교차점 계산: P = C + alpha * dir
                    # P.z = 0 => C.z + alpha * dir.z = 0 => alpha = -C.z / dir.z
                    
                    alpha = 0.0
                    if abs(edge_dir_world[2]) > 1e-6:
                        alpha_calc = -c_world[2] / edge_dir_world[2]
                        if alpha_calc > 0: # 유효한 교차 (코너로부터의 거리)
                            alpha = alpha_calc
                    else:
                        # 엣지가 바닥과 평행하고 코너가 바닥 아래에 있음 -> 전체 엣지가 잠김
                        alpha = max_lengths[i]
                    
                    # 박스 크기를 넘어서는 절단은 물리적으로 전체 면이 잠긴 것과 같으므로 clamp
                    if alpha > max_lengths[i]:
                        alpha = max_lengths[i]

                    # 기존 기록보다 크면 갱신
                    if alpha > self.corner_cuts[name][axes_keys[i]]:
                        self.corner_cuts[name][axes_keys[i]] = alpha
            
            # 리턴용: 해당 코너의 3축 절단 길이 중 최대값 (CSV 기록용)
            max_cut = max(self.corner_cuts[name].values())
            current_frame_defs[name] = max_cut
            
        return current_frame_defs

    def get_corner_kinematics(self, corner_name):
        """
        특정 코너의 시간별 위치, 속도, 가속도 이력을 반환합니다.
        
        Args:
            corner_name (str): 코너 이름 (예: 'C_TFL')
            
        Returns:
            dict: {'t': [], 'pos': [], 'vel': [], 'accel': []} 형태의 이력 데이터. 코너가 없으면 None.
        """
        if corner_name not in self.geo.corners:
            print(f"Corner {corner_name} not found.")
            return None
            
        c_local = self.geo.corners[corner_name]
        data = {'t': [], 'pos': [], 'vel': [], 'accel': []}
        
        for h in self.history:
            t = h['t']
            pose = h['pose']
            v_com = h['v_lin']
            w_com = h['v_ang']
            a_com = h['a_lin']
            alpha_com = h['a_ang']
            
            # 회전
            r_rot = R.from_rotvec(pose[3:])
            
            # 위치: p = t + R*p_local
            r_world = r_rot.apply(c_local)
            pos = pose[:3] + r_world
            
            # 속도: v = v_com + w x r_world
            vel = v_com + np.cross(w_com, r_world)
            
            # 가속도: a = a_com + alpha x r + w x (w x r)
            accel = a_com + np.cross(alpha_com, r_world) + np.cross(w_com, np.cross(w_com, r_world))
            
            data['t'].append(t)
            data['pos'].append(pos)
            data['vel'].append(vel)
            data['accel'].append(accel)
            
        return data

    def get_corner_deformations(self):
        """
        코너별 누적된 최대 절단(변형) 정보를 반환합니다.
        
        Returns:
            dict: {'C_...': {'x': val, 'y': val, 'z': val}} 형태의 딕셔너리
        """
        return self.corner_cuts

    def get_down_direction(self, local=True):
        """
        바닥 방향(중력 방향) 벡터를 반환합니다.
        기본적으로 월드 좌표계의 -Z 방향([0, 0, -1])을 바닥으로 가정합니다.
        
        Args:
            local (bool): True이면 박스 로컬 좌표계 기준의 벡터를 반환합니다.
                          False이면 월드 좌표계 기준([0, 0, -1])을 반환합니다.
                          
        Returns:
            np.array: (3,) 크기의 방향 벡터
        """
        world_down = np.array([0.0, 0.0, -1.0])
        if not local:
            return world_down
        
        # 로컬 Down = R^T * World_Down
        r = R.from_rotvec(self.current_pose[3:])
        return r.inv().apply(world_down)

    def check_ground_penetration(self):
        """
        현재 포즈에서 박스가 바닥(Z=0)을 침투했는지 검사하고, 해결 방법을 안내합니다.
        
        Returns:
            dict: {
                'is_penetrating': bool,
                'max_depth': float,      # 최대 침투 깊이 (이동해야 할 거리)
                'normal_vector': np.array, # 바닥 법선 벡터 (World Frame)
                'resolution_vector': np.array # 침투 해결을 위한 이동 벡터
            }
        """
        t = self.current_pose[:3]
        r = R.from_rotvec(self.current_pose[3:])
        
        # 모든 코너의 월드 Z 좌표 중 최솟값 찾기
        corners_local = self.geo.get_vertices()
        corners_world = r.apply(corners_local) + t
        min_z = np.min(corners_world[:, 2])
        
        is_penetrating = min_z < 0
        max_depth = -min_z if is_penetrating else 0.0
        normal_vector = np.array([0.0, 0.0, 1.0]) # 바닥면 법선 (World +Z)
        resolution_vector = normal_vector * max_depth
        
        if is_penetrating:
            print(f"\n[Ground Penetration Detected]")
            print(f" - Max Penetration Depth: {max_depth:.6f} m ({max_depth*1000:.2f} mm)")
            print(f" - Ground Normal Vector : {normal_vector}")
            print(f" - To resolve, move box by: {resolution_vector} (World Frame)")
        else:
            print(f"\n[No Ground Penetration] Clearance: {min_z:.6f} m")
            
        return {
            'is_penetrating': is_penetrating,
            'max_depth': max_depth,
            'normal_vector': normal_vector,
            'resolution_vector': resolution_vector
        }

    def export_to_csv(self, filename):
        """
        추적된 이력(CoM 위치, 선속도, 각속도, 코너 변형량)을 CSV 파일로 내보냅니다.
        
        Args:
            filename (str): 저장할 CSV 파일 경로
        """
        if not self.history:
            print("No history to export.")
            return

        # 헤더 수집
        headers = ['Time', 'CoM_X', 'CoM_Y', 'CoM_Z', 
                   'V_Lin_X', 'V_Lin_Y', 'V_Lin_Z', 
                   'V_Ang_X', 'V_Ang_Y', 'V_Ang_Z']
        
        # 코너 변형 헤더 추가
        corner_names = sorted(list(self.geo.corners.keys()))
        for c in corner_names:
            headers.append(f"Def_{c}")

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for h in self.history:
                row = [h['t']]
                row.extend(h['pose'][:3])
                row.extend(h['v_lin'])
                row.extend(h['v_ang'])
                
                defs = h.get('deformations', {})
                for c in corner_names:
                    row.append(defs.get(c, 0.0))
                
                writer.writerow(row)
        print(f"History exported to {filename}")

    def get_screw_axis(self):
        """
        순간 회전축(Screw Axis) 및 해당 축에 대한 병진/회전 속도 반환.
        v_screw = omega x r + v_parallel
        
        Returns:
            dict: {
                'axis': 회전축 벡터 (단위 벡터),
                'point': 회전축 상의 한 점,
                'pitch': 스크류 피치 (병진 속도 성분),
                'w_mag': 회전 각속도 크기
            }
        """
        w = self.velocity_angular
        v = self.velocity_linear
        w_norm = np.linalg.norm(w)
        
        if w_norm < 1e-6:
            return {'axis': np.array([0,0,1]), 'point': np.array([0,0,0]), 'pitch': 0, 'w_mag': 0}
        
        axis = w / w_norm
        # 피치 (h) = w . v / |w|^2
        pitch = np.dot(w, v) / (w_norm**2)
        
        # 축 위의 점 (r) = (w x v) / |w|^2
        r = np.cross(w, v) / (w_norm**2)
        
        return {
            'axis': axis,
            'point': r,
            'pitch': pitch,
            'w_mag': w_norm
        }

@dataclass
class IstaTestConfig:
    mass: float  # kg
    width: float # m
    depth: float # m
    height: float # m

class IstaFaceMapper:

    """
    ISTA Face Numbering Mapper.
    Dynamically maps ISTA Face Numbers (1~6) to Box Geometry Faces (T, D, F, B, R, L)
    depending on the test type (Parcel vs LTL) and dimensions.
    """
    def __init__(self, is_ltl, geo: BoxGeometry):
        self.is_ltl = is_ltl
        self.geo = geo
        self.mapping = self._build_mapping()
        self.rev_mapping = {v: k for k, v in self.mapping.items()}

    def _build_mapping(self):
        """Builds the mapping dict {ista_num: geo_face_key}"""
        mapping = {}
        # Geometry Keys: T(Top), D(Down/Bottom), F(Front/Screen), B(Back), R(Right), L(Left)
        # Note: 'Right' means Right side when viewing the Front (Screen).
        # In BoxGeometry with Front=+Y, Right=+X is correct (Standard math axes).
        # If Front is Screen, Right is to the right.
        
        if self.is_ltl:
            # LTL (Type H) - User Defined
            # 1: Top (T)
            # 2: Back (B)
            # 3: Bottom (D)
            # 4: Screen/Front (F)
            # 5: Right (R)
            # 6: Left (L)
            mapping = {
                1: 'T',
                2: 'B',
                3: 'D',
                4: 'F', 
                5: 'R',
                6: 'L'
            }
        else:
            # Parcel (Type G) - User Defined
            # 1: Back (B)
            # 2: Bottom (D)
            # 3: Screen/Front (F)
            # 4: Top (T)
            # 5: Right (R)
            # 6: Left (L)
            mapping = {
                1: 'B',
                2: 'D',
                3: 'F',
                4: 'T',
                5: 'R',
                6: 'L'
            }
            
        return mapping

    def get_face_label(self, geo_face_key):
        """Returns string like 'Face 1 [Top]'"""
        ista_num = self.rev_mapping.get(geo_face_key, '?')
        
        # Friendly Name
        long_names = {'T': 'Top', 'D': 'Bottom', 'F': 'Screen', 'B': 'Back', 'R': 'Right', 'L': 'Left'}
        fname = long_names.get(geo_face_key, geo_face_key)
        
        return f"Face {ista_num} [{fname}]"

    def get_edge_label(self, c1_name, c2_name):
        # Determine faces sharing this edge
        # Edge is intersection of 2 faces.
        # Simple heuristic: extract faces from corner names?
        # C_TFL, C_TFR -> Common: T, F.
        
        faces = self._get_faces_from_edge(c1_name, c2_name)
        if len(faces) != 2: return "Edge (?)"
        
        n1 = self.rev_mapping.get(faces[0], '?')
        n2 = self.rev_mapping.get(faces[1], '?')
        
        # Sort for consistency
        if str(n1) > str(n2): n1, n2 = n2, n1
        
        return f"Edge {n1}-{n2} [{faces[0]}-{faces[1]}]"
        
    def get_corner_label(self, c_name):
        """Returns string like 'Corner 1-2-5 [Top-Front-Right]'"""
        # C_TFL -> T, F, L
        faces = []
        if 'T' in c_name: faces.append('T')
        if 'D' in c_name: faces.append('D')
        if 'F' in c_name: faces.append('F')
        if 'B' in c_name: faces.append('B')
        if 'R' in c_name: faces.append('R')
        if 'L' in c_name: faces.append('L')
        
        nums = [self.rev_mapping.get(f, '?') for f in faces]
        nums.sort()
        nums_str = "-".join([str(n) for n in nums])
        faces_str = "-".join(faces)
        
        return f"Corner {nums_str} [{faces_str}]"

    def _get_faces_from_edge(self, c1, c2):
        # C_TFL ('T','F','L') and C_TFR ('T','F','R') -> intersection ('T','F')
        f1 = set([c for c in c1 if c in 'TDFBRL'])
        f2 = set([c for c in c2 if c in 'TDFBRL'])
        return list(f1.intersection(f2))

class ISTA6ASimulator:
    """
    ISTA 6A 규격에 따른 낙하 시험 초기 자세(Pose) 계산기.
    """
    def __init__(self, geometry: BoxGeometry):
        self.geo = geometry
        # Default Mapper (Parcel)
        self.mapper = IstaFaceMapper(is_ltl=False, geo=geometry)

    def determine_ista_type(self, mass_kg, width, depth, height, shipment_method, handling_method, product_type='General'):
        """
        Determines the ISTA 6A Test Type (A-H) based on selected method and inputs.
        """
        dims = sorted([width, depth, height])
        L, W_s, H_s = dims[2], dims[1], dims[0]
        length_plus_girth_mm = L + 2 * (W_s + H_s)
        # Inputs are in meters in BoxGeometry usually (e.g. 1.4, 0.2, 0.8).
        # Let's check update_sequence calls. 
        # m=kg, w=float(mm/1000 or mm?), d=float.
        # Inside update_sequence:
        # m = mass_var.get(), w = w_var.get()... 
        # w_var default is 1400.0 (mm). 
        # update_sequence: tv_geo.width = w (mm?). 
        # Wait, BoxGeometry might expect meters?
        # Let's check BoxGeometry init or usage. 
        # In `generate_test_sequence`, `width` is passed.
        # In `determine_ista_type`, `length_plus_girth_in = length_plus_girth_mm / 25.4`.
        # If input is mm, then L is mm.
        # verify_ista_logic.py: `simulator.determine_ista_type(15.0, 1.4, 0.2, 0.8)` -> meters?
        # But UI inputs default 1400.
        # Let's check `update_sequence`: `tv_geo.width = w` (where w=1400).
        # BoxVisualizer might expect meters or mm?
        # But in this function `determine_ista_type`, previous code used `dims = sorted([width, depth, height])`.
        # And `length_plus_girth_in = length_plus_girth_mm / 25.4`.
        # If input is 1400mm, `1400/25.4` is ~55in. Correct.
        # If input is 1.4m, `1.4/25.4` is tiny.
        # So inputs to this function seem to be in MM from the UI (w_var.get()).
        # But wait, in `update_sequence`:
        # `tv_geo.dx... = w/2`.
        # If BoxGeometry uses mm, fine.
        # The verify script likely uses Meters? No, verify code used 1.4 which is small for mm.
        # Let's look at `generate_test_sequence` print: `Dims: {width*1000:.0f}...`.
        # This implies inputs are METERS in generate_test_sequence?
        # In `update_sequence`: `m = mass_var.get()` (kg), `w = w_var.get()` (mm).
        # We invoke `simulator.generate_test_sequence(m, w, d_val, h, ...)`.
        # If Update Sequence passes MM, then `width*1000` inside generate would be huge (1,400,000).
        # BUT code says `width*1000:.0f`.
        # If inputs are MM, we should NOT multiply by 1000 if we want mm.
        # BUT the UI inputs are 1400.
        # So likely the code in `generate_test_sequence` assumes Meters, but UI passes MM.
        # This might be a bug or inconsistency I should fix or work around.
        # However, `determine_ista_type` calculates girth in INCHES.
        # MM: 1400 + 2*(200+800) = 3400mm. / 25.4 = 133in. < 165. Correct.
        # So `determine_ista_type` expects MM.
        # `generate_test_sequence` print: `{width*1000}`. If width=1400, result 1400000.
        # The user seems to view outputs like `1400x200x800`.
        # If user sees 1400000x..., that's wrong.
        # I will assume inputs are MM for this function as per UI logic.
        # I will clarify metrics in Reason string assuming inputs are MM.
        
        dims = sorted([width, depth, height])
        L, W_s, H_s = dims[2], dims[1], dims[0]
        length_plus_girth_mm = L + 2 * (W_s + H_s)
        length_plus_girth_in = length_plus_girth_mm / 25.4
        mass_lb = mass_kg * 2.20462
        
        reason = ""
        type_code = 'Unknown'

        if shipment_method == 'Parcel':
            # Parcel Logic
            if length_plus_girth_in > 165.0:
                type_code = 'Invalid'
                reason = f"Parcel selected but Girth+Length {length_plus_girth_in:.1f}in ({length_plus_girth_mm:.0f}mm) > 165in (4191mm). Must use LTL."
                return type_code, reason
            
            if handling_method == 'Palletized':
                 type_code = 'Invalid'
                 reason = "Parcel shipment cannot use Pallet handling."
                 return type_code, reason

            if mass_lb < 70.0: 
                type_code = 'A'
                reason = f"Parcel (Standard), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) < 70lb (32kg) -> Type A"
            elif mass_lb < 150.0: 
                type_code = 'B'
                reason = f"Parcel (Standard), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) is 70-150lb (32-68kg) -> Type B"
            else: 
                type_code = 'C'
                reason = f"Parcel (Standard), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) >= 150lb (68kg) -> Type C"
            
        elif shipment_method == 'LTL':
            # LTL Logic
            # Check Handling First to allow Manual Override for Standard (Floor)
            if handling_method == 'Standard':
                if mass_lb < 100.0: 
                    type_code = 'D'
                    reason = f"LTL, Standard, Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) < 100lb (45kg) -> Type D"
                else: 
                    type_code = 'E'
                    reason = f"LTL, Standard, Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) >= 100lb (45kg) -> Type E"
            
            elif product_type == 'TV/Monitor':
                # Map to TV types regardless of handling selection (usually Palletized implied)
                if mass_lb < 150.0: 
                    type_code = 'G'
                    reason = f"LTL (TV/Monitor), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) < 150lb (68kg) -> Type G"
                else: 
                    type_code = 'H'
                    reason = f"LTL (TV/Monitor), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) >= 150lb (68kg) -> Type H"
            
            else: # Palletized, General
                if mass_lb < 150.0: 
                    type_code = 'F'
                    reason = f"LTL, Palletized, Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) < 150lb (68kg) -> Type F"
                else:
                    type_code = 'E'
                    reason = f"LTL, Palletized, Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) >= 150lb (68kg) -> Type E"
            
        # Append Girth info to reason if relevant (not usually for weight-only triggers strings above, but useful context)
        # Or just rely on the strings constructed above.
        # User example: "Weight ... < 70lb, Girth ... <= 165in".
        # My strings above removed Girth text for brevity in previous edits?
        # I will re-add Girth info to Parcel types A/B/C as it's a constraint.
        if type_code in ['A', 'B', 'C']:
             reason += f", Girth {length_plus_girth_in:.1f}in ({length_plus_girth_mm:.0f}mm) <= 165in (4191mm)"
             
        return type_code, reason

    def generate_test_sequence(self, mass_kg, width, depth, height, shipment_method, handling_method, product_type='General'):
        """
        Dynamically generates test sequence based on ISTA Type.
        """
        seq = []
        type_code, reason = self.determine_ista_type(mass_kg, width, depth, height, shipment_method, handling_method, product_type)
        
        print(f"\n[ISTA Type Determination]")
        print(f" - Inputs: {shipment_method} / {handling_method} (Prod: {product_type})")
        print(f" - Dims: {width:.0f}x{depth:.0f}x{height:.0f} mm, Mass: {mass_kg:.2f} kg")
        print(f" - Decision: Type {type_code}")
        print(f" - Reason: {reason}\n")
        
        # Map Type to Sequence Logic
        # LTL Types: D, E, F, G, H
        is_ltl = type_code in ['D', 'E', 'F', 'G', 'H']
        
        # Create Mapper
        self.mapper = IstaFaceMapper(is_ltl=is_ltl, geo=self.geo)
        
        # Unit Helper
        # Using User's Metric Table values directly
        # 12" -> 300mm
        # 18" -> 460mm
        # 24" -> 610mm
        # 36" -> 910mm
        # 32" -> 810mm
        # 9" -> 230mm
        
        mass_lb = mass_kg * 2.20462
        
        if not is_ltl: # Parcel (Type A, B, G) - 17 Steps (Block 2 & 15)
            # Heights depend on Mass: < 70lb (32kg) vs 70-150lb (32-68kg).
            if mass_lb < 70.0: # ~32kg
                h_std = 460.0 # 18" -> 460mm
                h_high = 910.0 # 36" -> 910mm
            else: # 70 - 150 lb
                h_std = 300.0 # 12" -> 300mm
                h_high = 610.0 # 24" -> 610mm
            
            # --- Block 2 (1st Drop Sequence) ---
            # --- Block 2 (1st Drop Sequence) ---
            # 1. Edge 3-4
            seq.append({'type': 'edge', 'id': ('C_TFL', 'C_TFR'), 'height': h_std, 
                        'name': 'Edge 3-4', 'detail': 'Screen-Top',
                        'desc': f"1. Edge 3-4 (Screen-Top) - {h_std:.0f}mm"})
            
            # 2. Edge 3-6
            seq.append({'type': 'edge', 'id': ('C_TFL', 'C_DFL'), 'height': h_std, 
                        'name': 'Edge 3-6', 'detail': 'Screen-Left',
                        'desc': f"2. Edge 3-6 (Screen-Left) - {h_std:.0f}mm"})
            
            # 3. Edge 4-6
            seq.append({'type': 'edge', 'id': ('C_TFL', 'C_TBL'), 'height': h_std, 
                        'name': 'Edge 4-6', 'detail': 'Top-Left',
                        'desc': f"3. Edge 4-6 (Top-Left) - {h_std:.0f}mm"})
            
            # 4. Corner 3-4-6
            seq.append({'type': 'corner', 'id': 'C_TFL', 'height': h_std, 
                        'name': 'Corner 3-4-6', 'detail': 'Screen-Top-Left',
                        'desc': f"4. Corner 3-4-6 (Screen-Top-Left) - {h_std:.0f}mm"})
            
            # 5. Corner 2-3-5
            seq.append({'type': 'corner', 'id': 'C_DFR', 'height': h_std, 
                        'name': 'Corner 2-3-5', 'detail': 'Bottom-Screen-Right',
                        'desc': f"5. Corner 2-3-5 (Bottom-Screen-Right) - {h_std:.0f}mm"})
            
            # 6. Edge 2-3
            seq.append({'type': 'edge', 'id': ('C_DFL', 'C_DFR'), 'height': h_std, 
                        'name': 'Edge 2-3', 'detail': 'Bottom-Screen',
                        'desc': f"6. Edge 2-3 (Bottom-Screen) - {h_std:.0f}mm"})
            
            # 7. Edge 1-2
            seq.append({'type': 'edge', 'id': ('C_DBL', 'C_DBR'), 'height': h_std, 
                        'name': 'Edge 1-2', 'detail': 'Back-Bottom',
                        'desc': f"7. Edge 1-2 (Back-Bottom) - {h_std:.0f}mm"})
            
            # 8. Face 3 (Screen) [High]
            seq.append({'type': 'face', 'id': 'F', 'height': h_high, 
                        'name': 'Face 3', 'detail': 'Screen (High)',
                        'desc': f"8. Face 3 (Screen) [High] - {h_high:.0f}mm"})
            
            # 9. Face 3 (Screen) [Low]
            seq.append({'type': 'face', 'id': 'F', 'height': h_std, 
                        'name': 'Face 3', 'detail': 'Screen (Low)',
                        'desc': f"9. Face 3 (Screen) [Low] - {h_std:.0f}mm"})
            
            # --- Block 15 (2nd Drop Sequence) ---
            # 10. Edge 3-4
            seq.append({'type': 'edge', 'id': ('C_TFL', 'C_TFR'), 'height': h_std, 
                        'name': 'Edge 3-4', 'detail': 'Screen-Top (Repeated)',
                        'desc': f"10. Edge 3-4 (Screen-Top) - {h_std:.0f}mm"})
            
            # 11. Edge 3-6
            seq.append({'type': 'edge', 'id': ('C_TFL', 'C_DFL'), 'height': h_std, 
                        'name': 'Edge 3-6', 'detail': 'Screen-Left (Repeated)',
                        'desc': f"11. Edge 3-6 (Screen-Left) - {h_std:.0f}mm"})
            
            # 12. Edge 1-5
            seq.append({'type': 'edge', 'id': ('C_DBR', 'C_TBR'), 'height': h_std, 
                        'name': 'Edge 1-5', 'detail': 'Back-Right',
                        'desc': f"12. Edge 1-5 (Back-Right) - {h_std:.0f}mm"})
            
            # 13. Corner 3-4-6
            seq.append({'type': 'corner', 'id': 'C_TFL', 'height': h_std, 
                        'name': 'Corner 3-4-6', 'detail': 'Screen-Top-Left',
                        'desc': f"13. Corner 3-4-6 (Screen-Top-Left) - {h_std:.0f}mm"})
            
            # 14. Corner 1-2-6
            seq.append({'type': 'corner', 'id': 'C_DBL', 'height': h_std, 
                        'name': 'Corner 1-2-6', 'detail': 'Back-Bottom-Left',
                        'desc': f"14. Corner 1-2-6 (Back-Bottom-Left) - {h_std:.0f}mm"})
            
            # 15. Corner 1-4-5
            seq.append({'type': 'corner', 'id': 'C_TBR', 'height': h_std, 
                        'name': 'Corner 1-4-5', 'detail': 'Back-Top-Right',
                        'desc': f"15. Corner 1-4-5 (Back-Top-Right) - {h_std:.0f}mm"})
            
            # 16. Face 6
            seq.append({'type': 'face', 'id': 'L', 'height': h_high, 
                        'name': 'Face 6', 'detail': 'Left (High)',
                        'desc': f"16. Face 6 (Left) [High] - {h_high:.0f}mm"})
            
            # 17. Hazard Drop
            seq.append({'type': 'face', 'id': 'F', 'height': h_std, 
                        'name': 'Face 3', 'detail': 'Hazard (Screen)',
                        'desc': f"17. Hazard Drop on Face 3 (Screen) - {h_std:.0f}mm"})

        else: # LTL (Type D, E, F, G, H)
            # Check for Rotational Drop Condition
            # Standard LTL: Mass >= 100lb (45.3kg) OR Girth > 165in (4191mm) -> Rotational
            # TV/Monitor (G, H): Primarily Rotational as per user description ("Rotational Edge Drop & Tip-over essential for G")
            
            is_heavy_ltl = (mass_lb >= 100.0)
            if type_code in ['G', 'H']: 
                is_heavy_ltl = True # Force Rotational for TV/Monitor Types
            
            if is_heavy_ltl:
                # Rotational Drop Sequence
                lift_h = 230.0 # 9" -> 230mm
                
                # 1. Rot Edge Back-Bottom
                seq.append({'type': 'rot_edge', 'pivot_edge': ('C_DBL', 'C_DBR'), 'lift_height': lift_h, 
                            'name': 'Rot Edge Back-Bottom', 'detail': 'Pivot: Back-Bottom',
                            'desc': f"Rot. Edge: Pivot Back-Bottom - Lift {lift_h:.0f}mm"})
                # 2. Rot Edge Front-Bottom
                seq.append({'type': 'rot_edge', 'pivot_edge': ('C_DFL', 'C_DFR'), 'lift_height': lift_h, 
                            'name': 'Rot Edge Front-Bottom', 'detail': 'Pivot: Front-Bottom',
                            'desc': f"Rot. Edge: Pivot Front-Bottom - Lift {lift_h:.0f}mm"})
                # 3. Rot Edge Right-Bottom
                seq.append({'type': 'rot_edge', 'pivot_edge': ('C_DFR', 'C_DBR'), 'lift_height': lift_h, 
                            'name': 'Rot Edge Right-Bottom', 'detail': 'Pivot: Right-Bottom',
                            'desc': f"Rot. Edge: Pivot Right-Bottom - Lift {lift_h:.0f}mm"})
                # 4. Rot Edge Left-Bottom
                seq.append({'type': 'rot_edge', 'pivot_edge': ('C_DFL', 'C_DBL'), 'lift_height': lift_h, 
                            'name': 'Rot Edge Left-Bottom', 'detail': 'Pivot: Left-Bottom',
                            'desc': f"Rot. Edge: Pivot Left-Bottom - Lift {lift_h:.0f}mm"})
                # 5. Rot Corner Back-Left-Bottom
                seq.append({'type': 'rot_corner', 'pivot_corner': 'C_DBL', 'lift_height': lift_h, 
                            'name': 'Rot Corner Back-Le-Bot', 'detail': 'Pivot: Back-Left-Bottom',
                            'desc': f"Rot. Corner: Pivot Back-Left-Bottom - Lift {lift_h:.0f}mm"})
                
            else:
                # LTL Lightweight (<100lb) - 12 Drops (Block 4 & 16)
                h_12 = 300.0 # 12" -> 300mm
                h_18 = 460.0 # 18" -> 460mm
                h_32 = 810.0 # 32" -> 810mm
                
                # 1. Face 1 (Top)
                seq.append({'type': 'face', 'id': 'T', 'height': h_12, 
                            'name': 'Face 1', 'detail': 'Top',
                            'desc': f"1. Face 1 (Top) - {h_12:.0f}mm"})
                # 2. Face 2 (Back)
                seq.append({'type': 'face', 'id': 'B', 'height': h_12, 
                            'name': 'Face 2', 'detail': 'Back',
                            'desc': f"2. Face 2 (Back) - {h_12:.0f}mm"})
                # 3. Face 6 (Left)
                seq.append({'type': 'face', 'id': 'L', 'height': h_12, 
                            'name': 'Face 6', 'detail': 'Left',
                            'desc': f"3. Face 6 (Left) - {h_12:.0f}mm"})
                # 4. Corner 2-3-5
                seq.append({'type': 'corner', 'id': 'C_DBR', 'height': h_12, 
                            'name': 'Corner 2-3-5', 'detail': 'Back-Bottom-Right',
                            'desc': f"4. Corner 2-3-5 (Back-Bottom-Right) - {h_12:.0f}mm"})
                # 5. Edge 3-4
                seq.append({'type': 'edge', 'id': ('C_DFL', 'C_DFR'), 'height': h_12, 
                            'name': 'Edge 3-4', 'detail': 'Bottom-Screen',
                            'desc': f"5. Edge 3-4 (Bottom-Screen) - {h_12:.0f}mm"})
                # 6. Face 3 (Bottom)
                seq.append({'type': 'face', 'id': 'D', 'height': h_18, 
                            'name': 'Face 3', 'detail': 'Bottom (Med)',
                            'desc': f"6. Face 3 (Bottom) - {h_18:.0f}mm"})
                
                # --- Block 16 ---
                # 7. Edge 2-3
                seq.append({'type': 'edge', 'id': ('C_DBL', 'C_DBR'), 'height': h_18, 
                            'name': 'Edge 2-3', 'detail': 'Back-Bottom',
                            'desc': f"7. Edge 2-3 (Back-Bottom) - {h_18:.0f}mm"})
                # 8. Corner 3-4-6
                seq.append({'type': 'corner', 'id': 'C_DFL', 'height': h_18, 
                            'name': 'Corner 3-4-6', 'detail': 'Bottom-Screen-Left',
                            'desc': f"8. Corner 3-4-6 (Bottom-Screen-Left) - {h_18:.0f}mm"})
                # 9. Edge 4-5
                seq.append({'type': 'edge', 'id': ('C_TFR', 'C_DFR'), 'height': h_18, 
                            'name': 'Edge 4-5', 'detail': 'Screen-Right',
                            'desc': f"9. Edge 4-5 (Screen-Right) - {h_18:.0f}mm"})
                # 10. Corner 1-4-6
                seq.append({'type': 'corner', 'id': 'C_TFL', 'height': h_18, 
                            'name': 'Corner 1-4-6', 'detail': 'Top-Screen-Left',
                            'desc': f"10. Corner 1-4-6 (Top-Screen-Left) - {h_18:.0f}mm"})
                # 11. Edge 1-6
                seq.append({'type': 'edge', 'id': ('C_TFL', 'C_TBL'), 'height': h_18, 
                            'name': 'Edge 1-6', 'detail': 'Top-Left',
                            'desc': f"11. Edge 1-6 (Top-Left) - {h_18:.0f}mm"})
                # 12. Face 3 (Bottom)
                seq.append({'type': 'face', 'id': 'D', 'height': h_32, 
                            'name': 'Face 3', 'detail': 'Bottom (High)',
                            'desc': f"12. Face 3 (Bottom) [High] - {h_32:.0f}mm"})

        return seq, type_code


    def calculate_rotational_pose(self, drop_type, lift_height, pivot_idx):
        """
        회전 낙하(Rotational Drop)를 위한 초기 자세를 계산합니다.
        박스의 한 모서리나 코너를 바닥에 고정하고(Pivot), 반대편을 들어올린 상태입니다.
        
        Args:
            drop_type (str): 'rot_edge' or 'rot_corner'
            lift_height (float): 들어올리는 높이 (m)
            pivot_idx (str or tuple): 
                - rot_edge: 고정할 엣지의 코너 튜플 (c1, c2)
                - rot_corner: 고정할 코너 이름 (c_name)
        
        Returns:
            np.array: 초기 포즈 [tx, ty, tz, rx, ry, rz]
        """
        # 1. 기본적으로 박스는 바닥(Face 2, Down)에 놓여 있다고 가정하고 시작
        #    만약 다른 면이 바닥이라면 먼저 그 면을 바닥으로 보내야 하지만,
        #    ISTA Rotational Drop은 보통 'Supported Face' 기준임. 여기선 D(Bottom) 가정.
        
        # 바닥에 놓인 상태 (Face D is at Z=0) -> Center Z = +dz
        pose_flat = np.array([0, 0, self.geo.dz, 0, 0, 0])
        
        if drop_type == 'rot_edge':
            # pivot_idx = (c1, c2) defining the edge on the ground
            # We need to rotate around this edge until the "opposite edge" is at lift_height.
            # Opposite edge distance in a simple box is typically the width or depth.
            
            # 1. Find the axis of the edge in World Frame (assuming flat pose)
            c1_local = self.geo.corners[pivot_idx[0]]
            c2_local = self.geo.corners[pivot_idx[1]]
            
            # Pivot edge center (should be at Z=0 after adjustment)
            edge_center_local = (c1_local + c2_local) / 2.0
            
            # Vector from pivot edge to center (in XY plane)
            # This is the 'arm' we are lifting.
            # Local vector from edge to center
            # Center is (0,0,0) local.
            v_arm = -np.array([edge_center_local[0], edge_center_local[1], 0]) 
            arm_len = np.linalg.norm(v_arm) # Half width/depth
            full_arm_len = arm_len * 2 # Full width/depth
            
            # Calculate rotation angle theta
            # sin(theta) = h / L
            if full_arm_len < 1e-6: theta = 0
            else: theta = np.arcsin(np.clip(lift_height / full_arm_len, -1.0, 1.0))
            
            # Rotation Axis: Vector along the edge
            edge_vec = c2_local - c1_local
            edge_vec[2] = 0 # Projection on XY (should be already)
            edge_axis = edge_vec / np.linalg.norm(edge_vec)
            
            # Calculate Rotation Vector (Axis * Angle)
            # We need to rotate AROUND the edge. 
            # If we rotate the box, the center moves.
            
            # Let's construct the transform:
            # 1. Translate Center to Pivot Edge (so Pivot Edge is at Origin)
            # 2. Rotate
            # 3. Translate back (Pivot Edge stays at Origin)
            # 4. Move Pivot Edge to World Origin (optional, but good for vis) or keep flat.
            
            # Actually, we want the pivot edge to stay at Z=0.
            
            # Rotation Quat/Matrix
            r_rot = R.from_rotvec(edge_axis * theta)
            
            # New Center Position
            # Center was at (0,0,dz) relative to the pivot plane, but let's do relative to pivot line.
            # Vector from Pivot Line to Center: v_arm (XY) + (0,0,dz)
            v_pivot_to_center = np.array([-edge_center_local[0], -edge_center_local[1], self.geo.dz])
            
            v_new = r_rot.apply(v_pivot_to_center)
            
            # We want Pivot Line to be at Z=0.
            # If we fix Pivot Line at World Origin (for simplicity):
            pose = np.zeros(6)
            pose[:3] = v_new
            pose[3:] = r_rot.as_rotvec()
            
            return pose

        elif drop_type == 'rot_corner':
             # Pivot on a corner
             # Lift the opposite corner? Or lift the center?
             # ISTA: "Place a support block under one corner... 
             # wait, that's rotational flat.
             # Rotational Corner Drop: "Place the packaged-product on the floor... 
             # Place a support block under the corner to be tested... No, that's hazard.
             # Standard Rotational Corner Drop: Lift one corner 
             # while the other three corners of the base face remain on the floor? No, impossible rigid body.
             # Usually: Pivot on one corner, lift the diagonally opposite corner of the SAME FACE.
             
             # Let's assume Pivot is at Origin.
             c_pivot = self.geo.corners[pivot_idx]
             
             # Diagonal corner on the same face (Bottom)
             # If c_pivot is DFL, diagonal is DBR.
             # Vector from Pivot to Diagonal
             # Center is (0,0,0) local. 
             # Let's find the diagonal corner on the bottom face.
             # Assuming D face.
             # v_diag = -c_pivot (in XY only) -> since box is symmetric centered.
             # Length L = sqrt(W^2 + D^2)
             
             v_diag_vec = -np.array([c_pivot[0], c_pivot[1], 0]) * 2.0
             L = np.linalg.norm(v_diag_vec)
             
             theta = np.arcsin(np.clip(lift_height / L, -1.0, 1.0))
             
             # Rotation Axis: Perpendicular to diagonal vector in XY plane
             # v_diag = (dx, dy, 0) -> axis = (-dy, dx, 0)
             axis = np.array([-v_diag_vec[1], v_diag_vec[0], 0])
             axis = axis / np.linalg.norm(axis)
             
             r_rot = R.from_rotvec(axis * theta)
             
             # Center Position relative to Pivot Corner
             v_pivot_to_center = np.array([-c_pivot[0], -c_pivot[1], self.geo.dz])
             v_new = r_rot.apply(v_pivot_to_center)
             
             pose = np.zeros(6)
             pose[:3] = v_new
             pose[3:] = r_rot.as_rotvec()
             
             return pose

        return pose_flat

    def calculate_drop_pose(self, drop_type, height, impact_idx=None):
        """
        낙하 시험을 위한 초기 자세(Pose)를 계산합니다.
        박스의 최하단점이 지정된 높이(height)에 위치하도록 합니다.
        
        Args:
            drop_type (str): 낙하 유형 ('face', 'edge', 'corner')
            height (float): 낙하 높이 (m) - 박스의 최하단점 기준
            impact_idx (str or tuple): 충돌 부위 식별자
                - face: 'F', 'B', 'L', 'R', 'T', 'D'
                - edge: (corner_name1, corner_name2) 튜플
                - corner: corner_name (예: 'C_DFL')
            
        Returns:
            np.array: 계산된 초기 포즈 [tx, ty, tz, rx, ry, rz]
        """
        # 초기 회전 설정 (충돌 부위가 가장 아래로 오도록)
        r_init = R.identity()
        
        if drop_type == 'face':
            # 해당 면의 Normal이 (0,0,-1)이 되도록 회전
            target_face = impact_idx if impact_idx else 'D'
            normal = self.geo.faces[target_face]['normal']
            
            # Rotate normal to -Z
            target_vec = np.array([0, 0, -1])
            if not np.allclose(normal, target_vec):
                axis = np.cross(normal, target_vec)
                norm_axis = np.linalg.norm(axis)
                if norm_axis < 1e-6: # Parallel
                    if np.dot(normal, target_vec) < 0: # Opposite direction
                        axis = np.array([1, 0, 0])
                        angle = np.pi
                    else:
                        angle = 0
                else:
                    axis = axis / norm_axis
                    angle = np.arccos(np.clip(np.dot(normal, target_vec), -1.0, 1.0))
                
                if angle != 0:
                    r_init = R.from_rotvec(axis * angle)
                
        elif drop_type == 'corner':
            # 해당 코너가 가장 아래(-Z)에 오도록 회전
            # 코너 벡터(중심->코너)가 (0,0,-1) 방향이 되도록
            c_name = impact_idx if impact_idx else 'C_DFL'
            c_vec = self.geo.corners[c_name]
            c_dir = c_vec / np.linalg.norm(c_vec)
            target_vec = np.array([0, 0, -1])
            
            axis = np.cross(c_dir, target_vec)
            norm_axis = np.linalg.norm(axis)
            if norm_axis < 1e-6:
                if np.dot(c_dir, target_vec) < 0:
                    axis = np.array([1, 0, 0])
                    angle = np.pi
                else:
                    angle = 0
            else:
                axis = axis / norm_axis
                angle = np.arccos(np.clip(np.dot(c_dir, target_vec), -1.0, 1.0))
            
            if angle != 0:
                r_init = R.from_rotvec(axis * angle)
            
        elif drop_type == 'edge':
            # 엣지가 바닥과 평행하고 가장 아래에 오도록 회전
            # 엣지의 중점 벡터가 (0,0,-1) 방향이 되도록 함
            if isinstance(impact_idx, (tuple, list)) and len(impact_idx) == 2:
                c1 = self.geo.corners[impact_idx[0]]
                c2 = self.geo.corners[impact_idx[1]]
                m_vec = (c1 + c2) / 2.0
                m_dir = m_vec / np.linalg.norm(m_vec)
                target_vec = np.array([0, 0, -1])
                
                axis = np.cross(m_dir, target_vec)
                norm_axis = np.linalg.norm(axis)
                if norm_axis < 1e-6:
                    if np.dot(m_dir, target_vec) < 0:
                        axis = np.array([1, 0, 0])
                        angle = np.pi
                    else:
                        angle = 0
                else:
                    axis = axis / norm_axis
                    angle = np.arccos(np.clip(np.dot(m_dir, target_vec), -1.0, 1.0))
                
                if angle != 0:
                    r_init = R.from_rotvec(axis * angle)
            else:
                print("Warning: Edge drop requires a tuple of 2 corner names. Using default 45 deg.")
                r_init = R.from_euler('x', 45, degrees=True)
        
        # 박스의 최하단점 찾기 (회전 적용 후)
        corners = self.geo.get_vertices()
        rotated_corners = r_init.apply(corners)
        min_z_local = np.min(rotated_corners[:, 2])
        
        # 최하단점이 height에 오도록 z 이동량 계산
        tz = height - min_z_local
        
        pose = np.zeros(6)
        pose[:3] = [0, 0, tz]
        pose[3:] = r_init.as_rotvec()
        
        return pose

class BoxVisualizer:
    """
    Matplotlib 및 PyVista를 이용한 시각화 클래스.
    """
    # ISTA 6-Amazon.com Parcel Delivery (Type A, B) Face Mapping
    # 1: Top, 2: Bottom, 3: Front, 4: Back, 5: Right, 6: Left
    ISTA_MAPPING = {
        'T': 1, 'D': 2, 
        'F': 3, 'B': 4, 
        'R': 5, 'L': 6
    }

    def __init__(self, geometry: BoxGeometry):
        """
        BoxVisualizer 초기화.
        
        Args:
            geometry (BoxGeometry): 시각화할 박스의 기하학 정보
        """
        self.geo = geometry
        # Default Mapper (can be updated externally)
        self.mapper = IstaFaceMapper(is_ltl=False, geo=geometry)


    def _get_box_edges(self, pose=None):
        """
        박스의 와이어프레임 엣지(선분)와 꼭짓점 좌표를 계산하여 반환합니다.
        
        Returns:
            tuple: (lines, pts) - lines는 선분 리스트, pts는 꼭짓점 딕셔너리
        """
        corners = self.geo.corners
        
        # Local coords
        pts = {}
        for k, v in corners.items():
            pts[k] = v
            
        if pose is not None:
            t = pose[:3]
            r = R.from_rotvec(pose[3:])
            for k in pts:
                pts[k] = r.apply(pts[k]) + t
                
        # Edge definitions with IDs
        # Top: TF, TR, TB, TL
        # Bottom: DF, DR, DB, DL
        # Pillars: FL, FR, BR, BL
        self.edge_defs = {
            'TF': ('C_TFL', 'C_TFR'), 'TR': ('C_TFR', 'C_TBR'), 'TB': ('C_TBR', 'C_TBL'), 'TL': ('C_TBL', 'C_TFL'),
            'DF': ('C_DFL', 'C_DFR'), 'DR': ('C_DFR', 'C_DBR'), 'DB': ('C_DBR', 'C_DBL'), 'DL': ('C_DBL', 'C_DFL'),
            'FL': ('C_TFL', 'C_DFL'), 'FR': ('C_TFR', 'C_DFR'), 'BR': ('C_TBR', 'C_DBR'), 'BL': ('C_TBL', 'C_DBL')
        }
        
        lines = []
        for eid, (s, e) in self.edge_defs.items():
            lines.append([pts[s], pts[e]])
            
        return lines, pts

    def _get_face_centers(self, pose=None):
        """
        Calculate face centers and IDs.
        Returns: dict {id: center_coord}
        """
        centers = {}
        # Local face centers
        for fid, face in self.geo.faces.items():
            c = np.zeros(3)
            c[face['axis_idx']] = face['val']
            centers[fid] = c
            
        if pose is not None:
            t = pose[:3]
            r = R.from_rotvec(pose[3:])
            for k in centers:
                centers[k] = r.apply(centers[k]) + t
        return centers

    def _get_edge_centers(self, pts):
        """
        Calculate edge centers from transformed points.
        Returns: dict {id: center_coord}
        """
        centers = {}
        if not hasattr(self, 'edge_defs'):
             self._get_box_edges() # Init edge_defs

        for eid, (s, e) in self.edge_defs.items():
            if s in pts and e in pts:
                centers[eid] = (pts[s] + pts[e]) / 2.0
        return centers

    def calculate_world_fixed_info(self, pose, print_report=True):
        """
        월드 고정 뷰(World Fixed View)를 위한 변환 및 벡터 정보를 계산하고 안내 메시지를 출력합니다.
        
        Args:
            pose (np.array): 박스 포즈 [tx, ty, tz, rx, ry, rz]
            print_report (bool): 안내 메시지 출력 여부
            
        Returns:
            dict: 계산된 정보 (t, r, world_down_vec, world_normal_vec)
        """
        t = pose[:3]
        r_vec = pose[3:]
        r = R.from_rotvec(r_vec)
        
        world_down_vec = np.array([0.0, 0.0, -1.0])
        world_normal_vec = np.array([0.0, 0.0, 1.0])
        
        if print_report:
            print("\n" + "="*60)
            print(" [World Fixed View Guidance]")
            print("="*60)
            print("Scenario: The Ground is fixed at the World Origin (0,0,0).")
            print("          The Box moves relative to the World.")
            print("-" * 50)
            print("1. Transformation (Box -> World):")
            print(f"   Translation (t): {t}")
            print(f"   Rotation (R) Euler (XYZ): {r.as_euler('xyz', degrees=True)}")
            print("-" * 50)
            print("2. Gravity / Down Vector in World Frame:")
            print(f"   v_down_world = {world_down_vec}")
            print("-" * 50)
            print("3. Ground Plane Normal in World Frame:")
            print(f"   n_ground_world = {world_normal_vec}")
            print("="*60 + "\n")
            
        return {
            't': t,
            'r': r,
            'world_down_vec': world_down_vec,
            'world_normal_vec': world_normal_vec
        }

    def calculate_box_fixed_info(self, pose, print_report=True):
        """
        박스 고정 뷰(Box Fixed View)를 위한 역변환 및 벡터 정보를 계산하고 안내 메시지를 출력합니다.
        
        Args:
            pose (np.array): 박스 포즈 [tx, ty, tz, rx, ry, rz]
            print_report (bool): 안내 메시지 출력 여부
            
        Returns:
            dict: 계산된 정보 (t_inv, r_inv, local_down_vec, local_normal_vec)
        """
        t = pose[:3]
        r_vec = pose[3:]
        r = R.from_rotvec(r_vec)
        r_inv = r.inv()
        t_inv = -r_inv.apply(t)
        
        world_down_vec = np.array([0.0, 0.0, -1.0])
        local_down_vec = r_inv.apply(world_down_vec)
        
        world_normal_vec = np.array([0.0, 0.0, 1.0])
        local_normal_vec = r_inv.apply(world_normal_vec)
        
        if print_report:
            print("\n" + "="*60)
            print(" [Box Fixed View Guidance]")
            print("="*60)
            print("Scenario: The Box is fixed at the Local Origin (0,0,0).")
            print("          The World (Ground, Gravity, Points) moves relative to the Box.")
            print("-" * 50)
            print("1. Inverse Transformation (World -> Box):")
            print(f"   Translation (t_inv): {t_inv}")
            print(f"   Rotation (R_inv) Euler (XYZ): {r_inv.as_euler('xyz', degrees=True)}")
            print("-" * 50)
            print("2. Gravity / Down Vector in Box Frame:")
            print("   v_down_world = [0, 0, -1]")
            print("   v_down_local = R_inv * v_down_world")
            print(f"   Result: {local_down_vec}")
            print("-" * 50)
            print("3. Ground Plane Normal in Box Frame:")
            print("   n_ground_world = [0, 0, 1]")
            print("   n_ground_local = R_inv * n_ground_world")
            print(f"   Result: {local_normal_vec}")
            print("="*60 + "\n")
            
        return {
            't_inv': t_inv,
            'r_inv': r_inv,
            'local_down_vec': local_down_vec,
            'local_normal_vec': local_normal_vec
        }

    def show_matplotlib(self, pose, points_input, title="Box Motion Estimation", view_mode='world_fixed', deformation_info=None, penetration_info=None, print_guidance=True):
        """
        Matplotlib를 사용하여 박스와 점들을 3D로 시각화합니다.
        
        Args:
            pose (np.array): 박스 포즈 [tx, ty, tz, rx, ry, rz]
            points_input (list): 시각화할 점 데이터 리스트
            title (str): 그래프 제목
            view_mode (str): 'world_fixed' (박스 이동) 또는 'box_fixed' (박스 고정, 세상 이동)
            deformation_info (dict, optional): 변형(절단) 정보
            penetration_info (dict, optional): 지면 침투 정보 (해결 벡터 등)
            print_guidance (bool): Box Fixed 모드일 때 안내 메시지 출력 여부
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine transformations based on view_mode
        if view_mode == 'world_fixed':
            # Calculate Transform using helper method
            info = self.calculate_world_fixed_info(pose, print_report=print_guidance)
            
            # Box moves to 'pose', Points are in World Frame
            box_pose_to_draw = pose
            points_to_draw = copy.deepcopy(points_input) # coords are already world
            
            # Ground Plane is at Z=0 (Identity transform)
            ground_pose_t = np.zeros(3)
            ground_pose_R = np.eye(3)
            
            title += " (View: World Fixed, Box Moving)"
            
        else: # 'box_fixed'
            # Box stays at Origin, World moves by Inverse Pose
            box_pose_to_draw = np.zeros(6)
            
            # Calculate Inverse Transform using helper method
            info = self.calculate_box_fixed_info(pose, print_report=print_guidance)
            t_inv = info['t_inv']
            r_inv = info['r_inv']
            
            # Transform points to Box Local Frame
            points_to_draw = []
            for p in points_input:
                p_new = p.copy()
                p_new['coord'] = r_inv.apply(p['coord'] - pose[:3])
                points_to_draw.append(p_new)
                
            # Ground Plane moves by Inverse Transform
            ground_pose_t = t_inv
            ground_pose_R = r_inv.as_matrix()
            
            title += " (View: Box Fixed, World/Plane Moving)"

        # 1. Draw Box
        _, pts_tf = self._get_box_edges(pose=box_pose_to_draw)
        
        # Define faces by corner names for Poly3DCollection
        faces_def = [
            [pts_tf['C_DFL'], pts_tf['C_DFR'], pts_tf['C_TFR'], pts_tf['C_TFL']], # Front
            [pts_tf['C_DBL'], pts_tf['C_DBR'], pts_tf['C_TBR'], pts_tf['C_TBL']], # Back
            [pts_tf['C_DBL'], pts_tf['C_DFL'], pts_tf['C_TFL'], pts_tf['C_TBL']], # Left
            [pts_tf['C_DBR'], pts_tf['C_DFR'], pts_tf['C_TFR'], pts_tf['C_TBR']], # Right
            [pts_tf['C_TFL'], pts_tf['C_TFR'], pts_tf['C_TBR'], pts_tf['C_TBL']], # Top
            [pts_tf['C_DFL'], pts_tf['C_DFR'], pts_tf['C_DBR'], pts_tf['C_DBL']]  # Down
        ]
        
        # Assign different colors to each face
        face_colors = ['cyan', 'magenta', 'yellow', 'green', 'blue', 'red']
        
        poly_collection = Poly3DCollection(faces_def, facecolors=face_colors, linewidths=1, edgecolors='k', alpha=0.5)
        ax.add_collection3d(poly_collection)
            
        # Label Corners
        for name, pt in pts_tf.items():
            ax.text(pt[0], pt[1], pt[2], name.replace('C_', ''), fontsize=8, color='black')

        # Label Faces
        face_centers = self._get_face_centers(pose=box_pose_to_draw)
        
        # Use Mapper if available, else default
        mapper = getattr(self, 'mapper', None)
        if mapper is None:
            # Create a default mapper if not present (Parcel default)
            mapper = IstaFaceMapper(is_ltl=False, geo=self.geo)

        for fid, pt in face_centers.items():
             label = mapper.get_face_label(fid)
             # Simplify for plot: "Face 1 [Top]" -> "1 [T]" or keep Full?
             # Let's keep it informative but compact: "1\n[Top]"
             ista_num = mapper.rev_mapping.get(fid, '?')
             short_name = {'T':'Top', 'D':'Bot', 'F':'Frt', 'B':'Bck', 'R':'Rgt', 'L':'Lft'}.get(fid, fid)
             label_text = f"{ista_num}\n[{short_name}]"
             
             ax.text(pt[0], pt[1], pt[2], label_text, fontsize=10, fontweight='bold', color='navy')

        # Label Edges
        edge_centers = self._get_edge_centers(pts_tf)
        for eid, pt in edge_centers.items():
             ax.text(pt[0], pt[1], pt[2], eid, fontsize=8, color='darkgreen')

        # 1.1 Draw Deformation Markers (if info provided)
        if deformation_info:
            for name, val in deformation_info.items():
                # Handle dictionary (cut lengths) or scalar (penetration depth)
                if isinstance(val, dict):
                    scalar_val = max(val.values())
                else:
                    scalar_val = val

                if scalar_val > 1e-2: # 의미있는 변형량이 있는 경우 (0.01mm 이상)
                    pt = pts_tf[name]
                    # 빨간색 구 형태로 표시 (크기는 고정 혹은 비례)
                    ax.scatter(pt[0], pt[1], pt[2], s=100, c='red', marker='o', alpha=0.8, edgecolors='white')
                    # 텍스트로 변형량(mm) 표시
                    label_text = f"\nDef: {scalar_val:.1f}mm"
                    ax.text(pt[0], pt[1], pt[2], label_text, color='red', fontsize=9, fontweight='bold')
            
            # 1.2 Draw Cut Faces (Red Triangle)
            # Check format of deformation_info
            first_val = next(iter(deformation_info.values())) if deformation_info else 0
            if isinstance(first_val, dict):
                cut_faces = []
                
                # Pose Rotation Matrix for calculating cut points
                r_mat = R.from_rotvec(box_pose_to_draw[3:]).as_matrix()
                
                for name, cuts in deformation_info.items():
                    # cuts: {'x': val, 'y': val, 'z': val}
                    if max(cuts.values()) > 1e-5:
                        pt_corner = pts_tf[name]
                        local_pos = self.geo.corners[name]
                        signs = np.sign(local_pos) # [sx, sy, sz]
                        
                        # Calculate 3 cut points
                        cut_pts = []
                        for i, axis_key in enumerate(['x', 'y', 'z']):
                            cut_len = cuts[axis_key]
                            if cut_len > 0:
                                # Edge direction in World: Rot * (-sign * axis_vec)
                                axis_vec = np.zeros(3); axis_vec[i] = 1.0
                                edge_dir = r_mat @ (-signs[i] * axis_vec)
                                p_cut = pt_corner + cut_len * edge_dir
                                cut_pts.append(p_cut)
                            else:
                                cut_pts.append(pt_corner)
                        
                        # 삼각형 추가 (P_x, P_y, P_z)
                        cut_faces.append(cut_pts)
                        
                        # 원래 코너에서 절단점까지 선 그리기 (잘린 엣지 표현)
                        for cp in cut_pts:
                            ax.plot([pt_corner[0], cp[0]], [pt_corner[1], cp[1]], [pt_corner[2], cp[2]], 'r-', lw=2)

                if cut_faces:
                    poly_cut = Poly3DCollection(cut_faces, facecolors='red', linewidths=1, edgecolors='darkred', alpha=0.8)
                    ax.add_collection3d(poly_cut)


        # 2. Draw Points
        for p in points_to_draw:
            c = 'r' # Default red
            m = 'o'
            if p.get('reject_reason'): c = 'gray'; m = 'x' # Outlier
            elif p['type'].startswith('C'): c = 'g'; m = '^' # Corner
            elif p['type'] in ['F','B','L','R','T','D']: c = 'm'; m = 's' # Face
            
            ax.scatter(p['coord'][0], p['coord'][1], p['coord'][2], c=c, marker=m)
            ax.text(p['coord'][0], p['coord'][1], p['coord'][2], p['id'], fontsize=7)

        # 3. Draw Ground Plane (Grid) & World Axes
        # Create a grid on XY plane
        grid_side_length = max(self.geo.width, self.geo.depth, self.geo.height) * 1.5
        grid_res = 2 # A square surface only needs 2x2 points
        x = np.linspace(-grid_side_length / 2, grid_side_length / 2, grid_res)
        y = np.linspace(-grid_side_length / 2, grid_side_length / 2, grid_res)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Transform Grid
        pts_grid = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1) # (N, 3)
        pts_grid_tf = (ground_pose_R @ pts_grid.T).T + ground_pose_t
        
        X_tf = pts_grid_tf[:, 0].reshape(X.shape)
        Y_tf = pts_grid_tf[:, 1].reshape(Y.shape)
        Z_tf = pts_grid_tf[:, 2].reshape(Z.shape)
        
        ax.plot_surface(X_tf, Y_tf, Z_tf, color='gray', alpha=0.5)
        
        # Draw World Origin Axes
        origin = ground_pose_t
        axis_len = grid_side_length * 0.2
        for i, c in enumerate(['r', 'g', 'b']):
            vec = np.zeros(3); vec[i] = axis_len
            axis_end = ground_pose_R @ vec + origin
            ax.plot([origin[0], axis_end[0]], [origin[1], axis_end[1]], [origin[2], axis_end[2]], c+'-', lw=2)
            
        # 4. Draw Penetration Info (Resolution Vector & Normal)
        if penetration_info and penetration_info.get('is_penetrating', False):
            res_vec = penetration_info['resolution_vector']
            norm_vec = penetration_info['normal_vector']
            
            if view_mode == 'box_fixed':
                r_inv = R.from_rotvec(pose[3:]).inv()
                res_vec_draw = r_inv.apply(res_vec)
                norm_vec_draw = r_inv.apply(norm_vec)
                box_center = np.zeros(3)
                ground_origin = ground_pose_t
            else:
                res_vec_draw = res_vec
                norm_vec_draw = norm_vec
                box_center = pose[:3]
                ground_origin = np.zeros(3)
            
            # Resolution Vector (Green, from Box Center)
            ax.quiver(box_center[0], box_center[1], box_center[2], 
                      res_vec_draw[0], res_vec_draw[1], res_vec_draw[2],
                      color='lime', linewidth=2, label='Resolution Vec')
            
            # Ground Normal (Orange, from Ground Origin)
            ax.quiver(ground_origin[0], ground_origin[1], ground_origin[2],
                      norm_vec_draw[0], norm_vec_draw[1], norm_vec_draw[2],
                      color='orange', length=0.5, normalize=True, label='Ground Normal')

        # Axis labels and limits
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        
        # Equal aspect ratio hack
        all_pts_list = list(pts_tf.values())
        all_pts_list.append(ground_pose_t)
        if points_to_draw:
             all_pts_list.extend([p['coord'] for p in points_to_draw])
        
        all_pts = np.array(all_pts_list)
        
        if len(all_pts) > 0:
            max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                                  all_pts[:,1].max()-all_pts[:,1].min(), 
                                  all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
            mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
            mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
            mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
        plt.show()
    
    def show_pyvista(self, pose, points_input, deformation_info=None, penetration_info=None, text_font_size=8, view_mode='world_fixed', title_suffix=""):
        """
        PyVista를 사용하여 박스와 점들을 인터랙티브 3D로 시각화합니다.
        
        Args:
            pose (np.array): 박스 포즈
            points_input (list): 점 데이터 리스트
            deformation_info (dict, optional): 변형 정보
            penetration_info (dict, optional): 침투 정보
            text_font_size (int): 텍스트 라벨 크기 (Default 5 - approx 50% of previous 9)
            view_mode (str): 'world_fixed' (기본) 또는 'box_fixed'
            title_suffix (str): 윈도우 제목에 추가할 텍스트
        """
        if not HAS_PYVISTA:
            print("PyVista not installed. Skipping 3D visualization.")
            return
        
        full_title = f"Box Motion - PyVista {title_suffix}"
        plotter = pv.Plotter(title=full_title)
        
        # Determine transformations based on view_mode
        if view_mode == 'world_fixed':
            box_pose_to_draw = pose
            
            # Ground is at Origin
            ground_transform = np.eye(4)
            
            # Points are already in World Frame
            points_to_draw = points_input
            
            plotter.add_text(f"View: World Fixed (Box Moving)", position='upper_left', font_size=9)
            
        else: # 'box_fixed'
            box_pose_to_draw = np.zeros(6) # Box is at Origin
            
            # Calculate Inverse Transform for World Objects (Ground, Points)
            t = pose[:3]
            r_mat = R.from_rotvec(pose[3:]).as_matrix()
            
            # T_inv = [R^T | -R^T * t]
            r_inv = r_mat.T
            t_inv = -r_inv @ t
            
            ground_transform = np.eye(4)
            ground_transform[:3, :3] = r_inv
            ground_transform[:3, 3] = t_inv
            
            # Transform points to Box Local Frame
            points_to_draw = []
            r_inv_obj = R.from_matrix(r_inv)
            for p in points_input:
                p_new = p.copy()
                p_new['coord'] = r_inv_obj.apply(p['coord'] - t)
                points_to_draw.append(p_new)
                
            plotter.add_text(f"View: Box Fixed (Ground Moving)", position='upper_left', font_size=9)

        # 0. Ground Plane (Added)
        grid_side_length = max(self.geo.width, self.geo.depth, self.geo.height) * 1.5
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=grid_side_length, j_size=grid_side_length)
        ground.transform(ground_transform) # Apply transform (Identity or Inverse)
        plotter.add_mesh(ground, color='gray', opacity=0.5, show_edges=False, label='Ground Plane')
        
        # 1. Origin Box (Wireframe)
        box_org = pv.Box(bounds=(-self.geo.dx, self.geo.dx, -self.geo.dy, self.geo.dy, -self.geo.dz, self.geo.dz))
        plotter.add_mesh(box_org, style='wireframe', color='black', opacity=0.3, label='Origin Box')
        
        # 2. Estimated Box (Transformed or Fixed at Origin)
        box_tf = pv.Box(bounds=(-self.geo.dx, self.geo.dx, -self.geo.dy, self.geo.dy, -self.geo.dz, self.geo.dz))
        t_box = box_pose_to_draw[:3]
        r_mat_box = R.from_rotvec(box_pose_to_draw[3:]).as_matrix()
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = r_mat_box
        trans_mat[:3, 3] = t_box
        box_tf = box_tf.transform(trans_mat)
        
        plotter.add_mesh(box_tf, color='lightblue', opacity=0.7, show_edges=True, label='Estimated Box')
        plotter.add_axes_at_origin(labels_off=True)
        
        # 2.0 Labels (Face & Edge IDs)
        # Face IDs
        face_centers = self._get_face_centers(pose=box_pose_to_draw)
        
        # Use Mapper
        mapper = getattr(self, 'mapper', None)
        if mapper is None:
            mapper = IstaFaceMapper(is_ltl=False, geo=self.geo)

        f_pts = []
        f_labels = []
        for fid, pt in face_centers.items():
            f_pts.append(pt)
            # Create Label: "Face 1 [Top]"
            label = mapper.get_face_label(fid)
            f_labels.append(label)
            
        if f_pts:
            plotter.add_point_labels(np.array(f_pts), f_labels, point_size=0, font_size=text_font_size + 2, text_color='navy', always_visible=True)

        # Edge IDs
        _, pts_tf_dict = self._get_box_edges(pose=box_pose_to_draw)
        edge_centers = self._get_edge_centers(pts_tf_dict)
        e_pts = []
        e_labels = []
        for eid, pt in edge_centers.items():
            e_pts.append(pt)
            e_labels.append(eid)
        if e_pts:
            plotter.add_point_labels(np.array(e_pts), e_labels, point_size=0, font_size=text_font_size, text_color='darkgreen', always_visible=True)

        # 2.1 Draw Cut Faces (Red) if deformation_info exists
        if deformation_info:
            # Check format
            first_val = next(iter(deformation_info.values())) if deformation_info else 0
            if isinstance(first_val, dict):
                verts = []
                faces = []
                idx_counter = 0
                                
                r_mat = R.from_rotvec(box_pose_to_draw[3:]).as_matrix()
                # Get transformed corners                
                _, pts_tf = self._get_box_edges(pose=box_pose_to_draw)
                
                for name, cuts in deformation_info.items():
                    if max(cuts.values()) > 1e-5:
                        pt_corner = pts_tf[name]
                        local_pos = self.geo.corners[name]
                        signs = np.sign(local_pos)
                        
                        cut_pts = []
                        for i, axis_key in enumerate(['x', 'y', 'z']):
                            cut_len = cuts[axis_key]
                            # Edge direction logic same as matplotlib
                            axis_vec = np.zeros(3); axis_vec[i] = 1.0
                            edge_dir = r_mat @ (-signs[i] * axis_vec)
                            p_cut = pt_corner + cut_len * edge_dir
                            cut_pts.append(p_cut)
                        
                        # Add vertices
                        verts.extend(cut_pts)
                        # Add face (3 vertices: 3, i, i+1, i+2)
                        faces.extend([3, idx_counter, idx_counter+1, idx_counter+2])
                        idx_counter += 3
                
                if verts:
                    cut_mesh = pv.PolyData(np.array(verts), np.array(faces))
                    plotter.add_mesh(cut_mesh, color='red', opacity=0.9, label='Cut Faces')

        # 3. Points
        valid_coords = []
        valid_labels = []
        outlier_coords = []
                
        for p in points_to_draw:
            if p.get('reject_reason'):
                outlier_coords.append(p['coord'])
            else:
                valid_coords.append(p['coord'])
                valid_labels.append(p['id'])
                
        if valid_coords:
            valid_coords_np = np.array(valid_coords)
            plotter.add_points(valid_coords_np, color='red', point_size=10, render_points_as_spheres=True, label='Valid Points')
            plotter.add_point_labels(valid_coords_np, valid_labels, point_size=10, font_size=text_font_size)
        if outlier_coords:
            plotter.add_points(np.array(outlier_coords), color='gray', point_size=5, render_points_as_spheres=True, label='Outliers')
            
        # 4. Penetration Info
        if penetration_info and penetration_info.get('is_penetrating', False):
            res_vec = penetration_info['resolution_vector']
            norm_vec = penetration_info['normal_vector']
            if view_mode == 'box_fixed':
                # Transform vectors to local frame
                # ground_transform is the inverse matrix
                r_inv = ground_transform[:3, :3]
                res_vec_draw = r_inv @ res_vec
                norm_vec_draw = r_inv @ norm_vec
                box_center = np.zeros(3)
                ground_origin = ground_transform[:3, 3]
            else:
                res_vec_draw = res_vec
                norm_vec_draw = norm_vec
                box_center = pose[:3]
                ground_origin = np.zeros(3)
            
            plotter.add_arrows(np.array([box_center]), np.array([res_vec_draw]), mag=1.0, color='lime', label='Resolution Vec')
            plotter.add_arrows(np.array([ground_origin]), np.array([norm_vec_draw]), mag=0.5, color='orange', label='Ground Normal')

        plotter.add_legend()
        plotter.show()

def generate_test_points(geo: BoxGeometry, pose_gt, noise_std=0.01, outlier_ratio=0.1):
    """
    테스트용 가상 포인트 데이터를 생성합니다.
    
    Args:
        geo (BoxGeometry): 박스 기하학 정보
        pose_gt (np.array): 정답(Ground Truth) 포즈
        noise_std (float): 점 좌표에 추가할 가우시안 노이즈 표준편차
        outlier_ratio (float): 전체 점 중 이상치(Outlier)의 비율 (0.0 ~ 1.0)
        
    Returns:
        list: 생성된 점 데이터 리스트 [{'id':..., 'type':..., 'coord':...}]
    """
    points = []
    
    # Ground Truth Transform
    t_gt = pose_gt[:3]
    r_gt = R.from_rotvec(pose_gt[3:])
    
    # 1. Generate points on faces
    faces = ['F', 'B', 'L', 'R', 'T', 'D']
    for f in faces:
        # Generate random point on face in local coords
        # Simple logic: fix one axis, random others
        face_def = geo.faces[f]
        pt = np.zeros(3)
        pt[face_def['axis_idx']] = face_def['val']
        
        # Randomize other axes
        for i in range(3):
            if i != face_def['axis_idx']:
                limit = geo.width/2 if i==0 else (geo.depth/2 if i==1 else geo.height/2)
                pt[i] = np.random.uniform(-limit, limit)
        
        # Transform to world
        pt_world = r_gt.apply(pt) + t_gt
        # Add noise
        pt_world += np.random.normal(0, noise_std, 3)
        
        points.append({'id': f'#{len(points)}', 'type': f, 'coord': pt_world})
        
    # 2. Generate Corner points
    for c_name, c_pos in geo.corners.items():
        pt_world = r_gt.apply(c_pos) + t_gt
        pt_world += np.random.normal(0, noise_std, 3)
        points.append({'id': f'#{len(points)}', 'type': c_name, 'coord': pt_world})
        
    # 3. Generate Unknown points (N)
    for _ in range(5):
        # Random surface point
        f = np.random.choice(faces)
        face_def = geo.faces[f]
        pt = np.zeros(3)
        pt[face_def['axis_idx']] = face_def['val']
        for i in range(3):
            if i != face_def['axis_idx']:
                limit = geo.width/2 if i==0 else (geo.depth/2 if i==1 else geo.height/2)
                pt[i] = np.random.uniform(-limit, limit)
        
        pt_world = r_gt.apply(pt) + t_gt + np.random.normal(0, noise_std, 3)
        points.append({'id': f'#{len(points)}', 'type': 'N', 'coord': pt_world})

    # 4. Generate Outliers
    num_outliers = int(len(points) * outlier_ratio)
    for _ in range(num_outliers):
        pt_random = np.random.uniform(-1, 1, 3) # Random space
        points.append({'id': f'#OUT_{len(points)}', 'type': 'N', 'coord': pt_random})
        
    return points

def print_cad_guidance(pose):
    """
    CAD 프로그램에서 박스 또는 바닥면을 이동시키기 위한 가이드 텍스트를 출력합니다.
    
    Args:
        pose (np.array): 추정된 박스 포즈
    """
    t = pose[:3]
    r_vec = pose[3:]
    r = R.from_rotvec(r_vec)
    # Extrinsic rotation (Static Frame): Rotate around Global X, then Global Y, then Global Z
    euler = r.as_euler('XYZ', degrees=True)
    
    print("\n" + "="*60)
    print(" [CAD Transformation Guidance]")
    print("="*60)
    print("1. Move BOX to Estimated Position (World Fixed)")
    print("   (Initial State: Box Center at Origin (0,0,0))")
    print("-" * 50)
    print("   Step 1: Rotate Box around Origin (Extrinsic: X -> Y -> Z)")
    print(f"           Roll (X): {euler[0]:.4f}°")
    print(f"           Pitch(Y): {euler[1]:.4f}°")
    print(f"           Yaw  (Z): {euler[2]:.4f}°")
    print("   Step 2: Translate Box (Global Axes)")
    print(f"           X: {t[0]:.4f}")
    print(f"           Y: {t[1]:.4f}")
    print(f"           Z: {t[2]:.4f}")
    print("="*60)
    
    # Inverse Transform for Plane
    # T_inv = [R^T | -R^T * t]
    r_inv = r.inv()
    euler_inv = r_inv.as_euler('XYZ', degrees=True)
    t_inv = -r_inv.apply(t)
    
    print("2. Move REFERENCE PLANE (e.g., XZ Floor) (Box Fixed)")
    print("   (Scenario: Box is fixed at Origin, Move the Floor instead)")
    print("-" * 50)
    print("   Step 1: Rotate Plane around Origin (Extrinsic: X -> Y -> Z)")
    print(f"           Roll (X): {euler_inv[0]:.4f}°")
    print(f"           Pitch(Y): {euler_inv[1]:.4f}°")
    print(f"           Yaw  (Z): {euler_inv[2]:.4f}°")
    print("   Step 2: Translate Plane (Global Axes)")
    print(f"           X: {t_inv[0]:.4f}")
    print(f"           Y: {t_inv[1]:.4f}")
    print(f"           Z: {t_inv[2]:.4f}")
    print("="*60 + "\n")

def print_gravity_vector_explanation(g_accel, v_init, dt):
    """
    중력 가속도와 초기 속도가 주어졌을 때 바닥 방향 벡터를 계산하는 방법 설명 및 예시 출력.
    
    Args:
        g_accel (np.array): 중력 가속도 벡터
        v_init (np.array): 초기 속도 벡터
        dt (float): 시간 간격
    """
    print("\n" + "="*60)
    print(" [Gravity & Down Vector Calculation Guidance]")
    print("="*60)
    print("Methodology:")
    print("1. The 'Down' direction is defined by the direction of the gravity acceleration vector (g).")
    print("2. In a standard World Frame, g is typically [0, 0, -9.81].")
    print("3. Therefore, the World Down Vector is normalized(g) = [0, 0, -1].")
    print("4. To find the 'Down' direction relative to the Box (Local Frame):")
    print("   v_down_local = Inverse(R_box) * v_down_world")
    print("-" * 50)
    
    # Calculation Example
    v_next = v_init + g_accel * dt
    g_calc = (v_next - v_init) / dt
    norm_g = np.linalg.norm(g_calc)
    down_vec = g_calc / norm_g if norm_g > 0 else np.array([0,0,-1])
    
    print(f"Example Inputs:")
    print(f"  Gravity Accel (g) : {g_accel} m/s^2")
    print(f"  Initial Vel (v0)  : {v_init} m/s")
    print(f"  Time Step (dt)    : {dt} s")
    print(f"Calculation:")
    print(f"  v(t+dt) = v0 + g*dt = {v_next}")
    print(f"  Recovered g = (v(t+dt) - v0) / dt = {g_calc}")
    print(f"  Down Vector = normalize(g) = {down_vec}")
    print("="*60 + "\n")

def run_dimension_estimation_example(initial_geo, viz):
    """
    예제 5: 박스 크기와 자세를 동시에 추정합니다.
    
    Args:
        initial_geo (BoxGeometry): 초기 추정용 박스 기하학 (틀린 크기)
        viz (BoxVisualizer): 시각화 객체
    """
    print("\n" + "="*25 + " 예제 5: 박스 크기 및 자세 동시 추정 " + "="*25)
    # 실제 박스 크기 (GT)
    gt_geo = BoxGeometry(2.5, 1.0, 0.3)
    gt_pose = np.array([0.5, -0.2, 1.0, np.deg2rad(15), np.deg2rad(-10), np.deg2rad(40)])
    points_data = generate_test_points(gt_geo, gt_pose, noise_std=0.01)

    # 추정기 생성 (잘못된 초기 크기 + 크기 추정 옵션 활성화)
    estimator = BoxMotionEstimator(initial_geo, estimate_dims=True)
    est_pose, _, _ = estimator.fit(points_data)

    print(f"GT Dimensions      : W={gt_geo.width:.3f}, D={gt_geo.depth:.3f}, H={gt_geo.height:.3f}")
    print(f"Estimated Dimensions: W={estimator.geo.width:.3f}, D={estimator.geo.depth:.3f}, H={estimator.geo.height:.3f}")
    print(f"Estimated Pose     : {est_pose}")

def run_single_frame_estimation_example(geo, viz):
    """
    예제 1: 단일 프레임에서 박스 위치를 추정합니다.
    
    Args:
        geo (BoxGeometry): 박스 기하학 정보
        viz (BoxVisualizer): 시각화 객체
    """
    print("\n" + "="*25 + " 예제 1: 단일 프레임 추정 " + "="*25)
    estimator = BoxMotionEstimator(geo, outlier_threshold=0.05)
    
    # 임의의 정답 포즈 생성 (이동: [0.1, 0.2, 0.5], 회전: Z축 30도)
    gt_pose = np.array([0.1, 0.2, 0.5, np.deg2rad(10), np.deg2rad(20), np.deg2rad(30)])
    
    # 가상 데이터 생성
    points_data = generate_test_points(geo, gt_pose, noise_std=0.005)
    
    # 추정 수행
    est_pose, valid_pts, rejected_pts = estimator.fit(points_data)
    
    print(f"GT Pose : {gt_pose}")
    print(f"Est Pose: {est_pose}")
    print(f"Rejected Points: {len(rejected_pts)}")
    for p in rejected_pts:
        print(f" - {p['id']}: {p['reject_reason']}")
        
    print_cad_guidance(est_pose)

    # 시각화
    viz.show_matplotlib(est_pose, points_data, title="Estimation", view_mode='world_fixed')
    # World Fixed View
    world_fixed_info = viz.calculate_world_fixed_info(est_pose, print_report=True)
    viz.show_matplotlib(est_pose, points_data, title="Estimation", view_mode='world_fixed', print_guidance=False)
    
    # Box Fixed View: 계산 결과 미리 확인 후 시각화 (중복 출력 방지 위해 print_guidance=False)
    box_fixed_info = viz.calculate_box_fixed_info(est_pose, print_report=True)
    viz.show_matplotlib(est_pose, points_data, title="Estimation", view_mode='box_fixed', print_guidance=False)
    viz.show_pyvista(est_pose, points_data)
    
    # Down Vector 확인
    down_local = estimator.get_down_direction(local=True)
    print(f"Local Down Vector (in Box Frame): {down_local}")
    
    print_gravity_vector_explanation(np.array([0, 0, -9.81]), np.array([0, 0, 0]), 0.01)

    # Screw Axis 분석
    print("\n--- Screw Axis Analysis (from Single Frame) ---")
    # 단일 프레임 추정에서는 시간 정보가 없으므로 속도/각속도는 0으로 계산됩니다.
    screw = estimator.get_screw_axis()
    print(f"Angular Velocity Mag: {screw['w_mag']:.4f} rad/s")
    print(f"Screw Axis: {screw['axis']}")
    print(f"Screw Pitch: {screw['pitch']:.4f}")

def run_ista_drop_pose_example(geo, viz):
    """
    예제 2: ISTA 6A 규격에 따른 낙하 초기 자세를 계산하고 시각화합니다.
    
    Args:
        geo (BoxGeometry): 박스 기하학 정보
        viz (BoxVisualizer): 시각화 객체
    """
    print("\n" + "="*25 + " 예제 2: ISTA 6A 낙하 자세 계산 " + "="*25)
    simulator = ISTA6ASimulator(geo)
    
    # 코너 낙하 자세 계산
    drop_height = 0.8
    corner_target = 'C_DFL'
    pose_corner = simulator.calculate_drop_pose('corner', drop_height, corner_target)
    print(f"Calculated Pose for Corner Drop ({corner_target}, {drop_height}m):")
    print(f" - Translation: {pose_corner[:3]}")
    print(f" - Rotation Vec: {pose_corner[3:]}")
    
    print_cad_guidance(pose_corner)
    
    print_gravity_vector_explanation(np.array([0, 0, -9.81]), np.array([0, 0, 0]), 0.01)
    
    # 시각화
    pts_corner = generate_test_points(geo, pose_corner, noise_std=0.0, outlier_ratio=0.0)
    viz.show_matplotlib(pose_corner, pts_corner, title=f"ISTA 6A ({corner_target})", view_mode='world_fixed')
    
    # World Fixed View
    world_fixed_info = viz.calculate_world_fixed_info(pose_corner, print_report=True)
    viz.show_matplotlib(pose_corner, pts_corner, title=f"ISTA 6A ({corner_target})", view_mode='world_fixed', print_guidance=False)
    
    # Box Fixed View: 계산 결과 미리 확인 후 시각화
    box_fixed_info = viz.calculate_box_fixed_info(pose_corner, print_report=True)
    viz.show_matplotlib(pose_corner, pts_corner, title=f"ISTA 6A ({corner_target})", view_mode='box_fixed', print_guidance=False)
    viz.show_pyvista(pose_corner, pts_corner)

def run_time_series_tracking_example(geo):
    """
    예제 3: 시간에 따라 변화하는 점들을 추적하고 운동학적 데이터를 분석합니다.
    
    Args:
        geo (BoxGeometry): 박스 기하학 정보
    """
    print("\n" + "="*25 + " 예제 3: 시계열 추적 및 운동학 분석 " + "="*25)
    # 시나리오: 박스를 위로 던지면서 회전시킴 (Toss with Spin)
    # Z: 포물선 운동, X: 등속 운동, 회전: Y축 등각속도
    duration = 1.0
    dt = 0.02
    times_track = np.arange(0, duration, dt)
    
    # 새로운 추적을 위해 Estimator 리셋
    estimator = BoxMotionEstimator(geo)
    
    # 중력 벡터 설명 출력
    g_vec = np.array([0, 0, -9.81])
    print_gravity_vector_explanation(g_vec, np.array([0.5, 0, 2.0]), dt)

    print("Generating trajectory and estimating...")
    for t in times_track:
        # Ground Truth Trajectory
        z = 0.5 + 2.0 * t - 0.5 * 9.81 * t**2 # v0_z = 2.0
        x = 0.5 * t # v_x = 0.5
        y = 0.0
        
        # Rotation: Spin around Y axis (pi rad/s)
        angle = np.pi * t
        r_gt = R.from_euler('y', angle)
        pose_gt = np.concatenate([[x, y, z], r_gt.as_rotvec()])
        
        # Generate Noisy Points
        pts = generate_test_points(geo, pose_gt, noise_std=0.002, outlier_ratio=0.05)
        
        # Estimate
        estimator.fit(pts, time_stamp=t, continuous=True)

    # Analyze Kinematics
    hist_t = [h['t'] for h in estimator.history]
    hist_v = np.array([h['v_lin'] for h in estimator.history])
    hist_a = np.array([h['a_lin'] for h in estimator.history])
    hist_w = np.array([h['v_ang'] for h in estimator.history])
    
    corner_name = 'C_TFR'
    c_data = estimator.get_corner_kinematics(corner_name)
    c_vel = np.array(c_data['vel'])
    c_acc = np.array(c_data['accel'])
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Kinematics Analysis (Toss & Spin Trajectory)")
    
    axes[0,0].plot(hist_t, hist_v[:,0], label='Vx'); axes[0,0].plot(hist_t, hist_v[:,2], label='Vz')
    axes[0,0].set_title("CoM Linear Velocity"); axes[0,0].legend(); axes[0,0].grid()
    axes[0,1].plot(hist_t, hist_w[:,1], 'g', label='Wy (Pitch Rate)')
    axes[0,1].set_title("CoM Angular Velocity"); axes[0,1].legend(); axes[0,1].grid()
    axes[1,0].plot(hist_t, c_vel[:,0], label='Vx'); axes[1,0].plot(hist_t, c_vel[:,2], label='Vz')
    axes[1,0].set_title(f"Corner ({corner_name}) Velocity"); axes[1,0].legend(); axes[1,0].grid()
    axes[1,1].plot(hist_t, c_acc[:,0], label='Ax'); axes[1,1].plot(hist_t, c_acc[:,2], label='Az')
    axes[1,1].set_title(f"Corner ({corner_name}) Acceleration"); axes[1,1].legend(); axes[1,1].grid()
    
    plt.tight_layout()
    plt.show()
    print("Kinematics analysis plots displayed.")

def run_deformation_tracking_example(geo, viz):
    """
    예제 4: 지면 침투 및 변형 추적 기능 시연 (ISTA Corner Drop 시나리오).
    
    Args:
        geo (BoxGeometry): 박스 기하학 정보
        viz (BoxVisualizer): 시각화 객체
    """
    print("\n" + "="*25 + " 예제 4: 지면 침투 및 변형 추적 (ISTA Corner Drop) " + "="*25)
    
    # 변형 추적 옵션 활성화
    estimator = BoxMotionEstimator(geo, track_deformation=True)
    simulator = ISTA6ASimulator(geo)
    
    print_gravity_vector_explanation(np.array([0, 0, -9.81]), np.array([0, 0, -1.0]), 0.02)
    
    # 시나리오: ISTA 코너 낙하 (C_DFL) 자세로 낙하하여 지면을 약 40mm 침투
    corner_target = 'C_DFL'
    # 1. 초기 자세 계산 (높이는 0으로 설정하여 회전만 가져옴)
    pose_ref = simulator.calculate_drop_pose('corner', 0.0, corner_target)
    r_init = R.from_rotvec(pose_ref[3:])
    
    # 2. 최하단점 기준 Z축 궤적 생성
    # 박스 회전 시 최하단점의 로컬 Z값 (음수)
    corners = geo.get_vertices()
    rotated_corners = r_init.apply(corners)
    min_z_local = np.min(rotated_corners[:, 2])
    
    # 궤적: 0.1m에서 시작하여 -0.04m(40mm 침투)까지 하강
    z_start = 0.1
    z_end = -0.04
    steps = 20
    times = np.linspace(0, 1.0, steps)
    z_lowest_traj = np.linspace(z_start, z_end, steps)
    
    print(f"Simulating ISTA Corner Drop ({corner_target}) with penetration to {z_end*1000:.1f}mm...")
    
    for t, z_low in zip(times, z_lowest_traj):
        # CoM의 Z 좌표 계산 (최하단점 Z - 로컬 최하단점 Z)
        z_com = z_low - min_z_local
        
        # Pose 생성 (회전은 고정, Z만 변경)
        pose_gt = np.concatenate([[0, 0, z_com], r_init.as_rotvec()])
        
        # 포인트 생성 및 추정
        pts = generate_test_points(geo, pose_gt, noise_std=0.001, outlier_ratio=0.0)
        estimator.fit(pts, time_stamp=t, continuous=True)
        
    # 추적 완료 후 변형량 확인
    deformations = estimator.get_corner_deformations()
    print("\n[Result] Max Cut Lengths per Corner (x, y, z):")
    for c_name, cuts in deformations.items():
        max_cut = max(cuts.values())
        if max_cut > 0:
            print(f" - {c_name}: X={cuts['x']*1000:.1f}, Y={cuts['y']*1000:.1f}, Z={cuts['z']*1000:.1f} mm")
            
    # 침투 검사 및 해결 안내 기능 호출
    pen_info = estimator.check_ground_penetration()

    # 마지막 상태 시각화 (변형 정보 포함)
    last_pose = estimator.current_pose
    pts_last = generate_test_points(geo, last_pose, noise_std=0.0)
    
    print("\nVisualizing result with deformation markers...")
    # Matplotlib Visualization
    viz.show_matplotlib(last_pose, pts_last, title="Deformation Tracking Result", deformation_info=deformations, penetration_info=pen_info)
    
    # PyVista Visualization
    print("Attempting PyVista visualization...")
    viz.show_pyvista(last_pose, pts_last, deformation_info=deformations, penetration_info=pen_info)
    
    # CSV 내보내기
    estimator.export_to_csv("deformation_history.csv")

    run_deformation_tracking_example(geo, viz)
    run_dimension_estimation_example(geo, viz)

def verify_rotation_process(geo, final_pose):
    """
    Shows the rotation and translation process in steps using PyVista subplots (2x3).
    Step 1: Initial (Identity)
    Step 2: Rotate X
    Step 3: Rotate X -> Y
    Step 4: Rotate X -> Y -> Z (Final Rotation)
    Step 5: Translate (Final Pose)
    """
    if not HAS_PYVISTA:
        print("PyVista is required for this verification.")
        return

    # Decompose Final Rotation into Euler Angles (XYZ)
    r_final = R.from_rotvec(final_pose[3:])
    euler_angles = r_final.as_euler('xyz', degrees=False) # rad
    t_final = final_pose[:3]

    plotter = pv.Plotter(shape=(2, 3), title="Rotation Process Verification")

    steps = [
        ("1. Initial State", np.zeros(3), np.eye(3)),
        ("2. Rotate X", np.zeros(3), R.from_euler('x', euler_angles[0]).as_matrix()),
        ("3. Rotate X -> Y", np.zeros(3), R.from_euler('xy', euler_angles[:2]).as_matrix()),
        ("4. Rotate X -> Y -> Z", np.zeros(3), r_final.as_matrix()),
        ("5. Final Pose (Translate)", t_final, r_final.as_matrix())
    ]

    for i, (title, t, r_mat) in enumerate(steps):
        row = i // 3
        col = i % 3
        plotter.subplot(row, col)
        plotter.add_text(title, font_size=10)
        
        # Draw Axes at World Origin
        plotter.add_axes_at_origin(labels_off=True)
        
        # Draw Box
        box = pv.Box(bounds=(-geo.dx, geo.dx, -geo.dy, geo.dy, -geo.dz, geo.dz))
        
        # Apply Transform
        # T = [R | t]
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = r_mat
        trans_mat[:3, 3] = t
        
        box_tf = box.transform(trans_mat)
        
        plotter.add_mesh(box_tf, color='lightblue', opacity=0.8, show_edges=True)
        
        # Draw wireframe of initial box for reference in rotation steps
        if i < 4:
             box_init = pv.Box(bounds=(-geo.dx, geo.dx, -geo.dy, geo.dy, -geo.dz, geo.dz))
             plotter.add_mesh(box_init, style='wireframe', color='gray', opacity=0.3)

    plotter.show()

# --- Reference Data (Samsung TV Models 2024/2025) ---
# --- Reference Data (Samsung TV Models) ---
REFERENCE_MODELS = [
    # S95F (OLED)
    {'name': 'S95F(OLED)', 'inch': 55, 'pkg_size': '1625 x 935 x 170', 'pkg_m': 39.8, 'set_w_std_size': '1444 x 894 x 268', 'set_w_std_m': 29.0, 'set_wo_std_size': '1444 x 829 x 11.0', 'set_wo_std_m': 18.9, 'stand_base': '360 x 268'},
    {'name': 'S95F(OLED)', 'inch': 77, 'pkg_size': '1893 x 1153 x 185', 'pkg_m': 54.0, 'set_w_std_size': '1717 x 1048 x 286', 'set_w_std_m': 10.0, 'set_wo_std_size': '1717 x 984 x 11.2', 'set_wo_std_m': 20.2, 'stand_base': '360 x 286'},
    # S90F (OLED)
    {'name': 'S90F(OLED)', 'inch': 48, 'pkg_size': '1197 x 770 x 143', 'pkg_m': 17.8, 'set_w_std_size': '1069 x 683 x 210', 'set_w_std_m': 13.2, 'set_wo_std_size': '1069 x 620 x 39.6', 'set_wo_std_m': 12.3, 'stand_base': '584 x 210'},
    {'name': 'S90F(OLED)', 'inch': 55, 'pkg_size': '1385 x 832 x 153', 'pkg_m': 22.9, 'set_w_std_size': '1225 x 774 x 265', 'set_w_std_m': 17.3, 'set_wo_std_size': '1225 x 709 x 39.9', 'set_wo_std_m': 16.0, 'stand_base': '366 x 265'},
    {'name': 'S90F(OLED)', 'inch': 65, 'pkg_size': '1617 x 950 x 160', 'pkg_m': 29.9, 'set_w_std_size': '1444 x 897 x 265', 'set_w_std_m': 22.5, 'set_wo_std_size': '1444 x 832 x 39.9', 'set_wo_std_m': 21.2, 'stand_base': '366 x 265'},
    {'name': 'S90F(OLED)', 'inch': 77, 'pkg_size': '1885 x 1154 x 180', 'pkg_m': 45.9, 'set_w_std_size': '1719 x 1059 x 359', 'set_w_std_m': 36.6, 'set_wo_std_size': '1719 x 988 x 44.9', 'set_wo_std_m': 34.8, 'stand_base': '366 x 359'},
    # S85F (OLED)
    {'name': 'S85F(OLED)', 'inch': 55, 'pkg_size': '1383 x 832 x 147', 'pkg_m': 17.0, 'set_w_std_size': '1225 x 765 x 235', 'set_w_std_m': 12.6, 'set_wo_std_size': '1225 x 706 x 33.9', 'set_wo_std_m': 12.2, 'stand_base': '897 x 235'},
    {'name': 'S85F(OLED)', 'inch': 65, 'pkg_size': '1617 x 950 x 160', 'pkg_m': 22.7, 'set_w_std_size': '1444 x 897 x 263', 'set_w_std_m': 16.5, 'set_wo_std_size': '1444 x 829 x 33.9', 'set_wo_std_m': 16.1, 'stand_base': '953 x 263'},
    # QN90F (Neo QLED)
    {'name': 'QN90F(Neo QLED)', 'inch': 43, 'pkg_size': '1163 x 668 x 140', 'pkg_m': 17.5, 'set_w_std_size': '960 x 620 x 221', 'set_w_std_m': 13.4, 'set_wo_std_size': '960 x 559 x 28', 'set_wo_std_m': 9.4, 'stand_base': '518 x 221'},
    {'name': 'QN90F(Neo QLED)', 'inch': 50, 'pkg_size': '1318 x 767 x 140', 'pkg_m': 21.5, 'set_w_std_size': '1114 x 705 x 220', 'set_w_std_m': 17.6, 'set_wo_std_size': '1114 x 645 x 28', 'set_wo_std_m': 13.5, 'stand_base': '519 x 220'},
    {'name': 'QN90F(Neo QLED)', 'inch': 55, 'pkg_size': '1448 x 826 x 170', 'pkg_m': 26.2, 'set_w_std_size': '1227 x 767 x 236', 'set_w_std_m': 20.6, 'set_wo_std_size': '1227 x 706 x 28', 'set_wo_std_m': 17.7, 'stand_base': '369 x 236'},
    {'name': 'QN90F(Neo QLED)', 'inch': 65, 'pkg_size': '1623 x 935 x 185', 'pkg_m': 35.4, 'set_w_std_size': '1445 x 892 x 272', 'set_w_std_m': 27.5, 'set_wo_std_size': '1445 x 828 x 28', 'set_wo_std_m': 24.2, 'stand_base': '391 x 272'},
    {'name': 'QN90F(Neo QLED)', 'inch': 75, 'pkg_size': '1885 x 1118 x 191', 'pkg_m': 50.2, 'set_w_std_size': '1669 x 1016 x 302', 'set_w_std_m': 39.7, 'set_wo_std_size': '1669 x 958 x 28', 'set_wo_std_m': 34.1, 'stand_base': '417 x 302'},
    {'name': 'QN90F(Neo QLED)', 'inch': 85, 'pkg_size': '2141 x 1245 x 231', 'pkg_m': 63.3, 'set_w_std_size': '1892 x 1143 x 320', 'set_w_std_m': 49.9, 'set_wo_std_size': '1892 x 1082 x 28', 'set_wo_std_m': 43.5, 'stand_base': '417 x 320'},
    {'name': 'QN90F(Neo QLED)', 'inch': 98, 'pkg_size': '2370 x 1420 x 274', 'pkg_m': 92.7, 'set_w_std_size': '2185 x 1306 x 365', 'set_w_std_m': 70.6, 'set_wo_std_size': '2185 x 1249 x 31', 'set_wo_std_m': 61.4, 'stand_base': '479 x 365'},
    # QN80F (Neo QLED)
    {'name': 'QN80F(Neo QLED)', 'inch': 50, 'pkg_size': '1241 x 744 x 142', 'pkg_m': 17.0, 'set_w_std_size': '1114 x 699 x 239', 'set_w_std_m': 13.3, 'set_wo_std_size': '1114 x 644 x 48', 'set_wo_std_m': 12.9, 'stand_base': '967 x 239'},
    {'name': 'QN80F(Neo QLED)', 'inch': 55, 'pkg_size': '1352 x 813 x 148', 'pkg_m': 22.0, 'set_w_std_size': '1228 x 764 x 247', 'set_w_std_m': 17.2, 'set_wo_std_size': '1228 x 706 x 47', 'set_wo_std_m': 16.4, 'stand_base': '264 x 247'},
    {'name': 'QN80F(Neo QLED)', 'inch': 65, 'pkg_size': '1593 x 930 x 158', 'pkg_m': 30.9, 'set_w_std_size': '1447 x 886 x 279', 'set_w_std_m': 23.7, 'set_wo_std_size': '1447 x 829 x 47', 'set_wo_std_m': 22.7, 'stand_base': '314 x 279'},
    {'name': 'QN80F(Neo QLED)', 'inch': 75, 'pkg_size': '1822 x 1075 x 169', 'pkg_m': 42.2, 'set_w_std_size': '1671 x 1016 x 332', 'set_w_std_m': 33.5, 'set_wo_std_size': '1671 x 958 x 47', 'set_wo_std_m': 32.2, 'stand_base': '355 x 332'},
    {'name': 'QN80F(Neo QLED)', 'inch': 85, 'pkg_size': '2056 x 1200 x 178', 'pkg_m': 54.1, 'set_w_std_size': '1893 x 1138 x 367', 'set_w_std_m': 42.2, 'set_wo_std_size': '1893 x 1083 x 48', 'set_wo_std_m': 40.6, 'stand_base': '375 x 367'},
    # QN70F (Neo QLED)
    {'name': 'QN70F(Neo QLED)', 'inch': 55, 'pkg_size': '1369 x 819 x 142', 'pkg_m': 19.7, 'set_w_std_size': '1233 x 766 x 247', 'set_w_std_m': 15.0, 'set_wo_std_size': '1233 x 709 x 26', 'set_wo_std_m': 14.2, 'stand_base': '264 x 247'},
    {'name': 'QN70F(Neo QLED)', 'inch': 65, 'pkg_size': '1597 x 933 x 148', 'pkg_m': 29.1, 'set_w_std_size': '1452 x 890 x 279', 'set_w_std_m': 21.8, 'set_wo_std_size': '1452 x 832 x 26', 'set_wo_std_m': 20.8, 'stand_base': '314 x 279'},
    {'name': 'QN70F(Neo QLED)', 'inch': 75, 'pkg_size': '1815 x 1080 x 164', 'pkg_m': 40.3, 'set_w_std_size': '1678 x 1016 x 332', 'set_w_std_m': 30.9, 'set_wo_std_size': '1678 x 961 x 27', 'set_wo_std_m': 29.6, 'stand_base': '355 x 332'},
    {'name': 'QN70F(Neo QLED)', 'inch': 85, 'pkg_size': '2059 x 1200 x 171', 'pkg_m': 53.9, 'set_w_std_size': '1902 x 1144 x 367', 'set_w_std_m': 42.3, 'set_wo_std_size': '1902 x 1087 x 27', 'set_wo_std_m': 40.7, 'stand_base': '375 x 367'},
    # U8000F (Crystal UHD)
    {'name': 'U8000F(Crystal UHD)', 'inch': 43, 'pkg_size': '1070 x 650 x 123', 'pkg_m': 9.0, 'set_w_std_size': '958 x 609 x 157', 'set_w_std_m': 6.6, 'set_wo_std_size': '958 x 559 x 76', 'set_wo_std_m': 6.4, 'stand_base': '613 x 157'},
    {'name': 'U8000F(Crystal UHD)', 'inch': 50, 'pkg_size': '1228 x 743 x 127', 'pkg_m': 12.0, 'set_w_std_size': '1111 x 695 x 199', 'set_w_std_m': 8.3, 'set_wo_std_size': '1111 x 644 x 76', 'set_wo_std_m': 8.0, 'stand_base': '747 x 199'},
    {'name': 'U8000F(Crystal UHD)', 'inch': 55, 'pkg_size': '1354 x 810 x 127', 'pkg_m': 14.4, 'set_w_std_size': '1225 x 759 x 199', 'set_w_std_m': 9.9, 'set_wo_std_size': '1225 x 708 x 77', 'set_wo_std_m': 9.6, 'stand_base': '813 x 199'},
    {'name': 'U8000F(Crystal UHD)', 'inch': 58, 'pkg_size': '1421 x 846 x 135', 'pkg_m': 17.5, 'set_w_std_size': '1286 x 800 x 222', 'set_w_std_m': 12.3, 'set_wo_std_size': '1286 x 749 x 77', 'set_wo_std_m': 12.0, 'stand_base': '883 x 222'},
    {'name': 'U8000F(Crystal UHD)', 'inch': 65, 'pkg_size': '1578 x 930 x 142', 'pkg_m': 21.0, 'set_w_std_size': '1444 x 882 x 222', 'set_w_std_m': 14.5, 'set_wo_std_size': '1444 x 831 x 77', 'set_wo_std_m': 14.2, 'stand_base': '1005 x 222'},
    {'name': 'U8000F(Crystal UHD)', 'inch': 70, 'pkg_size': '1755 x 1056 x 164', 'pkg_m': 28.5, 'set_w_std_size': '1567 x 926 x 280', 'set_w_std_m': 19.4, 'set_wo_std_size': '1567 x 876 x 77', 'set_wo_std_m': 19.0, 'stand_base': '1123 x 280'},
    {'name': 'U8000F(Crystal UHD)', 'inch': 75, 'pkg_size': '1820 x 1100 x 164', 'pkg_m': 31.0, 'set_w_std_size': '1668 x 1006 x 280', 'set_w_std_m': 22.8, 'set_wo_std_size': '1668 x 958 x 77', 'set_wo_std_m': 22.4, 'stand_base': '1250 x 280'},
    {'name': 'U8000F(Crystal UHD)', 'inch': 85, 'pkg_size': '2075 x 1200 x 171', 'pkg_m': 40.7, 'set_w_std_size': '1890 x 1133 x 326', 'set_w_std_m': 29.2, 'set_wo_std_size': '1890 x 1084 x 77', 'set_wo_std_m': 28.7, 'stand_base': '1411 x 326'},
    # Frame Pro (LS03FW)
    {'name': 'Frame Pro(LS03FW)', 'inch': 65, 'pkg_size': '1598 x 933 x 164', 'pkg_m': 32.3, 'set_w_std_size': '1457 x 867 x 257', 'set_w_std_m': 22.0, 'set_wo_std_size': '1457 x 837 x 25', 'set_wo_std_m': 21.8, 'stand_base': '1042 x 257'},
    {'name': 'Frame Pro(LS03FW)', 'inch': 75, 'pkg_size': '1834 x 1082 x 171', 'pkg_m': 40.5, 'set_w_std_size': '1685 x 995 x 320', 'set_w_std_m': 30.9, 'set_wo_std_size': '1685 x 965 x 27', 'set_wo_std_m': 30.7, 'stand_base': '1244 x 320'},
    {'name': 'Frame Pro(LS03FW)', 'inch': 85, 'pkg_size': '2068 x 1200 x 178', 'pkg_m': 53.7, 'set_w_std_size': '1904 x 1127 x 369', 'set_w_std_m': 41.1, 'set_wo_std_size': '1904 x 1091 x 27', 'set_wo_std_m': 40.8, 'stand_base': '1410 x 369'}
]

def run_ista_6a_test_suite():
    """
    ISTA 6A 규격에 따른 낙하 시험 시뮬레이션 (Parcel & LTL 지원).
    Tkinter UI를 통해 시험 규격(Parcel/LTL)과 시각화 옵션을 선택하여 실행합니다.
    """
    import tkinter as tk
    from tkinter import ttk

    # 1. 대상물 정의 (기본값)
    # Default Geometry (TV Size for LTL, smaller for Parcel usually, but using one for demo)
    tv_w, tv_d, tv_h = 1.4, 0.2, 0.8
    tv_geo = BoxGeometry(tv_w, tv_d, tv_h)
    tv_viz = BoxVisualizer(tv_geo)
    simulator = ISTA6ASimulator(tv_geo)
    
    print("[Face Identification Guide - ISTA 6]")
    print("  1: Top, 2: Bottom, 3: Front, 4: Back, 5: Right, 6: Left")
    
    # --- UI Setup ---
    root = tk.Tk()
    root.title("ISTA 6A Test Suite Runner")

    # 1. Product Configuration Input
    config_frame = ttk.LabelFrame(root, text="Product & Shipping Configuration")
    config_frame.pack(padx=10, pady=5, fill="x")
    
    # Grid Setup
    config_frame.columnconfigure(1, weight=1)
    
    # 1-0. Product Type Selection
    prod_frame = ttk.Frame(config_frame)
    prod_frame.grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
    
    prod_var = tk.StringVar(value='TV/Monitor')
    ttk.Label(prod_frame, text="Product Type:").pack(side='left', padx=5)
    ttk.Radiobutton(prod_frame, text="TV/Monitor", variable=prod_var, value='TV/Monitor').pack(side='left')
    ttk.Radiobutton(prod_frame, text="General", variable=prod_var, value='General').pack(side='left')

    # Help Button - Sophisticated Style
    def show_guide():
        # Singleton Check
        if hasattr(show_guide, 'win') and show_guide.win and show_guide.win.winfo_exists():
            show_guide.win.lift()
            return
        
        help_win = tk.Toplevel(root)
        show_guide.win = help_win # Save reference
        help_win.title("ISTA 6-Amazon Test Protocol Guide")
        help_win.geometry("730x700")
        
        # Center
        root.update_idletasks()
        x = root.winfo_rootx() + (root.winfo_width() // 2) - (600 // 2)
        y = root.winfo_rooty() + (root.winfo_height() // 2) - (800 // 2)
        help_win.geometry(f"+{x}+{y}")
        
        # Container Frame
        text_frame = tk.Frame(help_win)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        txt = tk.Text(text_frame, wrap=tk.WORD, padx=20, pady=20, bg="white", borderwidth=0, font=("Segoe UI", 10))
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        txt.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=txt.yview)
        
        txt.tag_config("h1", font=("Segoe UI", 14, "bold"), foreground="#2c3e50", spacing1=10, spacing3=10)
        txt.tag_config("h2", font=("Segoe UI", 12, "bold"), foreground="#2980b9", spacing1=15, spacing3=5)
        txt.tag_config("bold", font=("Segoe UI", 10, "bold"), foreground="#333333")
        txt.tag_config("bullet", lmargin1=15, lmargin2=25)
        txt.tag_config("normal", spacing1=2)
        txt.tag_config("table", font=("Consolas", 9), foreground="#444444", background="#f9f9f9", lmargin1=20)

        help_md_text = """# ISTA 6-Amazon Classification

The core criteria for determining the Type are **Delivery Method (Parcel vs LTL)**, **Weight**, and **Girth**.

## 1. Type Classification Table

| Type | Method | Weight Limit        | Girth Limit      | Description              |
|------|--------|---------------------|------------------|--------------------------|
| A    | Parcel | < 70lb (32kg)       | <= 165" (4191mm) | Parcel, Standard (Small) |
| B    | Parcel | 70-150lb (32-68kg)  | <= 165" (4191mm) | Parcel, Standard (Medium)|
| C    | Parcel | >= 150lb (68kg)     | <= 165" (4191mm) | Parcel, Standard (Heavy) |
| D    | LTL    | < 100lb (45kg)      | > 165" (4191mm)  | LTL, Standard (Small)    |
| E    | LTL    | >= 100lb (45kg)     | > 165" (4191mm)  | LTL, Standard (Large)    |
| F    | LTL    | < 150lb (68kg)      | > 165" (+Pallet) | LTL, Palletized          |
| G    | LTL    | < 150lb (68kg)      | > 165" (+TV)     | **TV/Monitor** (Clamp)   |
| H    | LTL    | >= 150lb (68kg)     | > 165" (+TV)     | **TV/Monitor** (Large)   |

*(Girth = Length + 2*Width + 2*Height)*

## 2. Method Details

**Parcel (Type A, B, C)**
Environment: Conveyors, Sortation Hubs, Delivery Vans.
Key Tests: **Free Fall Drop** (17+ drops), Vibration.
Orientation: Based on Longest Dimension.

**LTL (Types D-H)**
Environment: Forklifts, Pallet Jacks, Trucks.
Key Tests: **Rotational Drop**, Incline Impact, Tip-Over.
Orientation: Based on Shipping Configuration (Normal Up).

## 3. TV / Monitor Specifics

For large TVs (e.g., 85-inch), Girth almost always exceeds 165" (4191mm), forcing **LTL** classification.

- **Standard (Parcel)**: Only if Girth <= 165". Extremely rare.
- **Type G (LTL)**: Weight < 150lb (68kg). Standard for 85" LED/OLED.
- **Type H (LTL)**: Weight >= 150lb (68kg). Heavy displays.

**Handling Selection**
- **Palletized**: Products shipped on a pallet (Default for TV).
- **Standard (Floor)**: If product is floor-loaded. Tests Type D/E.
"""

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
            elif line.startswith("|"):
                 # Basic Table Formatting (Monospace)
                 # Ensure alignment by using formatted string if possible, or just dump it with mono font
                 # Let's align columns manually for better display if needed, but Consolas handles fixed width well if input is aligned.
                 # The input table above is reasonably aligned in source.
                 txt.insert(tk.END, line + "\n", "table")
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

    btn_help = ttk.Button(prod_frame, text="?", width=3, command=show_guide)
    btn_help.pack(side='left', padx=10)

    # 1-1. Specs (Weight/Dims) - Moved to Row 1
    spec_frame = ttk.Frame(config_frame)
    spec_frame.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
    
    ttk.Label(spec_frame, text="Weight (kg):").pack(side='left')
    mass_var = tk.DoubleVar(value=15.0)
    ttk.Entry(spec_frame, textvariable=mass_var, width=8).pack(side='left', padx=5)
    
    ttk.Label(spec_frame, text="Dims (W x H x D) [mm]:").pack(side='left', padx=10)
    w_var = tk.DoubleVar(value=1400.0)
    d_var = tk.DoubleVar(value=200.0)
    h_var = tk.DoubleVar(value=800.0)
    
    # Increased width 5 -> 8 (approx 1.5x)
    # Order: W x H x D (Width, Height, Thickness)
    ttk.Entry(spec_frame, textvariable=w_var, width=8).pack(side='left')
    ttk.Label(spec_frame, text="x").pack(side='left')
    ttk.Entry(spec_frame, textvariable=h_var, width=8).pack(side='left')
    ttk.Label(spec_frame, text="x").pack(side='left')
    ttk.Entry(spec_frame, textvariable=d_var, width=8).pack(side='left')

    # 1-2. Method Selection - Moved to Row 2
    method_frame = ttk.Frame(config_frame)
    method_frame.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=2)
    
    ship_var = tk.StringVar(value='Parcel')
    hand_var = tk.StringVar(value='Standard')
    
    ttk.Label(method_frame, text="Detected Method:").pack(side='left', padx=5)
    rb_parcel = ttk.Radiobutton(method_frame, text="Parcel", variable=ship_var, value='Parcel')
    rb_parcel.pack(side='left')
    rb_ltl = ttk.Radiobutton(method_frame, text="LTL", variable=ship_var, value='LTL')
    rb_ltl.pack(side='left')
    
    ttk.Label(method_frame, text="|  Allotted Handling:").pack(side='left', padx=10)
    
    # Handling Radios
    ttk.Radiobutton(method_frame, text="Standard", variable=hand_var, value='Standard').pack(side='left')
    ttk.Radiobutton(method_frame, text="Palletized", variable=hand_var, value='Palletized').pack(side='left')

    # 1-3. Info Output
    info_frame = ttk.Frame(config_frame)
    info_frame.grid(row=3, column=0, columnspan=4, sticky='ew', padx=5, pady=5)
    
    # Use Grid inside Info Frame
    info_frame.columnconfigure(1, weight=1) # Expand column 1 (Girth) or make Reason expand?
    # Let's make column 0 expand to avoid wrapping if possible
    info_frame.columnconfigure(0, weight=1)
    
    # Type Label - ensure sufficient width or let it expand
    info_label = ttk.Label(info_frame, text="Type: --", foreground="blue", font=('Arial', 10, 'bold'))
    info_label.grid(row=0, column=0, sticky='w')
    
    # Reason Label (Target: Single line if possible, or long wrap)
    # Increased wraplength 350 -> 800 to prevent premature wrapping
    lbl_reason = ttk.Label(info_frame, text="", foreground="gray", wraplength=800)
    lbl_reason.grid(row=1, column=0, columnspan=2, sticky='w', padx=10)
    
    # Girth Label on the right of Type
    girth_label = ttk.Label(info_frame, text="(Girth: -- )", foreground="gray")
    girth_label.grid(row=0, column=1, sticky='e')

    # 2. Sequence Display
    seq_frame = ttk.LabelFrame(root, text="Test Sequence Preview")
    seq_frame.pack(padx=10, pady=5, fill="both", expand=True)
    
    # Selection Buttons
    sel_btn_frame = ttk.Frame(seq_frame)
    sel_btn_frame.pack(fill='x', padx=5, pady=2)
    
    def select_all():
        for item in tree.get_children():
            tree.selection_add(item)
            
    def deselect_all():
        tree.selection_remove(tree.selection())
        
    ttk.Button(sel_btn_frame, text="Select All", command=select_all).pack(side='left', padx=2)
    ttk.Button(sel_btn_frame, text="Deselect All", command=deselect_all).pack(side='left', padx=2)
    
    # Treeview
    cols = ('No', 'Name', 'Description', 'Height')
    tree = ttk.Treeview(seq_frame, columns=cols, show='headings', selectmode='extended', height=8)
    
    tree.heading('No', text='#')
    tree.column('No', width=30, anchor='center')
    tree.heading('Name', text='Name')
    tree.column('Name', width=120, anchor='w')
    tree.heading('Description', text='Description')
    tree.column('Description', width=200, anchor='w')
    tree.heading('Height', text='Height (mm)')
    tree.column('Height', width=80, anchor='center')
    
    tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    
    scrollbar = ttk.Scrollbar(seq_frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.config(yscrollcommand=scrollbar.set)
    
    # Dynamic Update Function
    current_sequence_data = []
    
    # Flags to prevent recursion loops if needed? Trace triggers are tricky.
    # We will compute auto-values, set them (which triggers trace), but we should handle it.
    # Actually, simpler: The Trace calls this. We set vars. Trace calls this again.
    # If we set vars to SAME value, no trace?
    # tk variable trace usually triggers on write.
    # We can temporarily disable trace or check values.
    # Or just let it run twice (fast enough).
    
    def refresh_display(event=None):
        nonlocal current_sequence_data
        try:
            m = mass_var.get()
            w = w_var.get()
            d_val = d_var.get()
            h = h_var.get()
            p_type = prod_var.get()
            final_ship = ship_var.get()
            final_hand = hand_var.get()
        except tk.TclError:
            return 
            
        # Update Geometry
        tv_geo.width = w
        tv_geo.depth = d_val
        tv_geo.height = h
        tv_geo.dx, tv_geo.dy, tv_geo.dz = w/2, d_val/2, h/2
        tv_geo.__init__(w, d_val, h) 
        
        # Generator
        current_sequence_data, type_code = simulator.generate_test_sequence(m, w, d_val, h, final_ship, final_hand, p_type)
        
        # Logic Details
        dims = sorted([w, d_val, h])
        girth_mm = dims[2] + 2*(dims[1]+dims[0]) 
        girth_in = girth_mm / 25.4 
        mass_lb = m * 2.2046
        
        # Color coding
        if type_code == 'Invalid':
            info_label.config(text=f"Detected: {type_code}", foreground="red")
        else:
            info_label.config(text=f"Detected: Type {type_code}", foreground="blue")
        
        # Reason
        lbl_reason.config(text=f"{simulator.determine_ista_type(m, w, d_val, h, final_ship, final_hand, p_type)[1]}")
            
        girth_label.config(text=f"Girth: {girth_in:.1f} in / Mass: {mass_lb:.1f} lb")
        
        # Refresh Tree
        for item in tree.get_children():
            tree.delete(item)
            
        for i, step in enumerate(current_sequence_data):
            name_txt = step.get('name', step.get('type', 'Step'))
            desc_txt = step.get('detail', step.get('desc', ''))
            h_val = step.get('height', 0.0)
            if step['type'] in ['rot_edge', 'rot_corner']:
                h_val = step.get('lift_height', 0.0)
            
            tree.insert("", "end", values=(i+1, name_txt, desc_txt, f"{h_val:.0f}"))

    def auto_select_method(*args):
        # Triggered by Input changes (Mass, Dims, Prod)
        # Calculates "Best Guess" -> Sets Vars -> Triggers refresh_display via trace
        
        try:
            m = mass_var.get()
            w = w_var.get()
            d_val = d_var.get()
            h = h_var.get()
            p_type = prod_var.get()
        except tk.TclError:
            return

        dims = sorted([w, d_val, h])
        girth_mm = dims[2] + 2*(dims[1]+dims[0]) 
        girth_in = girth_mm / 25.4 
        mass_lb = m * 2.2046
        
        target_ship = 'Parcel'
        target_hand = 'Standard'
        
        if girth_in > 165.0:
            target_ship = 'LTL'
            target_hand = 'Palletized'
        elif p_type == 'TV/Monitor' and mass_lb >= 150.0:
            target_ship = 'LTL'
            target_hand = 'Palletized'
        else:
            # For Parcel range cases, default to Parcel/Standard
            pass

        # Update if different. This triggers refresh_display because of trace on variables.
        # Check current to avoid unnecessary updates/traces
        if ship_var.get() != target_ship:
             ship_var.set(target_ship)
        if hand_var.get() != target_hand:
             hand_var.set(target_hand)
        
        # If no change in Method, we STILL need to refresh display (e.g. Mass changed -> Heights changed)
        # We can call refresh_display directly.
        # But if we did set vars, refresh_display is called via trace.
        # Double calling is harmless but let's be clean.
        # If we didn't change vars, call refresh.
        # Note: 'trace' might be asynchronous or immediate? In Tkinter typically immediate.
        
        # Actually, simpler: Always call refresh_display here?
        # If we changed var, refresh is called.
        # If we call it again, just redundant.
        # Or, we can UNBIND trace for Ship/Hand, and call refresh here? No, user clicking radio needs trace.
        
        # Let's just call refresh_display() explicitly if inputs changed but targets matched current.
        # If targets didn't match, the .set() triggers refresh.
        # But we don't easily know if .set() happened without flags.
        # Just call it. It's fast.
        refresh_display()

    # Bind Updates
    # Inputs -> Auto Logic -> (sets vars) -> Refresh
    root.bind('<Return>', lambda e: auto_select_method())
    mass_var.trace('w', lambda *args: auto_select_method())
    prod_var.trace('w', lambda *args: auto_select_method())
    w_var.trace('w', lambda *args: auto_select_method())
    h_var.trace('w', lambda *args: auto_select_method())
    d_var.trace('w', lambda *args: auto_select_method())
    
    # Method Manual Override -> Direct Refresh (Skip Auto Logic)
    ship_var.trace('w', lambda *args: refresh_display())
    hand_var.trace('w', lambda *args: refresh_display())
    
    # Old traces removed
    
    btn_refresh = ttk.Button(config_frame, text="Update", command=auto_select_method)
    btn_refresh.grid(row=0, column=2, rowspan=2, padx=10, sticky='ns')
    
    # 2. Reference Model Selector
    def open_ref_model_dialog():
        top = tk.Toplevel(root)
        top.title("Select Samsung TV Reference Model")
        top.geometry("1000x600") 
        
        # Layout Frames
        # 1. Tree Area (Top, Expanded)
        tree_frame = ttk.Frame(top)
        tree_frame.pack(side='top', fill='both', expand=True, padx=10, pady=(10, 5))
        
        # 2. Button Area (Bottom)
        btn_frame = ttk.Frame(top)
        btn_frame.pack(side='bottom', fill='x', pady=10)

        # Helper for sorting
        def sort_column(tv, col, reverse):
            l = [(tv.set(k, col), k) for k in tv.get_children('')]
            try:
                l.sort(key=lambda t: float(t[0].split()[0].replace(',','')), reverse=reverse)
            except ValueError:
                l.sort(reverse=reverse)

            for index, (val, k) in enumerate(l):
                tv.move(k, '', index)

            tv.heading(col, command=lambda: sort_column(tv, col, not reverse))

        # Columns
        cols = ('Model', 'Size', 'Package Size\n(mm)', 'Package Weight\n(kg)', 
                'Set w/ Stand Size', 'Set w/ Stand Weight', 
                'Set wo/ Stand Size', 'Set wo/ Stand Weight', 'Stand Basic')
        
        # Helper map for data keys if needed, but we insert by order below
        
        # Custom Style for taller headers (approx 40% increase)
        style = ttk.Style()
        style.configure("Ref.Treeview.Heading", padding=(5, 15)) # Vertical padding to increase height
        
        # Create Treeview with correct parent (tree_frame) and custom style
        tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=20, style="Ref.Treeview")
        
        for c in cols:
            tree.heading(c, text=c, command=lambda _c=c: sort_column(tree, _c, False))
            if 'Weight' in c or 'Size' in c or 'Stand' in c:
                tree.column(c, width=140, anchor='center')
            else:
                tree.column(c, width=100, anchor='center')
            
        # Scrollbars (Parent: tree_frame)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid Layout for Tree + Scrollbars to handle corners cleanly
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Populate
        for item in REFERENCE_MODELS:
            tree.insert("", "end", values=(
                item['name'], item['inch'], 
                item['pkg_size'], item['pkg_m'],
                item['set_w_std_size'], item['set_w_std_m'],
                item['set_wo_std_size'], item['set_wo_std_m'],
                item['stand_base']
            ))
                                           
        # Select Function
        def on_select():
            sel_id = tree.selection()
            if not sel_id: return
            item_vals = tree.item(sel_id[0], 'values')
            
            # Parse Pkg Size 'WxHxD'
            # Index 2 is 'Package Size\n(mm)'
            pkg_size_str = item_vals[2]
            try:
                # Remove spaces and split (format '1625 x 935 x 170')
                parts = pkg_size_str.lower().replace(' ', '').split('x')
                w_mm = float(parts[0])
                # Note: Table usually W x H x D
                h_mm = float(parts[1])
                d_mm = float(parts[2])
                
                # Check index for Weight (Index 3)
                m_kg = float(item_vals[3])
                
                # Update Main vars
                w_var.set(w_mm) 
                h_var.set(h_mm)
                d_var.set(d_mm)
                mass_var.set(m_kg)
                
                # Update Logic
                ship_var.set('Parcel') 
                auto_select_method()
                top.destroy()
                
            except Exception as e:
                print(f"Error parsing model data: {e}")
            
        btn_select = ttk.Button(btn_frame, text="Select Model", command=on_select)
        btn_select.pack(anchor='center')
        
    btn_ref = ttk.Button(config_frame, text="Ref. Model", command=open_ref_model_dialog)
    btn_ref.grid(row=0, column=3, rowspan=2, padx=5, sticky='ns')
    
    # 3. Sequence Display

    # 3. Visualization Selection
    viz_frame = ttk.LabelFrame(root, text="Select Visualizations")
    viz_frame.pack(padx=10, pady=5, fill="x")
    
    viz_opts = [
        ("Matplotlib (World Fixed)", "mpl_world"),
        ("Matplotlib (Box Fixed)", "mpl_box"),
        ("PyVista (World Fixed)", "pv_world"),
        ("PyVista (Box Fixed)", "pv_box")
    ]
    viz_vars = {}
    
    # 1-row layout
    for col_idx, (text, key) in enumerate(viz_opts):
        var = tk.BooleanVar(value=(key.startswith("pv"))) 
        chk = ttk.Checkbutton(viz_frame, text=text, variable=var)
        chk.pack(side='left', padx=10) # Horizontal pack
        viz_vars[key] = var

    # Action Functions
    def run_tests():
        # Treeview selection
        selected_items = tree.selection()
        if not selected_items:
            print("No test steps selected.")
            return

        selected_indices = [tree.index(item) for item in selected_items]

        print("\n" + "="*30 + f" Starting Detection Test " + "="*30)
        
        # Ensure visualizer uses the current mapper from simulator
        current_mapper = getattr(simulator, 'mapper', None)
        if current_mapper:
            tv_viz.mapper = current_mapper
            
        # Print Mapping Info
        print(" [Active Face Mapping]")
        for k, v in current_mapper.mapping.items():
            print(f"   Face {k} -> {tv_viz.mapper.get_face_label(v)}")
        print("-" * 30)
        
        for idx in selected_indices:
            step = current_sequence_data[idx]
            desc = step['desc']
            d_type = step['type']
            height = step.get('height', 0.0)
            
            print(f"\n>>> Running: {desc}")
            
            if d_type in ['rot_edge', 'rot_corner']:
                # Rotational Drop
                pivot_id = step.get('pivot_edge') if d_type == 'rot_edge' else step.get('pivot_corner')
                lift_h = step.get('lift_height', height)
                pose = simulator.calculate_rotational_pose(d_type, lift_h, pivot_id)
            else:
                # Free Fall Drop
                impact_id = step['id']
                pose = simulator.calculate_drop_pose(d_type, height, impact_id)
            
            # CAD 가이드 출력
            print_cad_guidance(pose)
            
            # 가상 포인트 생성
            pts = generate_test_points(tv_geo, pose, noise_std=0.0, outlier_ratio=0.0)
            
            # Title Suffix
            suffix = f" - {desc}"

            # Visualizations
            if viz_vars["mpl_world"].get():
                print("    [Visualizing] Matplotlib (World Fixed)...")
                tv_viz.show_matplotlib(pose, pts, title=desc, view_mode='world_fixed')
                
            if viz_vars["mpl_box"].get():
                print("    [Visualizing] Matplotlib (Box Fixed)...")
                tv_viz.show_matplotlib(pose, pts, title=desc, view_mode='box_fixed', print_guidance=False)
                
            if viz_vars["pv_world"].get():
                print("    [Visualizing] PyVista (World Fixed)...")
                tv_viz.show_pyvista(pose, pts, view_mode='world_fixed', title_suffix=suffix)
                
            if viz_vars["pv_box"].get():
                print("    [Visualizing] PyVista (Box Fixed)...")
                tv_viz.show_pyvista(pose, pts, view_mode='box_fixed', title_suffix=suffix)
                
        print("\n" + "="*30 + " Tests Completed " + "="*30)

    # Buttons
    btn_frame = ttk.Frame(root)
    btn_frame.pack(padx=10, pady=10, fill="x")
    
    btn_run = ttk.Button(btn_frame, text="Calculate Poses & Visualize", command=run_tests)
    btn_run.pack(side="left", padx=5, expand=True, fill="x")

    # Mesh Generation UI (New Frame)
    mesh_frame = ttk.LabelFrame(root, text="Mesh Generation (Gmsh)")
    mesh_frame.pack(padx=10, pady=5, fill="x")

    # Mesh Options Variables
    # Use -1 to indicate "Auto/Default" (80% of Box Size)
    opt_chassis_w = tk.DoubleVar(value=-1)
    # Default Ratio 80%
    opt_chassis_ratio = tk.DoubleVar(value=80.0)
    opt_chassis_h = tk.DoubleVar(value=-1)
    opt_chassis_d = tk.DoubleVar(value=50.0)
    
    opt_cell_w = tk.DoubleVar(value=-1)
    opt_cell_ratio = tk.DoubleVar(value=80.0)
    opt_cell_h = tk.DoubleVar(value=-1)
    opt_cell_d = tk.DoubleVar(value=50.0)
    
    # Hole Configuration (Default 60%)
    opt_hole_w = tk.DoubleVar(value=-1)
    opt_hole_ratio = tk.DoubleVar(value=60.0)
    opt_hole_h = tk.DoubleVar(value=-1)
    
    def open_mesh_options():
        d = tk.Toplevel(root)
        d.title("Mesh Options")
        d.geometry("500x480") # Increased height for Hole options
        
        # Center relative to root
        root_x = root.winfo_x()
        root_y = root.winfo_y()
        root_w = root.winfo_width()
        root_h = root.winfo_height()
        
        # Calculate center
        x = root_x + (root_w // 2) - (500 // 2)
        y = root_y + (root_h // 2) - (480 // 2)
        d.geometry(f"+{x}+{y}")
        
        # Helper to get current Box Dims
        box_w_val = w_var.get()
        box_h_val = h_var.get()
        
        # Helper to get default if -1
        def get_val_or_default(var, ratios):
             v = var.get()
             if v <= 0:
                 if ratios == 'w' or ratios == 'hw': return box_w_val * 0.8
                 if ratios == 'h' or ratios == 'hh': return box_h_val * 0.8 # Wait, Hole default 60%?
                 if ratios == 'hw_def': return box_w_val * 0.6
                 if ratios == 'hh_def': return box_h_val * 0.6
                 if ratios == 'd': return 50.0
             return v

        # --- Chassis Frame ---
        lf_chassis = ttk.LabelFrame(d, text="SET_CHASSIS Dimensions")
        lf_chassis.pack(fill='x', padx=10, pady=5)
        
        cw = tk.DoubleVar(value=get_val_or_default(opt_chassis_w, 'w'))
        ch = tk.DoubleVar(value=get_val_or_default(opt_chassis_h, 'h'))
        cd = tk.DoubleVar(value=get_val_or_default(opt_chassis_d, 'd'))
        c_ratio = tk.DoubleVar(value=opt_chassis_ratio.get())
        
        # Row 1: Dimensions
        f_dim_c = ttk.Frame(lf_chassis)
        f_dim_c.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(f_dim_c, text="W:").pack(side='left')
        ttk.Entry(f_dim_c, textvariable=cw, width=6).pack(side='left', padx=2)
        ttk.Label(f_dim_c, text="H:").pack(side='left')
        ttk.Entry(f_dim_c, textvariable=ch, width=6).pack(side='left', padx=2)
        ttk.Label(f_dim_c, text="D:").pack(side='left')
        ttk.Entry(f_dim_c, textvariable=cd, width=6).pack(side='left', padx=2)
        
        # Row 2: Ratio Controls
        f_ratio_c = ttk.Frame(lf_chassis)
        f_ratio_c.pack(fill='x', padx=5, pady=2)
        
        def apply_chassis_ratio():
            r = c_ratio.get() / 100.0
            cw.set(box_w_val * r)
            ch.set(box_h_val * r)
            
        ttk.Label(f_ratio_c, text="Ratio(%):").pack(side='left')
        ttk.Entry(f_ratio_c, textvariable=c_ratio, width=5).pack(side='left', padx=2)
        ttk.Button(f_ratio_c, text="Set Ratio to W/H", command=apply_chassis_ratio).pack(side='left', padx=5)
        
        # --- Cell Frame ---
        lf_cell = ttk.LabelFrame(d, text="SET_CELL Dimensions")
        lf_cell.pack(fill='x', padx=10, pady=5)
        
        clw = tk.DoubleVar(value=get_val_or_default(opt_cell_w, 'w'))
        clh = tk.DoubleVar(value=get_val_or_default(opt_cell_h, 'h'))
        cld = tk.DoubleVar(value=get_val_or_default(opt_cell_d, 'd'))
        cl_ratio = tk.DoubleVar(value=opt_cell_ratio.get())
        
        # Row 1: Dimensions
        f_dim_cl = ttk.Frame(lf_cell)
        f_dim_cl.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(f_dim_cl, text="W:").pack(side='left')
        ttk.Entry(f_dim_cl, textvariable=clw, width=6).pack(side='left', padx=2)
        ttk.Label(f_dim_cl, text="H:").pack(side='left')
        ttk.Entry(f_dim_cl, textvariable=clh, width=6).pack(side='left', padx=2)
        ttk.Label(f_dim_cl, text="D:").pack(side='left')
        ttk.Entry(f_dim_cl, textvariable=cld, width=6).pack(side='left', padx=2)
        
        # Row 2: Ratio Controls
        f_ratio_cl = ttk.Frame(lf_cell)
        f_ratio_cl.pack(fill='x', padx=5, pady=2)
        
        def apply_cell_ratio():
            r = cl_ratio.get() / 100.0
            clw.set(box_w_val * r)
            clh.set(box_h_val * r)
            
        ttk.Label(f_ratio_cl, text="Ratio(%):").pack(side='left')
        ttk.Entry(f_ratio_cl, textvariable=cl_ratio, width=5).pack(side='left', padx=2)
        ttk.Button(f_ratio_cl, text="Set Ratio to W/H", command=apply_cell_ratio).pack(side='left', padx=5)
        
        # --- Cushion Hole Frame ---
        lf_hole = ttk.LabelFrame(d, text="Cushion Part-Through Hole")
        lf_hole.pack(fill='x', padx=10, pady=5)
        
        hw = tk.DoubleVar(value=get_val_or_default(opt_hole_w, 'hw_def'))
        hh = tk.DoubleVar(value=get_val_or_default(opt_hole_h, 'hh_def'))
        h_ratio = tk.DoubleVar(value=opt_hole_ratio.get())
        
        f_dim_h = ttk.Frame(lf_hole)
        f_dim_h.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(f_dim_h, text="W:").pack(side='left')
        ttk.Entry(f_dim_h, textvariable=hw, width=6).pack(side='left', padx=2)
        ttk.Label(f_dim_h, text="H:").pack(side='left')
        ttk.Entry(f_dim_h, textvariable=hh, width=6).pack(side='left', padx=2)
        
        f_ratio_h = ttk.Frame(lf_hole)
        f_ratio_h.pack(fill='x', padx=5, pady=2)
        
        def apply_hole_ratio():
            r = h_ratio.get() / 100.0
            hw.set(box_w_val * r)
            hh.set(box_h_val * r)
            
        ttk.Label(f_ratio_h, text="Ratio(%):").pack(side='left')
        ttk.Entry(f_ratio_h, textvariable=h_ratio, width=5).pack(side='left', padx=2)
        ttk.Button(f_ratio_h, text="Set Ratio to W/H", command=apply_hole_ratio).pack(side='left', padx=5)
        
        # --- Actions ---
        def on_ok():
            opt_chassis_w.set(cw.get())
            opt_chassis_h.set(ch.get())
            opt_chassis_d.set(cd.get())
            opt_chassis_ratio.set(c_ratio.get())
            
            opt_cell_w.set(clw.get())
            opt_cell_h.set(clh.get())
            opt_cell_d.set(cld.get())
            opt_cell_ratio.set(cl_ratio.get())
            
            opt_hole_w.set(hw.get())
            opt_hole_h.set(hh.get())
            opt_hole_ratio.set(h_ratio.get())
            d.destroy()
            
        btn_frame = ttk.Frame(d)
        btn_frame.pack(fill='x', pady=10)
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side='left', padx=20, expand=True)
        ttk.Button(btn_frame, text="Cancel", command=d.destroy).pack(side='left', padx=20, expand=True)

    
    # Inputs: Size, Thickness
    ttk.Label(mesh_frame, text="W-Elem:").pack(side='left', padx=2)
    esh_w_var = tk.DoubleVar(value=150.0)
    ttk.Entry(mesh_frame, textvariable=esh_w_var, width=4).pack(side='left', padx=1)
    
    ttk.Label(mesh_frame, text="H-Elem:").pack(side='left', padx=2)
    esh_h_var = tk.DoubleVar(value=100.0)
    ttk.Entry(mesh_frame, textvariable=esh_h_var, width=4).pack(side='left', padx=1)

    ttk.Label(mesh_frame, text="D-Elem:").pack(side='left', padx=2)
    esh_d_var = tk.DoubleVar(value=50.0)
    ttk.Entry(mesh_frame, textvariable=esh_d_var, width=4).pack(side='left', padx=1)

    ttk.Label(mesh_frame, text="Floor Elem:").pack(side='left', padx=5)
    esh_floor_var = tk.DoubleVar(value=200.0)
    ttk.Entry(mesh_frame, textvariable=esh_floor_var, width=5).pack(side='left', padx=2)
    
    ttk.Label(mesh_frame, text="Thick:").pack(side='left', padx=5)
    thk_var = tk.DoubleVar(value=2.0)
    ttk.Entry(mesh_frame, textvariable=thk_var, width=5).pack(side='left', padx=2)
    
    view_mesh_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(mesh_frame, text="View", variable=view_mesh_var).pack(side='left', padx=10)
    
    ttk.Button(mesh_frame, text="Options", command=open_mesh_options).pack(side='left', padx=5)
    
    def create_box_mesh():
        if not HAS_GMSH_GEN:
            print("Error: Gmsh Generator not available.")
            return

        selected_items = tree.selection()
        if not selected_items:
            print("Select a step to generate mesh for.")
            return
        
        # Use first selected item
        idx = tree.index(selected_items[0])
        step = current_sequence_data[idx]
        
        d_type = step['type']
        height = step.get('height', 0.0)
        
        print(f"\n[Mesh Gen] Generating mesh for step: {step['desc']}")
        
        # Calculate Pose
        if d_type in ['rot_edge', 'rot_corner']:
            pivot_id = step.get('pivot_edge') if d_type == 'rot_edge' else step.get('pivot_corner')
            lift_h = step.get('lift_height', height)
            pose = simulator.calculate_rotational_pose(d_type, lift_h, pivot_id)
        else:
             impact_id = step['id']
             pose = simulator.calculate_drop_pose(d_type, height, impact_id)
             
        # Generate
        try:
            w = w_var.get()
            d_val = d_var.get() # Depth (Thickness of box)
            h = h_var.get()
            
            # Resolve Options
            cw = opt_chassis_w.get(); ch = opt_chassis_h.get(); cd = opt_chassis_d.get()
            if cw <= 0: cw = w * 0.8
            if ch <= 0: ch = h * 0.8
            
            clw = opt_cell_w.get(); clh = opt_cell_h.get(); cld = opt_cell_d.get()
            if clw <= 0: clw = w * 0.8
            if clh <= 0: clh = h * 0.8
            
            hole_w = opt_hole_w.get()
            hole_h = opt_hole_h.get()
            if hole_w <= 0: hole_w = w * 0.6
            if hole_h <= 0: hole_h = h * 0.6
            
            gen = BoxMeshByGmsh(w, d_val, h, thickness=thk_var.get(), 
                                elem_size_x=esh_w_var.get(),
                                elem_size_y=esh_d_var.get(), # Y-axis is Depth -> D-Elem
                                elem_size_z=esh_h_var.get(), # Z-axis is Height -> H-Elem
                                elem_size_floor=esh_floor_var.get(),
                                chassis_dims=(cw, ch, cd),
                                cell_dims=(clw, clh, cld),
                                hole_dims=(hole_w, hole_h))
            
            # File name based on step
            # Handle tuple IDs (e.g. for Rotational Drops)
            # Robustly sanitize: remove ( ) ' " and replace , with -
            s_id_str = str(step['id']).replace('(', '').replace(')', '').replace("'", "").replace('"', "").replace(', ', '-').replace(',', '-')
                
            # Sanitize description for filename
            clean_desc = "".join(c for c in step['desc'] if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
            out_file = f"mesh_Step_{s_id_str}_{clean_desc}.rad"
            
            # Use Box Name Numbering for Component Name
            # User requested just "Box" instead of Box_{id}
            box_comp_name = "Box"
            
            # Select move mode (hardcoded or UI? User said: "option... selectively proceed")
            # For now, default to 'box' moving as it's more intuitive for single frame export.
            
            gen.generate_mesh(pose, move_mode='box', output_path=out_file, view=view_mesh_var.get(), box_name=box_comp_name)
            print(f"[Mesh Gen] Success! Saved to {out_file}")
            
        except Exception as e:
            print(f"[Mesh Gen] Error: {e}")
            import traceback
            traceback.print_exc()

    btn_mesh = ttk.Button(mesh_frame, text="Create Box Mesh", command=create_box_mesh)
    btn_mesh.pack(side='left', padx=5, expand=True, fill='x')

    # Initial Update
    auto_select_method()

    root.mainloop()


if __name__ == "__main__":
    # --- 공통 설정 ---
    box_w, box_d, box_h = 2000.0, 1200.0, 250.0 # 2000mm x 1200mm x 250mm
    geo = BoxGeometry(box_w, box_d, box_h)
    viz = BoxVisualizer(geo)
    
    # --- 예제 실행 ---
    # run_single_frame_estimation_example(geo, viz)
    # run_ista_drop_pose_example(geo, viz)
    # run_time_series_tracking_example(geo)
    # run_deformation_tracking_example(geo, viz)
    # run_dimension_estimation_example(geo, viz)
    run_ista_6a_test_suite()
    