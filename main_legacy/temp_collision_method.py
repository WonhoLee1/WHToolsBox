
    def _check_collision_solid(self, R, pos, dims, col_mgr=None, use_boolean=True):
        """
        [Helper] 정밀 충돌 검사 (Boolean Subtraction)
        1. FCL(빠른 충돌)로 1차 필터링
        2. 충돌 감지 시, 실제 Boolean Subtraction으로 부피 감소 확인 (정밀 확인)
        3. 실패 시 FCL 결과 신뢰
        """
        # 1. Fast Pass: FCL or Simple AABB/Point Check
        # 만약 FCL이 "충돌 없음"이라고 하면, 굳이 Boolean을 할 필요가 없음.
        # (단, Undersize 모드에서는 '살짝 스치는' 것도 놓치면 안 되므로 주의가 필요하지만,
        #  FCL은 보통 보수적(Conservative)이므로 FCL이 없다고 하면 진짜 없는 것.)
        
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
            # Sample Faces safely
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
            # 사용자 요청: Gmsh 등 다른 엔진 시도
            if self._try_gmsh_boolean(box_mesh):
                 return True
            
            # If all booleans fail, conservative fallback:
            # FCL said touch, so likely collision.
            return True

    def _try_gmsh_boolean(self, box_mesh):
        """
        [Experimental] Gmsh를 이용한 Boolean Check
        """
        if not HAS_GMSH: return False
        try:
             # Gmsh Python API로 Two Meshes Boolean은 복잡함 (Mesh Pipeline)
             # 여기서는 placeholder 형태로 남겨두거나, 
             # 매우 간단한 'Bounding Box Overlap'만 재확인하는 정도로 타협.
             # 실제 Gmsh Mesh Boolean 구현은 STL I/O가 동반되어 느릴 수 있음.
             return False # 아직 구현 안됨 -> Fallback to FCL
        except:
             return False
