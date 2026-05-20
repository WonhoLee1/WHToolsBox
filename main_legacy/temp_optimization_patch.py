# Optimized Targeted Loop Logic
for (id1, e1_idx, id2, e2_idx) in detected_pairs:
    idx1, idx2 = id1 - 1, id2 - 1
    c1 = self.cutters[idx1]
    c2 = self.cutters[idx2]
    
    # Geometry
    R1 = c1['transform'][:3, :3]
    R2 = c2['transform'][:3, :3]
    _, edges1 = self._get_obb_geometry(c1)
    _, edges2 = self._get_obb_geometry(c2)
    
    e1 = edges1[e1_idx]
    e2 = edges2[e2_idx]
    
    # Distance
    dist, dist_vec = self._calc_edge_distance(e1, e2)
    
    # Logic (Copied and Simplified)
    if dist < 0.01: continue # Already touching
    
    # No snap_dist check here! We treat it as infinite tolerance for targets.
    
    gap_dir = dist_vec / (dist + 1e-9)
    align1 = max([abs(np.dot(gap_dir, R1[:, k])) for k in range(3)])
    align2 = max([abs(np.dot(gap_dir, R2[:, k])) for k in range(3)])
    
    ratio_c1, ratio_c2 = 0.5, 0.5
    if align1 > 0.9 and align2 < 0.8:
        ratio_c1, ratio_c2 = 1.0, 0.0
    elif align2 > 0.9 and align1 < 0.8:
        ratio_c1, ratio_c2 = 0.0, 1.0

    boost = 1.2
    
    # C1
    if ratio_c1 > 0:
        req_dist_c1 = dist * ratio_c1
        # Need move_dir? gap_dir is fine. No, dist_vec.
        # Logic uses proj.
        move_dir = dist_vec / (np.linalg.norm(dist_vec) + 1e-9)
        
        for axis_idx in [0, 1, 2]:
            if axis_idx == e1['axis']: continue
            proj = np.dot(move_dir, R1[:, axis_idx])
            side = e1['fixed_signs'][axis_idx]
            
            if proj * side > 0:
                expansion = req_dist_c1 * boost * abs(proj) # Simple projection logic
                # Actually, using my "Stable" logic from Step 1099/1101?
                # Step 1101 used "expansion = req_dist * boost" (Assuming full force).
                # Wait, Step 1101 replaced logic with `expansion = req_dist * boost`.
                # I should match that to be consistent.
                expansion = req_dist_c1 * boost
                
                if expansion > dist + 0.1: expansion = dist + 0.1
                if expansion > 5.0: expansion = 5.0
                
                if expansion > 0.001:
                    face_id = axis_idx * 2 + (0 if side > 0 else 1)
                    key = (idx1, face_id)
                    if expansion > pending_face_shifts.get(key, 0.0):
                        pending_face_shifts[key] = expansion

    # C2
    if ratio_c2 > 0:
        req_dist_c2 = dist * ratio_c2
        move_dir = -dist_vec / (np.linalg.norm(dist_vec) + 1e-9)
        
        for axis_idx in [0, 1, 2]:
            if axis_idx == e2['axis']: continue
            proj = np.dot(move_dir, R2[:, axis_idx])
            side = e2['fixed_signs'][axis_idx]
            
            if proj * side > 0:
                expansion = req_dist_c2 * boost
                if expansion > dist + 0.1: expansion = dist + 0.1
                if expansion > 5.0: expansion = 5.0
                
                if expansion > 0.001:
                    face_id = axis_idx * 2 + (0 if side > 0 else 1)
                    key = (idx2, face_id)
                    if expansion > pending_face_shifts.get(key, 0.0):
                        pending_face_shifts[key] = expansion
