
import numpy as np
from cad_simplifier import CADSimplifier

def verify_strict_alignment():
    print("========================================")
    print("   STRICT ALIGNMENT VERIFICATION TOOL   ")
    print("========================================")
    
    sim = CADSimplifier()
    sim.voxel_scale = 1.0 # Base scale
    
    # ---------------------------------------------------------
    # Test Case 1: Simple Face-to-Face Gap (Mutual)
    # ---------------------------------------------------------
    print("\n[Test 1] Face-to-Face Gap (0.5mm) - Mutual Expansion")
    # C1 at 0,0,0 (Size 10). Right Face at X=5.
    # C2 at 10.5,0,0 (Size 10). Left Face at X=5.5.
    c1 = {'type': 'oriented', 'center': np.array([0.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c2 = {'type': 'oriented', 'center': np.array([10.5,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c2['transform'][:3,3] = [10.5, 0.0, 0.0]
    
    sim.cutters = [c1, c2]
    
    # Run Alignment
    print("  -> Running _align_cutter_neighbors (Snap 1.0mm)...")
    sim._align_cutter_neighbors(snap_tolerance=1.0)
    
    # Verify Result
    # C1 Right Face should be at X=5.25
    # C2 Left Face should be at X=5.25
    # C1 Size -> 10.25. Center -> 0.125
    # C2 Size -> 10.25. Center -> 10.375
    
    c1_right = c1['transform'][0,3] + c1['extents'][0]/2
    c2_left = c2['transform'][0,3] - c2['extents'][0]/2
    gap = c2_left - c1_right
    
    print(f"  Result: C1 Right={c1_right:.5f}, C2 Left={c2_left:.5f}")
    print(f"  Final Gap: {gap:.5f} mm")
    
    if abs(gap) < 0.001:
        print("  [PASS] Gap Closed Perfectly (Zero Gap).")
    elif gap < -0.01:
        print(f"  [FAIL] Overlap Detected! ({gap:.5f} mm)")
    else:
        print(f"  [FAIL] Gap Remains! ({gap:.5f} mm)")

    # ---------------------------------------------------------
    # Test Case 2: Rotated Face vs Edge (Unilateral)
    # ---------------------------------------------------------
    print("\n[Test 2] Rotated Face vs Edge - Strict Unilateral")
    # C3 (45 deg) at 0,0,0. Face Normal (0.707, 0.707, 0).
    # Face dist from origin = 5.0. 
    # C4 (Box) at (10,10,0). Closest point (Corner) is at distance ~7.07.
    # Gap ~2.07mm.
    # Requirement: C3 Expands. C4 STAYS (No expansion).
    
    R_45 = np.array([
        [0.70710678, -0.70710678, 0, 0],
        [0.70710678,  0.70710678, 0, 0],
        [0,           0,          1, 0],
        [0,           0,          0, 1]
    ])
    c3 = {'type': 'oriented', 'center': np.array([0.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': R_45.copy()}
    c4 = {'type': 'oriented', 'center': np.array([10.0,10.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c4['transform'][:3,3] = [10.0, 10.0, 0.0]
    
    sim.cutters = [c3, c4]
    
    print(f"  -> Running (Snap 5.0mm)...")
    sim._align_cutter_neighbors(snap_tolerance=5.0)
    
    # Check C4 (The Edge provider)
    c4_changed = False
    if not np.allclose(c4['extents'], [10,10,10]): c4_changed = True
    if not np.allclose(c4['transform'][:3,3], [10,10,0]): c4_changed = True
    
    if c4_changed:
        print(f"  [FAIL] Cutter 4 (Edge) Moved! Extents: {c4['extents']}")
    else:
        print("  [PASS] Cutter 4 (Edge) remained fixed.")
        
    # Check C3 (Face)
    # Should have expanded X+ to close the gap.
    if c3['extents'][0] > 11.0:
        print(f"  [PASS] Cutter 3 (Face) Expanded to {c3['extents'][0]:.3f}")
    else:
        print(f"  [FAIL] Cutter 3 did not expand enough. ({c3['extents'][0]:.3f})")

    # ---------------------------------------------------------
    # Test Case 3: Already Touching (Deadband Check)
    # ---------------------------------------------------------
    print("\n[Test 3] Zero Gap Input (Deadband Check)")
    # C5, C6 touching exactly.
    c5 = {'type': 'oriented', 'center': np.array([0.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c6 = {'type': 'oriented', 'center': np.array([10.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c6['transform'][:3,3] = [10.0, 0.0, 0.0] # Right at edge 5.0 vs 5.0
    
    sim.cutters = [c5, c6]
    sim._align_cutter_neighbors(snap_tolerance=1.0)
    
    gap_final = (c6['transform'][0,3] - c6['extents'][0]/2) - (c5['transform'][0,3] + c5['extents'][0]/2)
    print(f"  Initial Gap: 0.000. Final Gap: {gap_final:.5f}")
    
    if abs(gap_final) < 0.00001:
        print("  [PASS] No movement occurred (Deadband active).")
    else:
        print(f"  [FAIL] Movement occurred despite zero gap! ({gap_final}mm overlap)")

if __name__ == "__main__":
    verify_strict_alignment()
