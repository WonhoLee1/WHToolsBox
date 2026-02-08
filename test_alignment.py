
import numpy as np
import trimesh
from cad_simplifier import CADSimplifier

def test_alignment():
    print("Testing Cutter Alignment Logic...")
    sim = CADSimplifier()
    
    # 1. Setup Dummy Cutters (Gap Scenario)
    # C1: Center (0,0,0), Size (10,10,10)
    # C2: Center (10.5, 0, 0), Size (10,10,10)
    # Gap = 0.5mm (Faces at x=5 and x=5.5)
    
    c1 = {
        'type': 'oriented',
        'center': np.array([0.0, 0.0, 0.0]),
        'extents': np.array([10.0, 10.0, 10.0]),
        'transform': np.eye(4)
    }
    c2 = {
        'type': 'oriented',
        'center': np.array([10.5, 0.0, 0.0]),
        'extents': np.array([10.0, 10.0, 10.0]),
        'transform': np.eye(4)
    }
    c2['transform'][:3, 3] = c2['center']
    
    sim.cutters = [c1, c2]
    sim.voxel_scale = 1.0
    
    print("\n--- Test 1: Gap (0.5mm) ---")
    print(f"Before: C1 Ext X={c1['extents'][0]}")
    sim._align_cutter_neighbors(snap_tolerance=1.0)
    print(f"After:  C1 Ext X={c1['extents'][0]}")
    
    # Mutual Shift Expectation:
    # Gap 0.5 + 0.01 = 0.51 Total.
    # Each moves 0.255.
    # C1 expands to 10.255.
    if 10.25 < c1['extents'][0] < 10.26:
        print(f"PASS: C1 Expanded by mutual shift (Ext: {c1['extents'][0]:.3f})")
    else:
        print(f"FAIL: C1 did not expand correctly (Ext: {c1['extents'][0]:.3f})")

    # 2. Setup Dummy Cutters (Overlap Scenario)
    # C1: (0,0,0), Size 10
    # C2: (9.5, 0, 0), Size 10
    # Overlap = 0.5mm (Faces at x=5 and x=4.5)
    
    c3 = {
        'type': 'oriented',
        'center': np.array([0.0, 0.0, 0.0]),
        'extents': np.array([10.0, 10.0, 10.0]),
        'transform': np.eye(4)
    }
    c4 = {
        'type': 'oriented',
        'center': np.array([9.5, 0.0, 0.0]),
        'extents': np.array([10.0, 10.0, 10.0]),
        'transform': np.eye(4)
    }
    c4['transform'][:3, 3] = c4['center']
    
    sim.cutters = [c3, c4]
    
    print("\n--- Test 2: Overlap (0.5mm) ---")
    # Expect Mutual Shrink/Move.
    # Shift = -0.5 + 0.01 = -0.49.
    # Half = -0.245.
    # C3 Extents should change by -0.245 => 9.755.
    
    print("Running alignment (Logic Check)...")
    sim._align_cutter_neighbors(snap_tolerance=1.0)
    print(f"After:  C3 Ext X={c3['extents'][0]}")
    
    if 9.75 < c3['extents'][0] < 9.76:
         print(f"PASS: C3 Contracted by mutual shift (Ext: {c3['extents'][0]:.3f})")
    else:
         print(f"FAIL: C3 did not contract correctly (Ext: {c3['extents'][0]:.3f})")

if __name__ == "__main__":
    test_alignment()
