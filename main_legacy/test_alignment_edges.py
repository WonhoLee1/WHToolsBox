
import numpy as np
import trimesh
from cad_simplifier import CADSimplifier

def test_alignment():
    print("Testing Edge/Vertex Alignment Logic...")
    sim = CADSimplifier()
    
    # 1. Edge-to-Edge Test
    # Two boxes, parallel edges, slightly offset.
    print("\n--- Test 1: Parallel Edge Gap (Y=0.5) ---")
    c1 = {'type': 'oriented', 'center': np.array([0.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c2 = {'type': 'oriented', 'center': np.array([0.0,10.5,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c2['transform'][:3,3] = [0.0,10.5,0.0]
    
    sim.cutters = [c1, c2]
    sim.voxel_scale = 1.0
    sim._align_cutters_edge_vertex(snap_tolerance=1.0)
    
    print(f"C1 Ext Y: {c1['extents'][1]:.3f} (Exp > 10.0)")
    print(f"C2 Ext Y: {c2['extents'][1]:.3f} (Exp > 10.0)")
    
    if c1['extents'][1] > 10.2:
        print("PASS: Edges aligned via Y-expansion.")
    else:
        print("FAIL: No expansion.")

    # 2. Vertex-to-Vertex Test
    # Two boxes corner to corner.
    print("\n--- Test 2: Vertex Gap (0.5, 0.5, 0.5) ---")
    c3 = {'type': 'oriented', 'center': np.array([0.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c4 = {'type': 'oriented', 'center': np.array([11.0, 11.0, 11.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c4['transform'][:3,3] = [10.5, 10.5, 10.5]
    
    sim.cutters = [c3, c4]
    sim._align_cutters_edge_vertex(snap_tolerance=1.0)
    
    print(f"C3 Extents: {c3['extents']}")
    if np.all(c3['extents'] > 10.0):
        print("PASS: Vertex expansion occurred on all axes.")
    else:
        print("FAIL: Vertex alignment failed.")

    # 3. Chain Test (Asymmetric)
    # C5 -- C6 -- C7
    # C5 at X=0, Size=10 (Right Edge X=5).
    # C6 at X=10.5, Size=10 (Left Edge X=5.5, Right Edge X=15.5).
    # C7 at X=20.5 (Left Edge X=15.5).
    # Gap C5-C6: 0.5. C6 needs to expand Left by 0.5.
    # Gap C6-C7: 0.0. C6 needs to expand Right by 0.0? (Already touching if C7 is at 20.5-5=15.5)
    # Wait, 10.5 center. Left: 5.5. Right 15.5.
    # C5 Right: 5. Gap 0.5.
    # C7 Left: 20.5 - 5 = 15.5. Gap 0.
    # Result: C6 should expand +0.5 on Left FacE (-X), and 0.0 on Right Face (+X).
    # Center should shift by -0.25 (Left).
    # Size should be 10.5.
    
    print("\n--- Test 3: Chain Asymmetric Gap ---")
    c5 = {'type': 'oriented', 'center': np.array([0.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    
    c6 = {'type': 'oriented', 'center': np.array([10.5,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c6['transform'][:3,3] = [10.5, 0.0, 0.0]
    
    c7 = {'type': 'oriented', 'center': np.array([21.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    # C7 at 21.0. Left Edge = 16.0.
    # C6 Right Edge = 15.5.
    # Gap C6-C7 = 0.5.
    
    # Let's start with C7 perfectly aligned to verify ONLY Left expansion.
    # C6 Right Edge 15.5. We want C7 Left Edge at 15.5.
    # C7 Center = 15.5 + 5 = 20.5.
    c7['transform'][:3,3] = [20.5, 0.0, 0.0]
    
    # Expected: 
    # C5-C6 Gap 0.5. C6 expands Left 0.25, C5 expands Right 0.25 (Symmetric pair).
    # C6-C7 Gap 0.0. No expansion.
    # Total C6: Expands Left 0.25. Right 0.0.
    # Wait, my logic is "Close Gap Symmetrically".
    # Gap 0.5. C5 moves R +0.25. C6 moves L +0.25.
    # So C6 extends Left Face by 0.25.
    # C6 Size: 10 + 0.25. Center shifts -0.125.
    
    # Wait. User wanted "Mutually".
    # Gap 0.5. C5 expands 0.25. C6 expands 0.25.
    # My code 'move_vec_c1 = dist_vec * 0.5'. Yes.
    
    sim.cutters = [c5, c6, c7]
    sim._align_cutters_edge_vertex(snap_tolerance=1.0)
    
    print(f"C6 Ext X: {c6['extents'][0]:.4f} (Expected ~10.25)")
    print(f"C6 Center X: {c6['transform'][0,3]:.4f} (Expected ~10.375)")
    
    if abs(c6['extents'][0] - 10.25) < 0.01:
        print("PASS: C6 expanded asymmetrically (Left only).")
    else:
        print(f"FAIL: C6 expansion incorrect.")

    # 4. Unequal Size Face-to-Face Test
    # C8 (Large 20x20) at 0,0,0. Right Face X=10.
    # C9 (Small 10x10) at 15.5, 0, 0. Left Face X=10.5.
    # Gap 0.5.
    # Edges of C9 (Z +/- 5) are NOT aligned with C8 (Z +/- 10).
    # Vertices far apart.
    # Only Face logic works here.
    
    print("\n--- Test 4: Unequal Size Face Gap ---")
    c8 = {'type': 'oriented', 'center': np.array([0.0,0.0,0.0]), 'extents': np.array([20.0,20.0,20.0]), 'transform': np.eye(4)}
    c9 = {'type': 'oriented', 'center': np.array([15.5,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c9['transform'][:3,3] = [15.5, 0.0, 0.0]
    
    sim.cutters = [c8, c9]
    sim._align_cutters_edge_vertex(snap_tolerance=1.0)
    
    print(f"C8 Ext X: {c8['extents'][0]:.4f} (Expected ~20.25)")
    print(f"C9 Ext X: {c9['extents'][0]:.4f} (Expected ~10.25)")
    
    if c8['extents'][0] > 20.2 and c9['extents'][0] > 10.2:
        print("PASS: Unequal size faces expanded to meet.")
    else:
        print("FAIL: Face alignment failed.")

    # 5. Rotated Face vs Edge Test
    # C10 (Rotated 45 deg Z).
    # C11 (Box).
    # C10 Face pointing at C11 Edge.
    # Gap 1.0mm.
    # Expectation: C10 Face expands 1.0mm (Unilateral). C11 Edge stays.
    
    print("\n--- Test 5: Rotated Face vs Edge ---")
    R_45 = np.array([
        [0.707, -0.707, 0, 0],
        [0.707,  0.707, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ])
    # C10 at 0,0,0. Extent 10. 
    # Its X+ face normal is (0.707, 0.707, 0).
    # Face Center is at (3.535, 3.535, 0). (5 * 0.707)
    c10 = {'type': 'oriented', 'center': np.array([0.0,0.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': R_45.copy()}
    c10['transform'][:3,3] = [0,0,0]
    
    # C11 at (10, 10, 0).
    # Closest corner is near (5, 5, 0)?
    # Distance from Face Plane:
    # Plane P: n=(0.707, 0.707, 0). d = dot(p, n) for face center (3.535, 3.535) -> 2.5 + 2.5 = 5.0.
    # Plane Dist = 5.0 from Origin.
    # C11 Corner at (5, 5, 0)
    # Proj = 5*0.707 + 5*0.707 = 7.07.
    # Gap = 7.07 - 5.0 = 2.07mm.
    c11 = {'type': 'oriented', 'center': np.array([10.0,10.0,0.0]), 'extents': np.array([10.0,10.0,10.0]), 'transform': np.eye(4)}
    c11['transform'][:3,3] = [10.0, 10.0, 0.0]
    # C11 is Axis Aligned. It presents a CORNER to C10's FACE.
    # Max align of C11 Axes to Normal (0.707, 0.707, 0) is 0.707.
    # 0.707 < 0.8 Threshold. -> Unilateral.
    
    sim.cutters = [c10, c11]
    sim._align_cutters_edge_vertex(snap_tolerance=5.0)
    
    # Check C10 Expansion
    # Expected: Gap ~2.07 closed by C10 X+ Face.
    # C10 Extent X should increase by 2.07.
    print(f"C10 Ext X: {c10['extents'][0]:.4f} (Original 10.0)")
    print(f"C11 Extents: {c11['extents']}")
    
    if c10['extents'][0] > 11.5 and np.allclose(c11['extents'], [10,10,10]):
         print("PASS: Rotated Face expanded unilaterally.")
    else:
         print("FAIL: Rotation logic incorrect.")

if __name__ == "__main__":
    test_alignment()
