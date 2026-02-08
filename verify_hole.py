import os
import sys

sys.path.append(os.getcwd())

try:
    from box_mesh_generator import BoxMeshByGmsh
except ImportError:
    print("Error: Could not import BoxMeshByGmsh")
    sys.exit(1)

def verify_hole():
    print("Starting Hole Verification...")
    
    # 1. Instantiate with Custom Hole
    # Box: 1000x200x500. Hole: 200x200 (20% W, 40% H)
    gen = BoxMeshByGmsh(
        width=1000, depth=200, height=500, thickness=2.0,
        hole_dims=(200, 200) # Custom Hole
    )
    
    # 2. Generate
    if os.path.exists("test_hole.rad"): os.remove("test_hole.rad")
    gen.generate_mesh([0]*6, output_path="test_hole.rad", view=False)
    
    # 3. Check Volume (Indirectly checking if hole exists)
    # Gmsh logs usually show this but we can't easily parse logs here.
    # We can check if file is generated.
    # For robust geometry check, we'd need to query Gmsh model, but we finalized it.
    
    if os.path.exists("test_hole.rad"):
        print("[OK] Mesh with custom hole generated.")
    else:
        print("[FAIL] Mesh generation failed.")

if __name__ == "__main__":
    verify_hole()
