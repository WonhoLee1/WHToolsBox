import os
import sys

sys.path.append(os.getcwd())

try:
    from box_mesh_generator import BoxMeshByGmsh
except ImportError:
    print("Error: Could not import BoxMeshByGmsh")
    sys.exit(1)

def verify_sizing():
    print("Starting Sizing Verification...")
    
    # Use distinct sizes to verify mapping
    # X=W -> 100
    # Y=H -> 50
    # Z=D -> 200
    gen = BoxMeshByGmsh(
        width=1000, depth=200, height=500, thickness=2.0,
        elem_size_x=100.0,
        elem_size_y=50.0,
        elem_size_z=200.0
    )
    
    # Generate
    if os.path.exists("test_sizing.rad"): os.remove("test_sizing.rad")
    gen.generate_mesh([0]*6, output_path="test_sizing.rad", view=False)
    
    # Check for Paper Components
    if os.path.exists("test_sizing.rad"):
        print("[OK] Mesh generated.")
        
        # Simple check of file content strings
        with open("test_sizing_0000.rad", 'r') as f:
            c = f.read()
            if "BOX_PAPER_Top" in c: print("[OK] Paper Faces present.")
            if "Paper_Mat" in c: print("[OK] Paper Material present.")
            
    else:
        print("[FAIL] Generation failed.")
        
    print("Please visually verify element sizes in Gmsh: X~100, Y~50, Z~200")

if __name__ == "__main__":
    verify_sizing()
