import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())

try:
    from box_mesh_generator import BoxMeshByGmsh
except ImportError:
    print("Error: Could not import BoxMeshByGmsh")
    sys.exit(1)

def verify():
    print("Starting Verification...")
    
    # Clean up previous
    if os.path.exists("test_mesh.rad"): os.remove("test_mesh.rad")
    if os.path.exists("test_mesh_0000.rad"): os.remove("test_mesh_0000.rad")
    
    # 1. Instantiate
    print("1. Instantiating Generator...")
    gen = BoxMeshByGmsh(
        width=1000, 
        depth=200, 
        height=500, 
        thickness=2.0,
        chassis_dims=(400, 300, 50),
        cell_dims=(200, 100, 50)
    )
    
    # 2. Generate
    print("2. Generating Mesh...")
    pose = [0, 0, 0, 0, 0, 0]
    gen.generate_mesh(pose, output_path="test_mesh.rad", view=False)
    
    # 3. Check Files
    print("3. Checking Files...")
    if not os.path.exists("test_mesh.rad"):
        print("FAIL: test_mesh.rad not found.")
        return
    if not os.path.exists("test_mesh_0000.rad"):
        print("FAIL: test_mesh_0000.rad not found.")
        return
        
    print("Files created successfully.")
    
    # 4. Check Content
    print("4. Checking Content for Components...")
    with open("test_mesh_0000.rad", 'r') as f:
        content = f.read()
        
    required = [
        "BOX_PAPER_Top",
        "BOX_PAPER_Bottom",
        "BOX_CUSHION", 
        "SET_CHASSIS", 
        "SET_CELL",
        "Paper_Mat",
        "Cushion_Mat"
    ]
    
    all_ok = True
    for item in required:
        if item in content:
            print(f"  [OK] Found {item}")
        else:
            print(f"  [FAIL] Missing {item}")
            all_ok = False
            
    if all_ok:
        print("\nVerification PASSED!")
    else:
        print("\nVerification FAILED!")

if __name__ == "__main__":
    verify()
