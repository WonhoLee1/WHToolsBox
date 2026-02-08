
import trimesh
import numpy as np

def check_boolean_engines():
    print("Trimesh Version:", trimesh.__version__)
    print("Available Boolean Engines:", trimesh.interfaces.blender.exists, trimesh.interfaces.scad.exists)
    
    # Create two boxes
    b1 = trimesh.creation.box(extents=[10, 10, 10])
    b2 = trimesh.creation.box(extents=[5, 5, 5])
    b2.apply_translation([5, 0, 0]) # Overlap half
    
    print("\nTesting Default Boolean Difference...")
    try:
        diff = trimesh.boolean.difference([b1], [b2])
        print("Result Volume:", diff.volume)
        print("Is Watertight:", diff.is_watertight)
        print("Success!")
    except Exception as e:
        print("Default Engine Failed:", str(e))

    print("\nTesting 'scad' Engine...")
    try:
        diff = trimesh.boolean.difference([b1], [b2], engine='scad')
        print("Result Volume:", diff.volume)
        print("Is Watertight:", diff.is_watertight)
        print("Success with SCAD!")
    except Exception as e:
        print("SCAD Engine Failed:", str(e))

if __name__ == "__main__":
    check_boolean_engines()
