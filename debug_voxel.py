
import trimesh
import numpy as np

try:
    # Create a simple mesh
    mesh = trimesh.creation.box(extents=(10, 10, 10))
    # Voxelize it
    voxel = mesh.voxelized(pitch=1.0)
    
    print("Voxel object type:", type(voxel))
    print("Dir:", dir(voxel))
    
    if hasattr(voxel, 'origin'):
        print("Origin:", voxel.origin)
    else:
        print("No 'origin' attribute.")
        
    if hasattr(voxel, 'transform'):
        print("Transform:\n", voxel.transform)
        print("Calculated Origin from transform:", voxel.transform[:3, 3])
        
    if hasattr(voxel, 'translation'):
        print("Translation:", voxel.translation)

except Exception as e:
    print(e)
