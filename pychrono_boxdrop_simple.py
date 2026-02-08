"""
PyChorno Box Drop Simulation - Simplified Test Version
Tests basic API compatibility
"""

import numpy as np

try:
    import pychrono as chrono
    print("✅ pychrono imported successfully")
except:
    try:
        import PyChrono as chrono
        print("✅ PyChrono imported successfully")
    except:
        print("❌ Could not import PyChorno")
        exit(1)

# Test system creation
system = chrono.ChSystemSMC()
print("✅ ChSystemSMC created")

# Test gravity setting - try different API versions
try:
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    print("✅ SetGravitationalAcceleration (new API) works")
    VEC_TYPE = "ChVector3d"
    QUAT_TYPE = "ChQuaterniond"
except:
    try:
        system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
        print("✅ Set_G_acc (old API) works")
        VEC_TYPE = "ChVectorD"
        QUAT_TYPE = "ChQuaternionD"
    except Exception as e:
        print(f"❌ Gravity setting failed: {e}")
        exit(1)

print(f"\n📋 API Configuration:")
print(f"   Vector type: {VEC_TYPE}")
print(f"   Quaternion type: {QUAT_TYPE}")
print(f"\n✅ API test completed successfully!")
print(f"\nNow update pychrono_boxdrop_analysis.py to use: {VEC_TYPE}, {QUAT_TYPE}")
