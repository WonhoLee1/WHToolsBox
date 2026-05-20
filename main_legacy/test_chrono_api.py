import pychrono as chrono
print("Testing PyChorno API...")
sys = chrono.ChSystemSMC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
print("✅ ChVector3d API works!")
