
import jax
import jax.numpy as jnp
from brax.v1 import generalized
from brax.v1 import brax_pb2
import brax_boxdrop

# Create a system once
sys, _ = brax_boxdrop.run_ista_dynamic_sim(0.1, 0.1, 0.1)

print("System Attributes:")
print(dir(sys))

print("\nSystem Geometries is usually in sys.geoms or similar. Let's check:")
for attr in ['geoms', 'colliders', 'bodies', 'mass', 'inertia']:
    if hasattr(sys, attr):
        print(f"\n--- {attr} ---")
        val = getattr(sys, attr)
        print(val)
        if hasattr(val, 'shape'):
            print("Shape:", val.shape)

# Check if we can find the vertices in the system
# They might be in a geometry definition
