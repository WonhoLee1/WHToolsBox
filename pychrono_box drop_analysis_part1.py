"""
PyChorno Box Drop Simulation (Corner Drop Configuration)
Converted from MuJoCo implementation with class-based architecture

Features:
- Class-based design for modularity
- SMC (Smooth Contact Model) for rubber/foam material behavior  
- 8 corner collision pads + 4 midpoint pads + 4 protective blocks
- Air cushion effect (ground effect)
- Custom aerodynamic forces (drag, lift)
- Comprehensive data collection and visualization

API: PyChorno 9.0.1 (ChVector3d, ChQuaterniond)
"""

import numpy as np
import pychrono as chrono
import pychrono.irrlicht as chronoirr
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import time


class BoxDropSimulation:
    """
    Box drop simulation using PyChorno physics engine.
    Implements corner drop configuration with advanced contact modeling.
    """
    
    def __init__(self):
        """Initialize simulation parameters"""
        # Physical Parameters
        self.L, self.W, self.H = 1.8, 1.2, 0.22  # Box dimensions (m)
        self.MASS = 30.0  # kg
        self.G_ACC = 9.806  # m/s^2
        
        # Simulation parameters
        self.DT = 0.001  # 1ms time step
        self.TOTAL_STEPS = 2500  # 2.5 seconds
        
        # Center of mass offset
        self.CoM_offset = np.array([0.0, 0.0, 0.0])
        
        # Air/Fluid Parameters
        self.AIR_DENSITY = 1.225  # kg/m^3
        self.AIR_VISCOSITY = 1.8e-5  # Pa.s
        self.COEF_BLUNT_DRAG = 0.5
        self.COEF_SLENDER_DRAG = 0.25
        self.COEF_LIFT = 1.0
        self.COEF_GROUND_EFFECT = 1.0
        
        # Collision Pad Parameters
        self.PAD_XY = 0.1  # Pad half-width
        self.PAD_Z = self.H / 6.0  # Pad half-height
        
        # Material Properties (SMC Model)
        # Corner pads - soft foam behavior
        self.corner_mat_props = {
            'young_modulus': 1e6,  # 1 MPa (soft foam)
            'poisson_ratio': 0.3,
            'restitution': 0.3,  # Low bounce
            'friction': 0.3,
            'adhesion': 0.0,
            'kn': 2e5,  # Normal stiffness
            'gn': 40,   # Normal damping
            'kt': 1e5,  # Tangential stiffness
            'gt': 20    # Tangential damping
        }
        
        # Ground material - hard surface
        self.ground_mat_props = {
            'young_modulus': 1e9,  # 1 GPa (concrete)
            'poisson_ratio': 0.2,
            'restitution': 0.1,
            'friction': 0.5,
            'adhesion': 0.0,
            'kn': 2e6,
            'gn': 100,
            'kt': 1e6,
            'gt': 50
        }
        
        # Internal State
        self.system = None
        self.box_body = None
        self.ground_body = None
        self.corner_bodies = []
        self.vis = None
        
        # Data collection
        self.history = {
            'time': [],
            'center': {'pos': [], 'vel': [], 'acc': []},
            'corners': [{'pos': [], 'vel': [], 'acc': []} for _ in range(8)],
            'impact_force': [],
            'cushion_force': []
        }
        
        # Corner positions in local coordinates
        self.corners_local = np.array([
            [x, y, z]
            for x in [-self.L/2, self.L/2]
            for y in [-self.W/2, self.W/2]
            for z in [-self.H/2, self.H/2]
        ])
        
        # Calculate corner drop rotation
        self._calculate_corner_drop_rotation()
    
    def _calculate_corner_drop_rotation(self):
        """Calculate rotation for corner drop configuration"""
        # Diagonal vector (corner to opposite corner)
        diagonal = np.array([self.L, self.W, self.H])
        diagonal_normalized = diagonal / np.linalg.norm(diagonal)
        
        # Target: align diagonal with Z-axis (with slight tilt for instability)
        target_axis = np.array([0.01, 0, 1])
        target_axis = target_axis / np.linalg.norm(target_axis)
        
        # Rotation axis: cross product
        rotation_axis = np.cross(diagonal_normalized, target_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            cos_angle = np.dot(diagonal_normalized, target_axis)
            rotation_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            rot = R.from_rotvec(rotation_angle * rotation_axis)
        else:
            rot = R.from_quat([0, 0, 0, 1])
        
        self.initial_rotation = rot
        
        # Calculate rotated corners
        rotated_corners = self.corners_local @ rot.as_matrix().T
        min_z = np.min(rotated_corners[:, 2])
        
        # Initial height: lowest corner at 0.5m
        self.initial_center_z = 0.5 - min_z
        
        # Store for debugging
        euler_angles = rot.as_euler('xyz', degrees=True)
        print(f"🔄 Corner Drop Rotation: Roll={euler_angles[0]:.1f}°, "
              f"Pitch={euler_angles[1]:.1f}°, Yaw={euler_angles[2]:.1f}°")
    
    def create_system(self):
        """Create PyChorno system with SMC contact model"""
        print("\n" + "="*70)
        print("🎯 PyChorno Box Drop Simulation - Corner Drop Configuration")
        print("="*70)
        print(f"📦 Box: {self.L*1000:.0f} × {self.W*1000:.0f} × {self.H*1000:.0f} mm, "
              f"{self.MASS} kg")
        print(f"📏 Drop height: {self.initial_center_z*1000:.1f} mm")
        print(f"⏱️  Time step: {self.DT*1000:.1f} ms, Duration: {self.TOTAL_STEPS*self.DT:.1f} s")
        print("="*70 + "\n")
        
        # Create system with SMC (Smooth Contact) model
        self.system = chrono.ChSystemSMC()
        
        # Set gravitational acceleration
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -self.G_ACC))
        
        # Solver settings
        self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        self.system.SetSolverMaxIterations(150)
        self.system.SetSolverForceTolerance(1e-6)
        
        # Time step
        self.system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
        self.system.SetStep(self.DT)
        
        return self.system
    
    def create_materials(self):
        """Create contact materials for different surfaces"""
        # Ground material (hard, concrete-like)
        self.ground_material = chrono.ChContactMaterialSMC()
        self.ground_material.SetYoungModulus(self.ground_mat_props['young_modulus'])
        self.ground_material.SetPoissonRatio(self.ground_mat_props['poisson_ratio'])
        self.ground_material.SetRestitution(self.ground_mat_props['restitution'])
        self.ground_material.SetFriction(self.ground_mat_props['friction'])
        self.ground_material.SetAdhesion(self.ground_mat_props['adhesion'])
        self.ground_material.SetKn(self.ground_mat_props['kn'])
        self.ground_material.SetGn(self.ground_mat_props['gn'])
        self.ground_material.SetKt(self.ground_mat_props['kt'])
        self.ground_material.SetGt(self.ground_mat_props['gt'])
        
        # Corner pad material (soft foam/rubber)
        self.corner_material = chrono.ChContactMaterialSMC()
        self.corner_material.SetYoungModulus(self.corner_mat_props['young_modulus'])
        self.corner_material.SetPoissonRatio(self.corner_mat_props['poisson_ratio'])
        self.corner_material.SetRestitution(self.corner_mat_props['restitution'])
        self.corner_material.SetFriction(self.corner_mat_props['friction'])
        self.corner_material.SetAdhesion(self.corner_mat_props['adhesion'])
        self.corner_material.SetKn(self.corner_mat_props['kn'])
        self.corner_material.SetGn(self.corner_mat_props['gn'])
        self.corner_material.SetKt(self.corner_mat_props['kt'])
        self.corner_material.SetGt(self.corner_mat_props['gt'])
        
        print("✅ Materials created: SMC contact model with foam/rubber properties")
