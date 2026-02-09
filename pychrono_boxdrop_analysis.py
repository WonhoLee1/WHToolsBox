"""
PyChorno Box Drop Simulation (Corner Drop Configuration)
Converted from MuJoCo implementation - PyChorno 9.0.1 compatible

Class-based architecture with SMC contact model for realistic foam/rubber behavior
"""

import numpy as np
import pychrono as chrono
import pychrono.irrlicht as chronoirr
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


class BoxDropSimulation:
    """Box drop simulation with corner-drop configuration"""
    
    def __init__(self):
        # Physical parameters
        self.L, self.W, self.H = 1.8, 1.2, 0.22
        self.MASS, self.G_ACC = 30.0, 9.806
        self.DT, self.TOTAL_STEPS = 0.001, 2500
        
        # Air parameters
        self.AIR_DENSITY = 1.225
        self.COEF_BLUNT_DRAG = 0.5
        self.COEF_GROUND_EFFECT = 1.0
        
        # Collision pad parameters
        self.PAD_XY = 0.1
        self.CORNER_PADS_NUMS = 5  # Number of divisions along depth (vertical edge)
        
        # Optimization parameters (SOLREF, SOLIMP)
        # SOLREF: [time_const, damping_ratio]
        self.DEFAULT_SOLREF = [0.02, 1.0] 
        self.DEFAULT_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2] # MuJoCo default-like
        
        self.pad_configs = [] # List to store pad attributes
        
        # Internal state
        self.system = None
        self.box_body = None
        self.vis = None
        self._current_cushion_force = 0.0
        
        # Data collection
        self.history = {
            'time': [], 'center': {'pos': [], 'vel': [], 'acc': []},
            'corners': [{'pos': [], 'vel': [], 'acc': []} for _ in range(8)],
            'impact_force': [], 'cushion_force': []
        }
        
        # Corner positions
        self.corners_local = np.array([[x, y, z]
            for x in [-self.L/2, self.L/2]
            for y in [-self.W/2, self.W/2]
            for z in [-self.H/2, self.H/2]])
        
        # Define vertical edges (indices of corners that form a vertical edge)
        # Based on corners_local logic: x loops first, then y, then z
        # 0: ---, 1: --+ (Edge 1)
        # 2: -+-, 3: -++ (Edge 2)
        # 4: +--, 5: +-+ (Edge 3)
        # 6: ++-, 7: +++ (Edge 4)
        self.vertical_edges = [(0, 1), (2, 3), (4, 5), (6, 7)]
        
        self._generate_pad_configs()
        self._calculate_corner_drop_rotation()
        
    def _generate_pad_configs(self):
        """
        Generate N split pads along vertical edges.
        Integrates previous corner/mid logic into a unified N-segment definition.
        """
        self.pad_configs = []
        
        # Pad Size
        # Total height H is divided by N (or slightly less to account for gaps if needed)
        # Here we use exact division for coverage
        pad_h = self.H / self.CORNER_PADS_NUMS
        
        for edge_idx, (idx_bottom, idx_top) in enumerate(self.vertical_edges):
            c_bottom = self.corners_local[idx_bottom]
            c_top = self.corners_local[idx_top]
            
            # Interpolate N pads
            for i in range(self.CORNER_PADS_NUMS):
                # Normalized position 0..1 (center of each segment)
                # i=0 -> bottom, i=N-1 -> top
                # Center of segment i: t = (i + 0.5) / N
                t = (i + 0.5) / self.CORNER_PADS_NUMS
                
                pos = c_bottom + (c_top - c_bottom) * t
                
                # Inset logic (move pads slightly inward to avoid edge artifacts same as MuJoCo logic)
                # Determine "outward" direction for this edge in XY plane
                sign_x = np.sign(pos[0])
                sign_y = np.sign(pos[1])
                
                # Inset position
                inset_pos = pos.copy()
                inset_pos[0] -= sign_x * self.PAD_XY * 0.0 # Keeping flush for now or apply inset
                inset_pos[1] -= sign_y * self.PAD_XY * 0.0
                
                # For collision shape size (half-size)
                size = np.array([self.PAD_XY, self.PAD_XY, pad_h / 2.0])
                
                pad_config = {
                    'id': f"edge_{edge_idx}_pad_{i}",
                    'edge_index': edge_idx,
                    'pad_index': i,
                    'pos': inset_pos,
                    'size': size, # Half-sizes for Chrono Box
                    'solref': list(self.DEFAULT_SOLREF), # Copy list
                    'solimp': list(self.DEFAULT_SOLIMP),
                    'color': [0.2, 0.8, 0.2] if (i == 0 or i == self.CORNER_PADS_NUMS - 1) else [0.8, 0.8, 0.2] # Green ends, Yellow mid
                }
                
                self.pad_configs.append(pad_config)
        
        print(f"✅ Generated {len(self.pad_configs)} pad configurations ({self.CORNER_PADS_NUMS} per edge)")

    def get_solref_xml_string(self, pad_config):
        """Convert list SOLREF to string for XML requirement"""
        return f"{pad_config['solref'][0]:.5f} {pad_config['solref'][1]:.5f}"
    
    def _calculate_corner_drop_rotation(self):
        diagonal = np.array([self.L, self.W, self.H])
        diagonal_normalized = diagonal / np.linalg.norm(diagonal)
        target_axis = np.array([0.01, 0, 1]) / np.linalg.norm(np.array([0.01, 0, 1]))
        
        rotation_axis = np.cross(diagonal_normalized, target_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis /= rotation_axis_norm
            cos_angle = np.dot(diagonal_normalized, target_axis)
            rotation_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            rot = R.from_rotvec(rotation_angle * rotation_axis)
        else:
            rot = R.from_quat([0, 0, 0, 1])
        
        self.initial_rotation = rot
        min_z = np.min((self.corners_local @ rot.as_matrix().T)[:, 2])
        self.initial_center_z = 0.5 - min_z
        
        euler = rot.as_euler('xyz', degrees=True)
        print(f"🔄 Rotation: Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
    
    def create_system(self):
        print("\n" + "="*70)
        print("🎯 PyChorno Box Drop - Corner Configuration")
        print("="*70)
        print(f"📦 Box: {self.L*1000:.0f}×{self.W*1000:.0f}×{self.H*1000:.0f}mm, {self.MASS}kg")
        print(f"📏 Drop: {self.initial_center_z*1000:.1f}mm | ⏱️ {self.DT*1000:.1f}ms, {self.TOTAL_STEPS*self.DT:.1f}s")
        print("="*70 + "\n")
        
        self.system = chrono.ChSystemSMC()
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -self.G_ACC))
        self.system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        return self.system
    
    def create_materials(self):
        # Ground material
        self.ground_material = chrono.ChContactMaterialSMC()
        self.ground_material.SetYoungModulus(1e9)
        self.ground_material.SetRestitution(0.1)
        self.ground_material.SetFriction(0.5)
        self.ground_material.SetKn(2e6)
        self.ground_material.SetGn(100)
        
        # Corner pad material (Base properties - individual pads use create_box logic)
        self.corner_material = chrono.ChContactMaterialSMC()
        self.corner_material.SetYoungModulus(1e6)  # 1 MPa soft foam
        self.corner_material.SetRestitution(0.3)
        self.corner_material.SetFriction(0.3)
        self.corner_material.SetKn(2e5)
        self.corner_material.SetGn(40)
        
        print("✅ Materials: SMC with foam properties")
    
    def create_ground(self):
        ground = chrono.ChBodyEasyBox(10, 10, 0.2, 1000, True, True, self.ground_material)
        ground.SetPos(chrono.ChVector3d(0, 0, -0.1))
        ground.SetFixed(True)
        ground.EnableCollision(True)
        self.system.Add(ground)
        print("✅ Ground created")
    
    def create_box(self):
        # Calculate inertia manually since we are using raw ChBody for custom collision
        vol = self.L * self.W * self.H
        mass = self.MASS
        
        # Rectangular box inertia: (1/12) * M * (d1^2 + d2^2)
        Ixx = (1/12) * mass * (self.W**2 + self.H**2)
        Iyy = (1/12) * mass * (self.L**2 + self.H**2)
        Izz = (1/12) * mass * (self.L**2 + self.W**2)
        
        # Create Body
        self.box_body = chrono.ChBody()
        self.box_body.SetMass(mass)
        self.box_body.SetInertiaXX(chrono.ChVector3d(Ixx, Iyy, Izz))
        
        # Set initial position/rotation
        quat = self.initial_rotation.as_quat()
        self.box_body.SetPos(chrono.ChVector3d(0, 0, self.initial_center_z))
        self.box_body.SetRot(chrono.ChQuaterniond(quat[3], quat[0], quat[1], quat[2]))
        
        # Visual Shape (Main Box - Blue transparent)
        vis_box = chrono.ChVisualShapeBox(self.L, self.W, self.H)
        vis_box.SetColor(chrono.ChColor(0.1, 0.5, 0.8))
        vis_box.SetOpacity(0.3)
        self.box_body.AddVisualShape(vis_box)
        
        # Collision System
        self.box_body.EnableCollision(True)
        cm = self.box_body.GetCollisionModel()
        cm.ClearModel()
        
        # Add pads based on configuration
        for pad in self.pad_configs:
            # Create material for this pad
            mat = chrono.ChContactMaterialSMC()
            
            # Map SOLREF (time_const, damping_ratio) to Key parameters if possible
            # Note: Explicit mapping requires formula. For now, using default foam properties 
            # or we could attempt to tune Kn/Gn based on solref logic if strictly required.
            # Using standard foam properties as base, but modified by solref 'stiffness' concept if we wanted.
            # Here keeping the reliable foam settings but acknowledging the config values.
            
            # To respect "Optimizability", we assume external optimizer will tweak these Kn/Gn values 
            # based on the solref variables stored in pad_config.
            # For this implementations, I'll use the constants defined in create_materials default for now,
            # but allow them to be distinct per pad.
            
            mat.SetYoungModulus(1e6) # Base stiffness
            mat.SetFriction(0.5)
            mat.SetRestitution(0.3)
            
            # Pad Position/Size
            pos = pad['pos']
            size = pad['size']
            
            # Add box shape for this pad
            # Note: AddBox takes (material, size_x, size_y, size_z, pos, rot)
            cm.AddBox(mat, size[0], size[1], size[2], 
                      chrono.ChVector3d(pos[0], pos[1], pos[2]), 
                      chrono.ChQuaterniond(1,0,0,0))
                      
            # Visualization for Pad (distinct colors)
            vis_pad = chrono.ChVisualShapeBox(size[0]*2, size[1]*2, size[2]*2)
            rgb = pad['color']
            vis_pad.SetColor(chrono.ChColor(rgb[0], rgb[1], rgb[2]))
            self.box_body.AddVisualShape(vis_pad, chrono.ChFrameD(chrono.ChVector3d(pos[0], pos[1], pos[2])))

        cm.BuildModel()
        
        # Initial perturbation
        np.random.seed(42)
        angvel = np.random.uniform(-0.003, 0.003, 3)
        self.box_body.SetAngVelLocal(chrono.ChVector3d(angvel[0], angvel[1], angvel[2]))
        
        self.system.Add(self.box_body)
        print(f"✅ Box created with {len(self.pad_configs)} contact pads")

    def _create_pads(self):
        # Superseded by create_box integration
        pass
    
    def apply_custom_forces(self):
        pos = self.box_body.GetPos()
        vel = self.box_body.GetPosDt()
        
        # Clear previous accumulated forces
        self.box_body.EmptyAccumulator(0)
        h, vz = pos.z, vel.z
        if h < 0.2 and vz < 0:
            geo_factor = ((self.L * self.W) / (2 * (self.L + self.W))) ** 2
            safe_h = max(h, 0.001)
            area = self.L * self.W
            coef = 0.5 * self.AIR_DENSITY * geo_factor * self.COEF_GROUND_EFFECT
            force = min(coef * area * (vz / safe_h) ** 2, 1000.0)
            # Apply force at center of mass (idx=0, force, point, local=True)
            self.box_body.AccumulateForce(0, chrono.ChVector3d(0, 0, force), chrono.ChVector3d(0,0,0), True)
            self._current_cushion_force = force
        else:
            self._current_cushion_force = 0.0
        
        # Aerodynamic drag
        v_mag = vel.Length()
        if v_mag > 0.01:
            frontal_area = self.W * self.H
            drag_mag = 0.5 * self.AIR_DENSITY * self.COEF_BLUNT_DRAG * frontal_area * v_mag**2
            drag_vec = chrono.ChVector3d(-vel.x, -vel.y, -vel.z)
            drag_vec.Normalize()
            self.box_body.AccumulateForce(0, drag_vec * drag_mag, chrono.ChVector3d(0,0,0), True)
    
    def create_visualization(self):
        try:
            self.vis = chronoirr.ChVisualSystemIrrlicht()
            self.vis.AttachSystem(self.system)
            self.vis.SetWindowSize(1024, 768)
            self.vis.SetWindowTitle('PyChorno Box Drop - Corner Configuration')
            self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
            self.vis.Initialize()
            
            # Add scene elements after Initialize
            self.vis.AddSkyBox()
            self.vis.AddTypicalLights()
            self.vis.AddCamera(chrono.ChVector3d(3, 3, 2), chrono.ChVector3d(0, 0, 0.5))
            
            print("✅ Visualization ready")
        except Exception as e:
            print(f"⚠️ Could not initialize visualization: {e}")
            print("   Falling back to headless mode.")
            self.vis = None
    
    def run_interactive(self):
        if not self.vis:
            return
            
        print("\n🎮 Interactive Mode (close window to continue)\n")
        
        # Limit simulation to 10 seconds of simulation time
        MAX_SIM_TIME = 10.0
        
        try:
            while self.vis.Run():
                if self.system.GetChTime() >= MAX_SIM_TIME:
                    print("   Reached maximum simulation time.")
                    break
                    
                self.vis.BeginScene()
                self.vis.Render()
                
                self.apply_custom_forces()
                self.system.DoStepDynamics(self.DT)
                
                self.vis.EndScene()
                
                # Progress report
                t = self.system.GetChTime()
                if int(t / self.DT) % 1000 == 0:
                    pos = self.box_body.GetPos()
                    print(f"   Time: {t:.2f}s | z: {pos.z*1000:.1f} mm")
                    
        except Exception as e:
            print(f"❌ Error during simulation: {e}")
            
        print(f"✅ Preview completed\n")
    
    def run_data_collection(self):
        print("📊 Data Collection")
        print(f"   Duration: {self.TOTAL_STEPS * self.DT:.1f}s\n")
        
        # Reset
        self.system.SetChTime(0)
        quat = self.initial_rotation.as_quat()
        self.box_body.SetPos(chrono.ChVector3d(0, 0, self.initial_center_z))
        self.box_body.SetRot(chrono.ChQuaterniond(quat[3], quat[0], quat[1], quat[2]))
        self.box_body.SetPosDt(chrono.ChVector3d(0, 0, 0))
        
        np.random.seed(42)
        angvel = np.random.uniform(-0.003, 0.003, 3)
        self.box_body.SetAngVelLocal(chrono.ChVector3d(angvel[0], angvel[1], angvel[2]))
        
        print("   Running...", end="", flush=True)
        prev_vel = np.zeros(3)
        
        for step in range(self.TOTAL_STEPS):
            self.apply_custom_forces()
            self.system.DoStepDynamics(self.DT)
            
            t = self.system.GetChTime()
            pos = self.box_body.GetPos()
            vel = self.box_body.GetPosDt()
            
            pos_np = np.array([pos.x, pos.y, pos.z])
            vel_np = np.array([vel.x, vel.y, vel.z])
            acc_np = (vel_np - prev_vel) / self.DT
            
            self.history['time'].append(t)
            self.history['center']['pos'].append(pos_np)
            self.history['center']['vel'].append(vel_np)
            self.history['center']['acc'].append(acc_np)
            self.history['cushion_force'].append(self._current_cushion_force)
            
            # Corner positions
            rot_mat = chrono.ChMatrix33d(self.box_body.GetRot())
            for i in range(8):
                c_local = self.corners_local[i]
                c_vec = chrono.ChVector3d(c_local[0], c_local[1], c_local[2])
                c_global = rot_mat * c_vec
                c_world = pos + c_global
                self.history['corners'][i]['pos'].append(np.array([c_world.x, c_world.y, c_world.z]))
                self.history['corners'][i]['vel'].append(np.zeros(3))
                self.history['corners'][i]['acc'].append(np.zeros(3))
            
            # Impact force (approximation from acceleration)
            impact = max(0, -acc_np[2] * self.MASS - self.MASS * self.G_ACC)
            self.history['impact_force'].append(impact)
            
            prev_vel = vel_np.copy()
            
            if step % (self.TOTAL_STEPS // 10) == 0:
                print(".", end="", flush=True)
        
        print(" Done!\n✅ Data collected\n")
    
    def process_data(self):
        self.history['time'] = np.array(self.history['time'])
        self.history['impact_force'] = np.array(self.history['impact_force'])
        self.history['cushion_force'] = np.array(self.history['cushion_force'])
        
        for key in ['pos', 'vel', 'acc']:
            self.history['center'][key] = np.array(self.history['center'][key])
            for i in range(8):
                self.history['corners'][i][key] = np.array(self.history['corners'][i][key])
    
    def plot_results(self):
        print("📈 Plotting...\n")
        
        # Figure 1: Kinematics
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle('PyChorno Box Drop: Kinematics', fontsize=14, fontweight='bold')
        
        labels = ['X', 'Y', 'Z']
        row_titles = ['Position (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s²)']
        data_keys = ['pos', 'vel', 'acc']
        colors = plt.cm.tab10(np.linspace(0, 1, 9))
        
        for row, (key, title) in enumerate(zip(data_keys, row_titles)):
            for col, axis in enumerate(labels):
                ax = axes[row, col]
                ax.plot(self.history['time'], 
                       self.history['center'][key][:, col] * 1000,
                       label='Center', color=colors[0], linewidth=2, alpha=0.8)
                
                if key == 'pos':
                    for i in range(8):
                        ax.plot(self.history['time'],
                               self.history['corners'][i][key][:, col] * 1000,
                               label=f'C{i+1}', color=colors[i+1], linewidth=1, alpha=0.6)
                
                ax.set_xlabel('Time (s)', fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.set_title(f'{title} - {axis}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                if row == 0 and col == 2:
                    ax.legend(loc='upper right', fontsize=8, ncol=1)
        
        plt.tight_layout()
        plt.savefig('box_drop_analysis_chrono.png', dpi=150, bbox_inches='tight')
        print("📊 Saved: box_drop_analysis_chrono.png")
        
        # Figure 2: Forces
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['time'], self.history['impact_force'],
                'r-', linewidth=1.5, label='Impact Force (approx)')
        plt.plot(self.history['time'], self.history['cushion_force'],
                'b--', linewidth=1.5, label='Air Cushion', alpha=0.7)
        
        max_f = np.max(self.history['impact_force'])
        if max_f > 0:
            idx = np.argmax(self.history['impact_force'])
            t_max = self.history['time'][idx]
            plt.scatter(t_max, max_f, color='black', zorder=5)
            plt.annotate(f'Peak: {max_f:.1f}N @ {t_max:.3f}s',
                        xy=(t_max, max_f), xytext=(t_max+0.2, max_f),
                        arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.title('Forces', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)'); plt.ylabel('Force (N)')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig('box_drop_impact_force_chrono.png', dpi=150, bbox_inches='tight')
        print("📊 Saved: box_drop_impact_force_chrono.png")
        plt.close('all')
    
    def run(self, interactive=True):
        self.create_system()
        self.create_materials()
        self.create_ground()
        self.create_box()
        
        if interactive:
            self.create_visualization()
            self.run_interactive()
        
        self.run_data_collection()
        self.process_data()
        self.plot_results()
        
        print("\n✅ All tasks completed!")


def main():
    sim = BoxDropSimulation()
    sim.run(interactive=True)


if __name__ == "__main__":
    main()
