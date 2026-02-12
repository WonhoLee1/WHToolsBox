import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import time
import copy
import msvcrt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==========================================
# Global Constants & Configuration
# ==========================================
# Solver Presets (Iterations, Tolerance, Solver, Cones)
# [기본값 설명]
# iterations: 솔버가 접촉력을 계산하기 위해 반복하는 횟수 (높을수록 정밀하지만 느림)
# tolerance: 솔버가 계산을 중단하는 오차 범위 (작을수록 정밀하지만 오래 걸림)
# solver: 'Newton'은 빠르고 정밀하며, 'PGS'는 안정적이고 접촉이 많을 때 유리함
# cones: 'elliptic'은 마찰력을 원형으로 계산(정교함), 'pyramidal'은 사각뿔로 근사(빠름)
SOLVER_PRESETS = {
    "1": {"name": "Accurate (정확)", "iter": 100, "tol": 1e-8, "solver": "Newton", "cone": "elliptic"},
    "2": {"name": "Normal (보통)",   "iter": 50,  "tol": 1e-5, "solver": "Newton", "cone": "pyramidal"},
    "3": {"name": "Fast (빠른)",     "iter": 30,  "tol": 1e-4, "solver": "PGS",    "cone": "pyramidal"},
    "4": {"name": "Very Fast (매우빠름)", "iter": 20, "tol": 1e-3, "solver": "PGS", "cone": "pyramidal"}
}
# Optimization Parameters
DEFAULT_SOLREF = [0.05, 0.1]
DEFAULT_SOLIMP = [0.2, 0.7, 0.02, 0.5, 2] # dmin, dmax, width, mid, power

# Physics Constants
G_ACC = 9.806
AIR_DENSITY = 1.225
AIR_VISCOSITY = 1.8e-5

# Box Friction
BOX_FRICTION_PARAMS = "0.3 0.005 0.0001"

# Pad Configuration
CORNER_PADS_NUMS = 5    # Number of pads along height (Z)
CORNER_SPLIT_XY = 3     # Number of pads along L and W within each corner (M)
GAP_RATIO = 0.05        # 패드와 패드 사이의 간격 비율 (분할된 조각들이 서로 겹치지 않게 함)
PAD_XY = 0.1           # 코너 패드 한 변의 '반폭' 길이 (0.1m = 전체 폭 200mm)
BOX_PAD_OFFSET = 0.00001 # 상자 본체와 패드 사이의 물리적 이격 거리 (불필요한 초기 접촉 방지)
PLASTIC_DEFORMATION_RATIO = 0.5 # 소성 변형율 (최대 압축량의 몇 %를 영구 변형으로 남길지 설정)
ENABLE_PAD_CONTACT = True # 패드 조각 상호 간의 충격/접촉 계산 여부

# Flexible Body Configuration
FLEX_STIFFNESS = 8000.0 # 굽힘 강성 (높을수록 단단함)
FLEX_DAMPING = 200.0   # 굽힘 감쇠 (진동 억제)

# ==========================================
# Class: BoxDropInstance
# ==========================================
class BoxDropInstance:
    def __init__(self, uid, position_offset, drop_type="corner", box_params=None, com_random=False, com_offset=None, 
                 label="", enable_pad_contact=ENABLE_PAD_CONTACT, use_flex=False, 
                 flex_stiffness=FLEX_STIFFNESS, flex_damping=FLEX_DAMPING):
        self.uid = uid
        self.base_pos = np.array(position_offset)  # Grid position (x, y, 0)
        self.drop_type = drop_type 
        self.label = label 
        self.enable_pad_contact = enable_pad_contact
        self.use_flex = use_flex 
        self.flex_stiffness = flex_stiffness
        self.flex_damping = flex_damping
        
        # Default Box Parameters
        self.L = box_params.get('L', 1.8)
        self.W = box_params.get('W', 1.2)
        self.H = box_params.get('H', 0.22)
        self.MASS = box_params.get('MASS', 30.0)
        
        # CoM Offset
        self.CoM_offset = np.zeros(3)
        if com_offset is not None:
            self.CoM_offset = np.array(com_offset)
        elif com_random:
            # Random offset within 100mm (-0.1 ~ 0.1)
            self.CoM_offset = np.random.uniform(-0.1, 0.1, 3)
            
        # State Data Storage
        self.history = {
            'time': [],
            'pos': [],          # CoM World Pos
            'vel': [],          # CoM World Vel
            'acc': [],          # CoM World Acc (derived)
            'impact_force': [],
            'cushion_force': [],
            'corner_pos': [[] for _ in range(8)], # Site positions (x,y,z)
            'corner_vel': [[] for _ in range(8)], # Site velocities
            'corner_acc': [[] for _ in range(8)], # Site accelerations
            'bending': [],       # Max joint angle (rad)
            'deflection': [],    # Max tip displacement (mm)
            'joint_angles': {},  # Individual joint angles over time
            'corner_comp': {},   # Time-series avg compression per corner
            'final_pad_comp': {} # Final compression for EVERY pad
        }
        self.prev_vel = np.zeros(3)
        self.prev_corner_vels = [np.zeros(3) for _ in range(8)]

        # Internal Calculation for Drop Orientation
        self.quat_mj = [1, 0, 0, 0] # w, x, y, z
        self.initial_z_offset = 0.0
        self._calculate_orientation()
        
        # Helper: Pad Configurations (Local)
        self.pad_configs = self._generate_pad_configs()

    def _calculate_orientation(self):
        """Calculate initial rotation and height offset based on Drop Type"""
        # Default: Flat
        rot = R.from_quat([0, 0, 0, 1])
        min_z_local = -self.H / 2.0
        
        self.corners_local = np.array([
            [x, y, z] for x in [-self.L/2, self.L/2]
            for y in [-self.W/2, self.W/2]
            for z in [-self.H/2, self.H/2]
        ])

        if self.drop_type == "corner":
            # Corner Drop Logic (Diagonal aligned with Z)
            diagonal = np.array([self.L, self.W, self.H])
            diagonal_norm = diagonal / np.linalg.norm(diagonal)
            target_axis = np.array([0.01, 0, 1]) # Slight tilt
            target_axis /= np.linalg.norm(target_axis)
            
            rotation_axis = np.cross(diagonal_norm, target_axis)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm > 1e-6:
                rotation_axis /= rotation_axis_norm
                cos_angle = np.dot(diagonal_norm, target_axis)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                rot = R.from_rotvec(angle * rotation_axis)
            
            # Recalculate lowest point
            rotated_corners = self.corners_local @ rot.as_matrix().T
            min_z_local = np.min(rotated_corners[:, 2])

        elif self.drop_type == "edge":
            # (Placeholder) Edge drop logic
            pass
        elif self.drop_type == "face":
            # (Placeholder) Face drop logic
            pass

        # Convert to MuJoCo Quaternion [w, x, y, z]
        q = rot.as_quat()
        self.quat_mj = [q[3], q[0], q[1], q[2]]
        
        # Initial Height: Lowest point should be at 500mm (0.5m)
        self.initial_z_offset = 0.5 - min_z_local

    def _generate_pad_configs(self):
        """Generate 3D split pad configurations at corners (Local coords)"""
        configs = []
        
        # 1. Corner Pads (Split in 3D: Z stack of X-Y grids)
        pad_segment_h = self.H / CORNER_PADS_NUMS
        pad_h_actual = pad_segment_h / (1.0 + GAP_RATIO)
        pad_z_half = pad_h_actual / 2.0
        
        # Sub-split in X-Y plane
        # Original single pad size was 2*PAD_XY (since PAD_XY was half-length)
        total_pad_full_width = 2.0 * PAD_XY
        pad_sub_full_width = total_pad_full_width / CORNER_SPLIT_XY
        pad_sub_half = (pad_sub_full_width / (1.0 + GAP_RATIO)) / 2.0
        
        vertical_edges = [(0, 1), (2, 3), (4, 5), (6, 7)]
        for edge_idx, (idx_bottom, idx_top) in enumerate(vertical_edges):
            c_bottom = self.corners_local[idx_bottom]
            c_top = self.corners_local[idx_top]
            sign_x = np.sign(c_bottom[0])
            sign_y = np.sign(c_bottom[1])
            
            # Base corner position (Inner corner of the pad footprint: 200mm inward)
            base_corner_x = c_bottom[0] - sign_x * (total_pad_full_width + BOX_PAD_OFFSET)
            base_corner_y = c_bottom[1] - sign_y * (total_pad_full_width + BOX_PAD_OFFSET)
            
            for i in range(CORNER_PADS_NUMS):
                # Z-position
                t_z = (i + 0.5) / CORNER_PADS_NUMS
                z_pos = c_bottom[2] + (c_top[2] - c_bottom[2]) * t_z
                
                for ix in range(CORNER_SPLIT_XY):
                    for iy in range(CORNER_SPLIT_XY):
                        t_x = (ix + 0.5) / CORNER_SPLIT_XY
                        t_y = (iy + 0.5) / CORNER_SPLIT_XY
                        
                        # Color logic: Outer (Green) to Inner (Gray)
                        layer = max(ix, iy)
                        if layer >= 2: 
                            rgba = "0.2 0.7 0.3 1.0"  # Green (Outer)
                        elif layer == 1:
                            rgba = "0.2 0.4 0.8 1.0"  # Blue (Middle)
                        else:
                            rgba = "0.5 0.5 0.5 1.0"  # Gray (Inner)

                        # Local offsets within the total_pad_full_width box
                        offset_x = sign_x * (total_pad_full_width * t_x)
                        offset_y = sign_y * (total_pad_full_width * t_y)
                        
                        configs.append({
                            'name_suffix': f"v_corner_{edge_idx}_z{i}_x{ix}_y{iy}",
                            'pos': [base_corner_x + offset_x, base_corner_y + offset_y, z_pos],
                            'size': [pad_sub_half, pad_sub_half, pad_z_half],
                            'rgba': rgba
                        })

        # 2. Horizontal Edge Blocks
        blk_z = self.H / 2.0
        total_pad_full_width = 2.0 * PAD_XY
        fb_sx = max(self.L/2.0 - total_pad_full_width - BOX_PAD_OFFSET, 0.001)
        fb_pos_y = self.W/2.0 - PAD_XY - BOX_PAD_OFFSET
        lr_sy = max(self.W/2.0 - total_pad_full_width - BOX_PAD_OFFSET, 0.001)
        lr_pos_x = self.L/2.0 - PAD_XY - BOX_PAD_OFFSET
        
        # Front/Back Edges
        for side in [-1, 1]:
            name_base = 'back' if side == 1 else 'front'
            if self.use_flex:
                num_seg = 3
                seg_len = (fb_sx * 2.0) / num_seg
                for s in range(num_seg):
                    pos_x = -fb_sx + (s + 0.5) * seg_len
                    configs.append({
                        'name_suffix': f"h_edge_{name_base}_seg{s}",
                        'pos': [pos_x, side * fb_pos_y, 0],
                        'size': [seg_len/2, PAD_XY, blk_z],
                        'rgba': "0.9 0.9 0.2 1.0"
                    })
            else:
                configs.append({
                    'name_suffix': f"h_edge_{name_base}_solid",
                    'pos': [0, side * fb_pos_y, 0],
                    'size': [fb_sx, PAD_XY, blk_z],
                    'rgba': "0.9 0.9 0.2 1.0"
                })

        # Left/Right Edges
        for side in [-1, 1]:
            name_base = 'right' if side == 1 else 'left'
            if self.use_flex:
                num_seg = 3
                seg_len = (lr_sy * 2.0) / num_seg
                for s in range(num_seg):
                    pos_y = -lr_sy + (s + 0.5) * seg_len
                    configs.append({
                        'name_suffix': f"h_edge_{name_base}_seg{s}",
                        'pos': [side * lr_pos_x, pos_y, 0],
                        'size': [PAD_XY, seg_len/2, blk_z],
                        'rgba': "0.9 0.9 0.2 1.0"
                    })
            else:
                configs.append({
                    'name_suffix': f"h_edge_{name_base}_solid",
                    'pos': [side * lr_pos_x, 0, 0],
                    'size': [PAD_XY, lr_sy, blk_z],
                    'rgba': "0.9 0.9 0.2 1.0"
                })
        return configs
        
        return configs

    def get_xml(self):
        """Generate MJCF XML string for this instance"""
        # Collision Bitmask Logic: 
        # Each instance gets a unique bit (0~31). 
        # Only geoms with matching bitmask collide.
        # 1. Floor for this instance
        # Bit pairs for toggling inter-box contact
        # Group i: bits 2i and 2i+1
        idx = self.uid % 15
        bit_floor = 1 << (2 * idx)
        bit_box = 1 << (2 * idx + 1)
        
        # Floor: contype = floor_bit, conaffinity = box_bit
        f_half = max(self.L, self.W) * 0.8
        floor_xml = f"""
        <geom name="floor_{self.uid}" type="plane" 
              pos="{self.base_pos[0]} {self.base_pos[1]} 0" zaxis="0 0 1" 
              size="{f_half} {f_half} 1" material="grid" 
              friction="{BOX_FRICTION_PARAMS}" solref="0.01 1"
              contype="{bit_floor}" conaffinity="{bit_box}"/>
        """
        
        # 2. Body Definition
        start_pos = [self.base_pos[0], self.base_pos[1], self.initial_z_offset]
        
        # Inertia
        Ixx = (1/12) * self.MASS * (self.W**2 + self.H**2)
        Iyy = (1/12) * self.MASS * (self.L**2 + self.H**2)
        Izz = (1/12) * self.MASS * (self.L**2 + self.W**2)
        
        # Pads Geometry
        # Box: contype = box_bit, conaffinity = floor_bit (| box_bit if contact enabled)
        affinity_box = bit_floor
        if self.enable_pad_contact:
            affinity_box |= bit_box
            
        solref_str = f"{DEFAULT_SOLREF[0]:.5f} {DEFAULT_SOLREF[1]:.5f}"
        solimp_str = " ".join(map(str, DEFAULT_SOLIMP))

        # --- Sub-Body Generation for Flex Mode ---
        def get_geoms_in_range(x_range, y_range, offset_x=0, offset_y=0):
            """Filter pads that belong to a specific XY segment and convert to LOCAL segment coords"""
            xml = ""
            for pad in self.pad_configs:
                px, py = pad['pos'][0], pad['pos'][1]
                if x_range[0] <= px <= x_range[1] and y_range[0] <= py <= y_range[1]:
                    p_name = f"box_{self.uid}_{pad['name_suffix']}"
                    # Convert global box pos to local segment pos
                    lx, ly = px - offset_x, py - offset_y
                    xml += f"""
                    <geom name="{p_name}" type="box" size="{pad['size'][0]} {pad['size'][1]} {pad['size'][2]}"
                          pos="{lx} {ly} {pad['pos'][2]}"
                          rgba="{pad['rgba']}" solref="{solref_str}" solimp="{solimp_str}"
                          friction="{BOX_FRICTION_PARAMS}" contype="{bit_box}" conaffinity="{affinity_box}"/>
                    """
            return xml

        if self.use_flex:
            # 3x3 Grid (9 Bodies)
            body_xml = ""
            sl, sw = self.L / 3.0, self.W / 3.0
            sm = self.MASS / 9.0
            six, siy, siz = Ixx/9.0, Iyy/9.0, Izz/9.0
            
            # Helper to generate a segment body
            def make_seg(name, grid_x, grid_y):
                # Local coords for the segment center
                lx, ly = grid_x * sl, grid_y * sw
                # Range for geom filtering
                rx = [lx - sl/2 - 0.01, lx + sl/2 + 0.01]
                ry = [ly - sw/2 - 0.01, ly + sw/2 + 0.01]
                
                content = f'<inertial pos="0 0 0" mass="{sm}" diaginertia="{six} {siy} {siz}"/>\n'
                # Visual Box Piece
                content += f'<geom name="box_{self.uid}_vis_{name}" type="box" size="{sl/2} {sw/2} {self.H/2}" rgba="0.8 0.6 0.3 0.2" contype="0" conaffinity="0"/>\n'
                
                # Distribution of pads/brands
                if grid_x == 0 and grid_y == 0:
                    content += f'<geom name="box_{self.uid}_brand" type="box" size="{sl*0.4} {sw*0.4} 0.002" pos="0 0 {self.H/2 + 0.001}" rgba="0.1 0.2 0.6 1.0" mass="0.001"/>\n'
                    content += f'<site name="box_{self.uid}_label" pos="0 0 {self.H/2 + 0.01}" size="0.01" rgba="0 0 0 0"/>\n'
                    content += f'<site name="s_center_{self.uid}" pos="0 0 0" size="0.01" rgba="1 1 0 1"/>\n'
                
                # Distribution of Corner Sites
                for i, c in enumerate(self.corners_local):
                    # Check if corner belongs to this segment's grid range
                    if grid_x == np.sign(c[0]) and grid_y == np.sign(c[1]):
                        # Local pos relative to segment center
                        lx, ly = grid_x * sl, grid_y * sw
                        content += f'<site name="s_corner_{self.uid}_{i}" pos="{c[0]-lx} {c[1]-ly} {c[2]}" size="0.005" rgba="0.5 0.5 0.5 0.2"/>\n'

                content += get_geoms_in_range(rx, ry, lx, ly)
                
                if name == "c": # Root
                    return f"""
                    <body name="box_{self.uid}_c" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}" quat="{self.quat_mj[0]} {self.quat_mj[1]} {self.quat_mj[2]} {self.quat_mj[3]}">
                        <freejoint name="joint_{self.uid}"/>
                        {content}
                    """
                else: # Child connected via spring
                    return f"""
                    <body name="box_{self.uid}_{name}" pos="{lx} {ly} 0">
                        <joint name="j_{self.uid}_{name}_x" type="hinge" axis="1 0 0" stiffness="{self.flex_stiffness}" damping="{self.flex_damping}"/>
                        <joint name="j_{self.uid}_{name}_y" type="hinge" axis="0 1 0" stiffness="{self.flex_stiffness}" damping="{self.flex_damping}"/>
                        {content}
                    </body>"""

            # Combine 9 segments
            body_xml += make_seg("c", 0, 0)
            for gx in [-1, 0, 1]:
                for gy in [-1, 0, 1]:
                    if gx == 0 and gy == 0: continue
                    name = f"{'s' if gy < 0 else ('n' if gy > 0 else '')}{'w' if gx < 0 else ('e' if gx > 0 else '')}"
                    body_xml += make_seg(name, gx, gy)
            body_xml += "</body>"
        else:
            # Original Rigid Body logic
            pads_xml = ""
            for pad in self.pad_configs:
                p_name = f"box_{self.uid}_{pad['name_suffix']}"
                pads_xml += f"""
                <geom name="{p_name}" type="box" size="{pad['size'][0]} {pad['size'][1]} {pad['size'][2]}"
                      pos="{pad['pos'][0]} {pad['pos'][1]} {pad['pos'][2]}"
                      rgba="{pad['rgba']}" solref="{solref_str}" solimp="{solimp_str}"
                      friction="{BOX_FRICTION_PARAMS}" contype="{bit_box}" conaffinity="{affinity_box}"/>
                """
                
            body_xml = f"""
            <body name="box_{self.uid}" pos="{start_pos[0]} {start_pos[1]} {start_pos[2]}" 
                  quat="{self.quat_mj[0]} {self.quat_mj[1]} {self.quat_mj[2]} {self.quat_mj[3]}">
                <freejoint name="joint_{self.uid}"/>
                <inertial pos="{self.CoM_offset[0]} {self.CoM_offset[1]} {self.CoM_offset[2]}" mass="{self.MASS}" diaginertia="{Ixx} {Iyy} {Izz}"/>
                
                <!-- Samsung Branding Block -->
                <geom name="box_{self.uid}_brand" type="box" size="{self.L*0.3} {self.W*0.3} 0.002" 
                      pos="0 0 {self.H/2 + 0.001}" rgba="0.1 0.2 0.6 1.0" mass="0.001"/>
                <site name="box_{self.uid}_label" pos="0 0 {self.H/2 + 0.01}" size="0.01" rgba="0 0 0 0"/>
    
                <!-- Main Visual -->
                <geom name="box_{self.uid}_visual" type="box" size="{self.L/2} {self.W/2} {self.H/2}" 
                      rgba="0.8 0.6 0.3 0.3" contype="0" conaffinity="0"
                      fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 1.0 1.0"/>
                
                {pads_xml}
                
                <!-- Sites for Sensing -->
                <site name="s_center_{self.uid}" pos="0 0 0" size="0.01" rgba="1 1 0 1"/>
                {self._get_corner_sites_xml()}
            </body>
            """
        return floor_xml + body_xml

    def _get_corner_sites_xml(self):
        """Generate XML for 8 sensing sites at the corners"""
        xml = ""
        for i, c in enumerate(self.corners_local):
            xml += f'<site name="s_corner_{self.uid}_{i}" pos="{c[0]} {c[1]} {c[2]}" size="0.005" rgba="0.5 0.5 0.5 0.2"/>\n'
        return xml

def _run_single_instance_worker(args):
    """Worker function for parallel simulation (runs in separate process)"""
    inst, use_ac, use_pl, solver_mode, duration, record_details = args
    
    # Each process needs its own Manager and Model
    sim = SimulationManager(enable_air_cushion=use_ac, enable_plasticity=use_pl, solver_mode=solver_mode)
    
    # We want to simulate this instance at its designated position
    sim.add_instance(inst)
    sim.init_simulation()
    
    # Run simulation (this will update inst.history)
    sim.run_headless(duration=duration, record_details=record_details)
    
    return inst


# ==========================================
# Class: SimulationManager
# ==========================================
class SimulationManager:
    def __init__(self, enable_air_cushion=True, enable_plasticity=True, solver_mode="2"):
        self.instances = []
        self.model = None
        self.data = None
        self.geom_state_tracker = {}
        
        # Feature Toggles
        self.enable_air_cushion = enable_air_cushion
        self.enable_plasticity = enable_plasticity
        
        # Solver Configuration
        self.solver_mode = solver_mode # Store for singleton re-runs
        self.solver_params = SOLVER_PRESETS.get(solver_mode, SOLVER_PRESETS["2"])
        print(f"⚙️  Solver Setting: {self.solver_params['name']}")
        
    def add_instance(self, instance):
        self.instances.append(instance)
        
    def generate_full_xml(self):
        body_str = ""
        for inst in self.instances:
            body_str += inst.get_xml()
            
        xml = f"""
        <mujoco>
          <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".8 .8 .8" rgb2=".9 .9 .9" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="8 8" texuniform="true"/>
          </asset>
          
          <option timestep="0.001" gravity="0 0 -{G_ACC}" density="{AIR_DENSITY}" viscosity="{AIR_VISCOSITY}"
                  iterations="{self.solver_params['iter']}" 
                  tolerance="{self.solver_params['tol']}" 
                  solver="{self.solver_params['solver']}" 
                  cone="{self.solver_params['cone']}">
            <flag contact="enable"/>
          </option>
          
          <worldbody>
            <light pos="10 10 20" dir="-1 -1 -1" diffuse="0.7 0.7 0.7"/>
            {body_str}
          </worldbody>
        </mujoco>
        """
        return xml

    def init_simulation(self):
        # 0. DE-REGISTER global callbacks before reloading model to prevent "Python exception raised"
        mujoco.set_mjcb_control(None)
        
        # 1. Clean up previous resources
        try:
            if hasattr(self, 'model') and self.model: del self.model
            if hasattr(self, 'data') and self.data: del self.data
        except: pass
        self.model = None
        self.data = None
        
        # 2. Generate and Load with Retry
        xml_str = self.generate_full_xml()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # MuJoCo parser might trigger callbacks, so ensure it's None
                self.model = mujoco.MjModel.from_xml_string(xml_str)
                self.data = mujoco.MjData(self.model)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ XML Loading Error: {e}")
                    with open("failed_model.xml", "w", encoding='utf-8') as f:
                        f.write(xml_str)
                    print("   -> Dumped XML to failed_model.xml")
                    raise e
                print(f"⚠️  Retrying XML Load due to engine busy (Attempt {attempt+2}/{max_retries})...")
                time.sleep(1.0)
        
        # 3. Reset tracker & Initial velocity
        self.geom_state_tracker = {}
        np.random.seed(42)
        for i, inst in enumerate(self.instances):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{inst.uid}")
            if joint_id != -1:
                qvel_adr = self.model.jnt_dofadr[joint_id]
                angvel = np.random.uniform(-0.003, 0.003, 3)
                self.data.qvel[qvel_adr+3 : qvel_adr+6] = angvel

        # 4. Store initial state for reset (DEEP COPY for geometry reset)
        self.initial_geom_pos = np.array(self.model.geom_pos).copy()
        self.initial_geom_size = np.array(self.model.geom_size).copy()
        
        # 5. RE-REGISTER callback after model/data are ready
        mujoco.set_mjcb_control(self._cb_control_wrapper)
        print(f"✅ Simulation Initialized ({len(self.instances)} instances)")

    def _cb_control_wrapper(self, model, data):
        # 1. Apply Air Cushion (if enabled)
        if self.enable_air_cushion:
            self.apply_air_cushion(model, data)
        # 2. Plastic Deformation is applied POST-STEP outside of control callback in main loop
        
    def apply_air_cushion(self, model, data):
        """Multi-body Air Cushion Logic"""
        for i, inst in enumerate(self.instances):
            body_name = f"box_{inst.uid}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1: continue
            
            # Retrieve State
            pos = data.xpos[body_id]
            rmat = data.xmat[body_id].reshape(3, 3)
            # Free joint qvel: [lin, ang] or [ang, lin]? 
            # In qvel array for freejoint: 0-2 translation, 3-5 rotation
            # But we generated flattened structure.
            # Safe way: use cvel (spatial velocity at CoM) or explicit qvel index
            # Simplified for demo:
            jnt_adr = model.body_jntadr[body_id]
            qvel_adr = model.jnt_dofadr[jnt_adr]
            lin_vel = data.qvel[qvel_adr:qvel_adr+3]
            ang_vel = data.qvel[qvel_adr+3:qvel_adr+6]

            # Face Detection Logic (Similar to previous, but local to this instance)
            xaxis, yaxis, zaxis = rmat[:, 0], rmat[:, 1], rmat[:, 2]
            dots = [xaxis[2], yaxis[2], zaxis[2]]
            abs_dots = [abs(d) for d in dots]
            axis_idx = np.argmax(abs_dots)
            sign = np.sign(dots[axis_idx])

            # Dimension selection
            if axis_idx == 0:   # X-face
                local_normal_dist = inst.L/2.0 * sign * np.array([1,0,0])
                u_len, v_len = inst.W, inst.H
                u_vec, v_vec = yaxis, zaxis
            elif axis_idx == 1: # Y-face
                local_normal_dist = inst.W/2.0 * sign * np.array([0,1,0])
                u_len, v_len = inst.L, inst.H
                u_vec, v_vec = xaxis, zaxis
            else:               # Z-face
                local_normal_dist = inst.H/2.0 * sign * np.array([0,0,1])
                u_len, v_len = inst.L, inst.W
                u_vec, v_vec = xaxis, yaxis

            if dots[axis_idx] > 0: # Normal points UP, flip
                local_normal_dist = -local_normal_dist
            
            # Grid Integration
            N = 4 # Reduced grid for performance in multi-body
            dA = (u_len * v_len) / (N * N)
            
            total_force_z = 0.0
            total_torque = np.zeros(3)
            
            grid_steps = np.linspace(-0.5 + 0.5/N, 0.5 - 0.5/N, N)
            body_pos = data.qpos[qvel_adr - 7 : qvel_adr - 4] # Approx pos from qpos (freejoint has 7 qpos)
            # Better: data.xpos[body_id] is CoM world pos
            
            face_center_world_vec = rmat @ local_normal_dist
            geometric_factor = ((inst.L * inst.W) / (2 * (inst.L + inst.W))) ** 2
            PHYSICS_COEF = 0.5 * AIR_DENSITY * geometric_factor # Coef=1.0 for simplicity
            
            for u in grid_steps:
                for v in grid_steps:
                    rel_pos = face_center_world_vec + (u * u_len) * u_vec + (v * v_len) * v_vec
                    point_pos = pos + rel_pos
                    h = point_pos[2] # Global Z
                    
                    if h < 0.001 or h > 0.2: continue
                    
                    point_vel = lin_vel + np.cross(ang_vel, rel_pos)
                    v_z = point_vel[2]
                    
                    if v_z < 0:
                        safe_h = max(h, 0.001)
                        dF = PHYSICS_COEF * dA * (v_z / safe_h)**2
                        dF = min(dF, 500.0) # Lower cap for stability
                        total_force_z += dF
                        total_torque[0] += rel_pos[1] * dF
                        total_torque[1] -= rel_pos[0] * dF
            
            data.xfrc_applied[body_id][2] = total_force_z
            data.xfrc_applied[body_id][3:6] = total_torque

    def apply_plastic_deformation(self):
        """Apply Hysteretic Plastic Deformation on all active contacts"""
        # Group contacts by geom
        current_penetrations = {}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            # Identify if contact involves a valid deformable geom (pad)
            # Check names of g1, g2
            g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            
            target_geom = -1
            if g1_name and "floor" in g1_name: target_geom = con.geom2
            elif g2_name and "floor" in g2_name: target_geom = con.geom1
            
            if target_geom != -1:
                pen = -con.dist
                if pen > 1e-4:
                    if pen > current_penetrations.get(target_geom, 0.0):
                        current_penetrations[target_geom] = pen

        # 1. Register new
        for gid in current_penetrations:
            if gid not in self.geom_state_tracker:
                self.geom_state_tracker[gid] = {'max_p': 0.0, 'applied': False}
        
        # 2. Logic (Iterate all tracked)
        for gid, state in self.geom_state_tracker.items():
            g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            # Filter for deformable components: corner pads and edge blocks
            if not g_name or ("v_corner" not in g_name and "h_edge" not in g_name and "pad" not in g_name): 
                continue
                
            curr_p = current_penetrations.get(gid, 0.0)
            
            # Compression
            if curr_p >= state['max_p']:
                state['max_p'] = curr_p
                state['applied'] = False
            
            # Recovery
            if state['max_p'] > 0.001 and not state['applied']:
                recovery = state['max_p'] - curr_p
                if recovery >= state['max_p'] * PLASTIC_DEFORMATION_RATIO:
                    # Apply
                    deformation = state['max_p'] * PLASTIC_DEFORMATION_RATIO
                    
                    # Inward direction
                    # Local pos is relative to body. 
                    # model.geom_pos is local to body.
                    # We simply move towards 0,0,0 of body frame.
                    # Careful: Pads are offset.
                    # Simple heuristic: move opposite to current local pos sign
                    local_pos = self.model.geom_pos[gid]
                    inward_dir = -np.sign(local_pos[:3])
                    
                    shrink = deformation * 0.2
                    shift = deformation * 0.8
                    
                    self.model.geom_size[gid][0] = max(0.001, self.model.geom_size[gid][0] - shrink)
                    self.model.geom_size[gid][1] = max(0.001, self.model.geom_size[gid][1] - shrink)
                    
                    self.model.geom_pos[gid] += inward_dir * shift
                    
                    state['applied'] = True
            
            if curr_p == 0.0 and state['applied']:
                state['max_p'] = 0.0
                state['applied'] = False

    def run_simulation_loop(self, mode='passive'):
        """Main Loop for Visual Simulation"""
        if mode == 'standard':
            # Standard blocking
            print("🎮 Opening Standard Viewer...")
            print("   Space: Pause/Resume")
            
            with mujoco.viewer.launch(self.model, self.data) as viewer:
                while viewer.is_running():
                    # We can't intervene easily in standard viewer.
                    time.sleep(0.1)
        else:
            # Passive with Custom Loop using key_callback
            print("🎮 Opening Passive Viewer (Custom Control)...")
            print("   [Space]: Toggle Pause/Resume")
            print("   [Right Arrow]: Step (when paused)")
            print("   [Backspace]: Reset")
            print("   [-/=]: Slower/Faster")
            print("   [Esc]: Quit")
            
            # State variables for callback
            self.paused = True
            self.reset_trigger = False
            self.step_trigger = False
            self.should_quit = False
            self.slow_motion = 1.0
            
            def key_callback(keycode):
                # Spacebar (32)
                if keycode == 32:
                    self.paused = not self.paused
                    print(f"   {'[PAUSED]' if self.paused else '[RUNNING]'}")
                
                # Right Arrow (262)
                elif keycode == 262 and self.paused:
                    self.step_trigger = True
                    
                # Backspace (259) or R (82)
                elif keycode == 259 or keycode == 82:
                    self.reset_trigger = True
                    
                # ESC (256)
                elif keycode == 256:
                    self.should_quit = True
                    
                # Minus (-)
                elif keycode == 45: 
                    self.slow_motion = min(self.slow_motion + 1.0, 20.0)
                    print(f"   [SPEED] Slower -> 1/{self.slow_motion:.1f}x")
                    
                # Equal (=)
                elif keycode == 61:
                    self.slow_motion = max(self.slow_motion - 1.0, 1.0)
                    print(f"   [SPEED] Faster -> 1/{self.slow_motion:.1f}x")

            # Reset logic wrapper
            def perform_reset():
                print("   [RESET]")
                self.reset_sim()
                self.paused = True
            
            # Initial Reset
            self.reset_sim()
            
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
                if len(self.instances) <= 1:
                    viewer.cam.distance = 2.5
                    viewer.cam.lookat = [0, 0, 0.2] # Slightly above ground
                else:
                    viewer.cam.distance = 25.0
                    viewer.cam.lookat = [10, 8, 0]
                viewer.sync()
                
                step_start = time.time()
                
                while viewer.is_running():
                    # Draw Labels (Manual Marker Injection for Compatibility)
                    try:
                        viewer.user_scn.ngeom = 0 # Reset user geoms each frame
                        for inst in self.instances:
                            # --- Real-time Stats Calculation ---
                            # 1. Bending
                            curr_flex = 0.0
                            if inst.use_flex:
                                sl, sw = inst.L/3.0, inst.W/3.0
                                seg_dists = {
                                    'n': sw, 's': sw, 'e': sl, 'w': sl,
                                    'ne': np.sqrt(sl**2+sw**2), 'nw': np.sqrt(sl**2+sw**2),
                                    'se': np.sqrt(sl**2+sw**2), 'sw': np.sqrt(sl**2+sw**2)
                                }
                                for sfx, dist in seg_dists.items():
                                    for axis in ['x', 'y']:
                                        jname = f"j_{inst.uid}_{sfx}_{axis}"
                                        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                                        if jid != -1:
                                            angle = abs(self.data.qpos[self.model.jnt_qposadr[jid]])
                                            defl = dist * np.sin(angle) * 1000.0
                                            if defl > curr_flex: curr_flex = defl
                            
                            # 2. Avg Pad Compression
                            total_comp, count = 0.0, 0
                            prefix = f"box_{inst.uid}_v_corner"
                            for i in range(self.model.ngeom):
                                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                                if name and name.startswith(prefix):
                                    comp = (self.initial_geom_size[i, 2] - self.model.geom_size[i, 2]) * 1000.0
                                    total_comp += max(0, comp)
                                    count += 1
                            avg_comp = total_comp / count if count > 0 else 0.0
                            
                            # Update Label string
                            display_text = f"{inst.label}\n[Flex: {curr_flex:.1f}mm | Comp: {avg_comp:.1f}mm]"

                            if viewer.user_scn.ngeom < 100:
                                f_half = max(inst.L, inst.W) * 0.8
                                l_pos = [inst.base_pos[0], inst.base_pos[1] - f_half - 0.4, 0.1]
                                
                                idx = viewer.user_scn.ngeom
                                viewer.user_scn.ngeom += 1
                                mujoco.mjv_initGeom(
                                    viewer.user_scn.geoms[idx],
                                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=[0.01, 0.01, 0.01],
                                    pos=l_pos,
                                    mat=np.eye(3).flatten(),
                                    rgba=[1, 1, 1, 0] 
                                )
                                viewer.user_scn.geoms[idx].label = display_text
                    except:
                        pass # Ignore errors if viewer is closing

                    if self.should_quit:
                        viewer.close()
                        break
                        
                    if self.reset_trigger:
                        perform_reset()
                        self.reset_trigger = False
                        viewer.sync()
                        continue
                        
                    if self.step_trigger:
                        self.step_trigger = False
                        mujoco.mj_step(self.model, self.data)
                        self.apply_plastic_deformation()
                        viewer.sync()
                        print("   [STEP] +1")
                        
                    if not self.paused:
                        loop_start = time.time()
                        
                        # Optimization: Run multiple physics steps per render frame
                        # This aims for Real-Time Performance even with heavy calculation
                        BATCH_STEPS = 5 # 5ms physics per frame
                        
                        for _ in range(BATCH_STEPS):
                            mujoco.mj_step(self.model, self.data)
                            if self.enable_plasticity:
                                self.apply_plastic_deformation()
                        
                        viewer.sync()
                        
                        # Sync Speed
                        # We simulated BATCH_STEPS * 0.001 seconds
                        target_dt = (BATCH_STEPS * 0.001) * self.slow_motion
                        elapsed = time.time() - loop_start
                        if elapsed < target_dt:
                            time.sleep(target_dt - elapsed)
                    else:
                        viewer.sync()
                        time.sleep(0.01)

    def reset_sim(self):
        """Reset simulation state and Restore original geometry (Moved pads)"""
        # 1. Reset Physics State
        mujoco.mj_resetData(self.model, self.data)
        
        # 2. Restore Geometry (Reset plasticity effects)
        if hasattr(self, 'initial_geom_pos'):
            # Use np.copyto for direct memory update in MuJoCo structs
            np.copyto(self.model.geom_pos, self.initial_geom_pos)
            np.copyto(self.model.geom_size, self.initial_geom_size)
            
        # 3. Clear trackers
        self.geom_state_tracker = {}
        
        # 4. Re-apply Initial Perturbation (Angular Velocity)
        np.random.seed(42)
        for i, inst in enumerate(self.instances):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{inst.uid}")
            if joint_id != -1:
                qvel_adr = self.model.jnt_dofadr[joint_id]
                angvel = np.random.uniform(-0.003, 0.003, 3)
                self.data.qvel[qvel_adr+3 : qvel_adr+6] = angvel
        
        # 5. Force update of collision/visual structures
        mujoco.mj_forward(self.model, self.data)
        print("   [GEOMETRY RESTORED]")

    def run_headless(self, duration=2.5, record_details=False):
        print(f"🚀 Running Headless Simulation for {duration}s... (Record Details: {record_details})")
        steps = int(duration / 0.001)
        
        # Initial Forward to settle state
        mujoco.mj_forward(self.model, self.data)
        
        for step_idx in range(steps):
            mujoco.mj_step(self.model, self.data)
            if self.enable_plasticity:
                self.apply_plastic_deformation()
            
            t = self.data.time
            # Progress dot every 0.25s
            if step_idx % 250 == 0: print(".", end="", flush=True)

            for inst in self.instances:
                # For flex body, the "main" body is the root segment named 'box_uid_c'
                bname = f"box_{inst.uid}_c" if inst.use_flex else f"box_{inst.uid}"
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
                if bid != -1:
                    # 1. Basic Recording
                    inst.history['time'].append(t)
                    curr_pos = self.data.xpos[bid].copy()
                    inst.history['pos'].append(curr_pos)
                    curr_vel = self.data.cvel[bid][3:6].copy()
                    inst.history['vel'].append(curr_vel)
                    
                    # 3. Flexible Bending/Deflection & Joint Recording (Core Flex Metrics)
                    if inst.use_flex:
                        max_angle = 0.0
                        max_defl_mm = 0.0
                        sl, sw = inst.L/3.0, inst.W/3.0
                        seg_dists = {
                            'n': sw, 's': sw, 'e': sl, 'w': sl,
                            'ne': np.sqrt(sl**2+sw**2), 'nw': np.sqrt(sl**2+sw**2),
                            'se': np.sqrt(sl**2+sw**2), 'sw': np.sqrt(sl**2+sw**2)
                        }
                        
                        for sfx, dist in seg_dists.items():
                            for axis in ['x', 'y']:
                                jname = f"j_{inst.uid}_{sfx}_{axis}"
                                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                                if jid != -1:
                                    angle = abs(self.data.qpos[self.model.jnt_qposadr[jid]])
                                    if angle > max_angle: max_angle = angle
                                    
                                    # Record individual joint (Always record if flex enabled for consistent plots)
                                    j_key = f"{sfx}_{axis}"
                                    if j_key not in inst.history['joint_angles']:
                                        inst.history['joint_angles'][j_key] = []
                                    inst.history['joint_angles'][j_key].append(angle * 180.0 / np.pi)
                                    
                                    defl = dist * np.sin(angle) * 1000.0
                                    if defl > max_defl_mm: max_defl_mm = defl
                                    
                        inst.history['bending'].append(max_angle)
                        inst.history['deflection'].append(max_defl_mm)
                    else:
                        inst.history['bending'].append(0.0)
                        inst.history['deflection'].append(0.0)

                    # 2. Detailed Recording (if requested)
                    if record_details:
                        # Acceleration (Numerical Diff)
                        acc = (curr_vel - inst.prev_vel) / 0.001
                        inst.history['acc'].append(acc)
                        inst.prev_vel = curr_vel.copy()
                        
                        # Air Cushion Force (Applied Z force)
                        ext_force = self.data.xfrc_applied[bid][2]
                        inst.history['cushion_force'].append(ext_force)
                        
                        # Impact Force (Sum of Normal Forces for this instance)
                        total_impact = 0.0
                        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"floor_{inst.uid}")
                        for i in range(self.data.ncon):
                            con = self.data.contact[i]
                            if con.geom1 == floor_id or con.geom2 == floor_id:
                                # Normal force is index 0 of contact force in local frame
                                c_force = np.zeros(6)
                                mujoco.mj_contactForce(self.model, self.data, i, c_force)
                                total_impact += c_force[0]
                        inst.history['impact_force'].append(total_impact)
                        
                        # Virtual Corners (8 Sites)
                        for i in range(8):
                            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"s_corner_{inst.uid}_{i}")
                            if sid != -1:
                                pos = self.data.site_xpos[sid].copy()
                                inst.history['corner_pos'][i].append(pos)
                                
                                # Corner Velocity mapping (Linear only)
                                obj_vel = np.zeros(6)
                                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, sid, obj_vel, 0)
                                c_vel = obj_vel[3:6].copy()
                                inst.history['corner_vel'][i].append(c_vel)
                                
                                # Corner Acc (Numerical Diff)
                                c_acc = (c_vel - inst.prev_corner_vels[i]) / 0.001
                                inst.history['corner_acc'][i].append(c_acc)
                                inst.prev_corner_vels[i] = c_vel.copy()
                            
                        # 4. Detailed Pad Compression Recording (Time-series)
                        for edge_idx in range(4):
                            prefix = f"box_{inst.uid}_v_corner_{edge_idx}_"
                            total_c, count = 0.0, 0
                            for i in range(self.model.ngeom):
                                gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                                if gname and gname.startswith(prefix):
                                    comp = (self.initial_geom_size[i, 2] - self.model.geom_size[i, 2]) * 1000.0
                                    total_c += max(0, comp)
                                    count += 1
                            
                            c_key = f"Corner_{edge_idx}"
                            if c_key not in inst.history['corner_comp']:
                                inst.history['corner_comp'][c_key] = []
                            inst.history['corner_comp'][c_key].append(total_c / count if count > 0 else 0.0)

            # Record final state at the very end
            if step_idx == steps - 1:
                for inst in self.instances:
                    for i in range(self.model.ngeom):
                        gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                        if gname and f"box_{inst.uid}_v_corner" in gname:
                            inst.history['final_pad_comp'][gname] = (self.initial_geom_size[i, 2] - self.model.geom_size[i, 2]) * 1000.0

        print("\n✅ Simulation Complete.")

    def run_headless_parallel(self, duration=2.5, record_details=False, max_workers=None):
        """Run all instances in parallel using multiple CPU cores"""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(self.instances))
        
        print(f"🚀 Running Parallel Headless Simulation ({len(self.instances)} instances on {max_workers} cores)...")
        start_t = time.time()
        
        # Prepare arguments for workers
        tasks = []
        for inst in self.instances:
            tasks.append((inst, self.enable_air_cushion, self.enable_plasticity, self.solver_mode, duration, record_details))
            
        new_instances = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single_instance_worker, task) for task in tasks]
            
            completed = 0
            for future in as_completed(futures):
                inst_result = future.result()
                new_instances.append(inst_result)
                completed += 1
                if completed % 1 == 0:
                    print(f"\r   Progress: {completed}/{len(self.instances)} tasks finished...", end="", flush=True)
        
        # Sort back by UID to maintain order
        new_instances.sort(key=lambda x: x.uid)
        self.instances = new_instances
        
        print(f"\n✅ All Parallel Tasks Complete. (Elapsed: {time.time() - start_t:.2f}s)")

# ==========================================
# TESTCASE: Helper to ask for common configs
# ==========================================
def ask_common_configs():
    print("\n🔧 Physics Configuration:")
    ac_in = input("Enable Air Cushion (y/n, default y): ").strip().lower()
    pl_in = input("Enable Plastic Deformation (y/n, default y): ").strip().lower()
    pc_in = input("Enable Pad-to-Pad Contact (y/n, default y): ").strip().lower()
    fx_in = input("Enable Flexible Body (y/n, default n): ").strip().lower()
    
    use_ac = (ac_in not in ['n', 'ㅜ']) # Default is True, only False if 'n' or 'ㅜ'
    use_pl = (pl_in not in ['n', 'ㅜ'])
    use_pc = (pc_in not in ['n', 'ㅜ'])
    use_fx = (fx_in in ['y', 'ㅛ'])      # Default is False, only True if 'y' or 'ㅛ'
    
    fs_val = FLEX_STIFFNESS
    fd_val = FLEX_DAMPING
    if use_fx:
        # User might type 'ㅛ' or 'ㅜ' by mistake (y/n keys in Korean mode)
        fs_in = input(f"   - Set Stiffness (default {FLEX_STIFFNESS}): ").strip().lower()
        fd_in = input(f"   - Set Damping (default {FLEX_DAMPING}): ").strip().lower()
        
        # Helper to convert input, treating Korean 'ㅛ'/'ㅜ' as empty (defaults)
        def robust_float(val, default):
            if not val or val in ['y', 'n', 'ㅛ', 'ㅜ']: return default
            try: return float(val)
            except: return default
            
        fs_val = robust_float(fs_in, FLEX_STIFFNESS)
        fd_val = robust_float(fd_in, FLEX_DAMPING)

    mc_in = input("Enable Multi-core (y/n, default y): ").strip().lower()
    use_mc = (mc_in not in ['n', 'ㅜ'])
    
    print("\n🚀 Solver Configuration:")
    print("1. Accurate (정확 - Newton, Elliptic, Iter 100)")
    print("2. Normal   (보통 - Newton, Pyramidal, Iter 50)")
    print("3. Fast     (빠름 - PGS, Pyramidal, Iter 30)")
    print("4. Very Fast(매우빠름 - PGS, Pyramidal, Iter 20)")
    solver_mode = input("Select Solver Mode (1-4, default 3): ").strip()
    if not solver_mode: solver_mode = "3"
    
    print("\n📺 Viewer Configuration:")
    print("1. Standard Viewer")
    print("2. Passive Viewer (Custom Control, Recommended)")
    view_mode_in = input("Select Viewer Mode (1/2, default 2): ").strip()
    view_mode = 'standard' if view_mode_in == '1' else 'passive'
    
    return {
        'use_ac': use_ac,
        'use_pl': use_pl,
        'use_pc': use_pc,
        'use_fx': use_fx,
        'flex_stiffness': fs_val,
        'flex_damping': fd_val,
        'use_mc': use_mc,
        'solver_mode': solver_mode,
        'view_mode': view_mode
    }

# ==========================================
# TESTCASE: One Box Drop
# ==========================================
def run_testcase_OneBox():
    print("\n📦 Running Testcase: One Box Drop")
    cfg = ask_common_configs()
    
    sim = SimulationManager(enable_air_cushion=cfg['use_ac'], enable_plasticity=cfg['use_pl'], solver_mode=cfg['solver_mode'])
    sim.use_mc = cfg['use_mc']
    
    inst = BoxDropInstance(
        uid=0,
        position_offset=[0, 0, 0],
        drop_type="corner",
        box_params={}, 
        com_offset=[0, 0, 0],
        label="Single Box Drop",
        enable_pad_contact=cfg['use_pc'],
        use_flex=cfg['use_fx'],
        flex_stiffness=cfg['flex_stiffness'],
        flex_damping=cfg['flex_damping']
    )
    sim.add_instance(inst)
    sim.init_simulation()
    
    sim.run_simulation_loop(mode=cfg['view_mode'])
    
    print("\n� Starting Headless Data Collection...")
    sim.init_simulation()
    sim.run_headless(duration=2.5, record_details=True)
    
    plot_detailed_singleton_results(inst)
    return sim

# ==========================================
# TESTCASE: Random CoM Grid
# ==========================================
def run_testcase_CoM_Random():
    print("\n🎲 Running Testcase: Random CoM Grid (4x5)")
    cfg = ask_common_configs()
    
    sim = SimulationManager(enable_air_cushion=cfg['use_ac'], enable_plasticity=cfg['use_pl'], solver_mode=cfg['solver_mode'])
    sim.use_mc = cfg['use_mc']
    
    rows, cols = 4, 5
    spacing_x, spacing_y = 4.0, 4.0
    
    print(f"📦 Generating {rows*cols} Box Instances...")
    box_id = 0
    for r in range(rows):
        for c in range(cols):
            inst = BoxDropInstance(
                uid=box_id,
                position_offset=[c * spacing_x, r * spacing_y, 0],
                drop_type="corner",
                box_params={},
                com_random=True,
                enable_pad_contact=cfg['use_pc'],
                use_flex=cfg['use_fx'],
                flex_stiffness=cfg['flex_stiffness'],
                flex_damping=cfg['flex_damping']
            )
            cx, cy, cz = inst.CoM_offset * 1000.0
            inst.label = f"Box {box_id:02d} (Rand CoM: X{cx:+.1f} Y{cy:+.1f} Z{cz:+.1f})"
            sim.add_instance(inst)
            box_id += 1
            
    sim.init_simulation()
    sim.run_simulation_loop(mode=cfg['view_mode'])
    
    print("\n📊 Starting Headless Data Collection...")
    if sim.use_mc:
        sim.run_headless_parallel(duration=2.5, record_details=True)
    else:
        sim.init_simulation()
        sim.run_headless(duration=2.5, record_details=True)
    return sim

# ==========================================
# TESTCASE: Parametric CoM Sweep
# ==========================================
def run_testcase_CoM_Mov():
    print("\n🧪 Running Testcase: Parametric CoM Sweep")
    cfg = ask_common_configs()
    
    sim = SimulationManager(enable_air_cushion=cfg['use_ac'], enable_plasticity=cfg['use_pl'], solver_mode=cfg['solver_mode'])
    sim.use_mc = cfg['use_mc']
    
    rows, cols = 3, 5
    spacing_x, spacing_y = 4.0, 4.0
    offsets_mm = [-50, -25, 0, 25, 50]
    
    box_id = 0
    for r in range(rows):
        for c in range(cols):
            com = [0.0, 0.0, 0.0]
            com[r] = offsets_mm[c] / 1000.0
            axis_name = ["X", "Y", "Z"][r]
            label_str = f"Sweep {axis_name}-Axis: {offsets_mm[c]:+}mm Offset"
            
            inst = BoxDropInstance(
                uid=box_id,
                position_offset=[c * spacing_x, r * spacing_y, 0],
                drop_type="corner",
                box_params={},
                com_offset=com,
                label=label_str,
                enable_pad_contact=cfg['use_pc'],
                use_flex=cfg['use_fx'],
                flex_stiffness=cfg['flex_stiffness'],
                flex_damping=cfg['flex_damping']
            )
            sim.add_instance(inst)
            box_id += 1
            
    sim.init_simulation()
    sim.run_simulation_loop(mode=cfg['view_mode'])
    
    print("\n📊 Starting Headless Data Collection...")
    if sim.use_mc:
        sim.run_headless_parallel(duration=2.5, record_details=True)
    else:
        sim.init_simulation()
        sim.run_headless(duration=2.5, record_details=True)
    
    plot_testcase_B_summary(sim)
    return sim

def plot_testcase_B_summary(sim):
    """Testcase B 전용: CoM Sweep 비교 그래프"""
    print("\n📈 Generating Testcase B Summary Charts...")
    instances = sim.instances
    if not instances or not instances[0].history['time']:
        print("No data recorded.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axis_names = ["X-Axis CoM Sweep", "Y-Axis CoM Sweep", "Z-Axis CoM Sweep"]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6'] 
    offset_labels = ["-50mm", "-25mm", "0mm", "+25mm", "+50mm"]

    for r in range(3):
        ax = axes[r]
        ax.set_title(f"{axis_names[r]} (Drop Behavior)", fontsize=12, fontweight='bold')
        
        for c in range(5):
            idx = r * 5 + c
            if idx >= len(instances): break
            inst = instances[idx]
            time_arr = np.array(inst.history['time'])
            pos_z = np.array(inst.history['pos'])[:, 2] * 1000.0 # to mm
            
            ax.plot(time_arr, pos_z, color=colors[c], label=offset_labels[c], linewidth=1.5, alpha=0.9)
            
        ax.set_ylabel("Z-Height (mm)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc='upper right', title="Offset", fontsize='x-small')
        
    axes[2].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()

def plot_bending_doe_comparison(sim):
    """Testcase B 전용: 모든 DOE 케이스의 휨 변량(Bending Deflection) 비교 그래프"""
    print("\n📈 Generating DOE Bending Comparison Charts...")
    instances = sim.instances
    if not instances or not instances[0].history['time']:
        print("No data recorded.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    axis_names = ["X-Axis CoM Sweep", "Y-Axis CoM Sweep", "Z-Axis CoM Sweep"]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6'] 
    offset_labels = ["-50mm", "-25mm", "0mm", "+25mm", "+50mm"]

    for r in range(3):
        ax = axes[r]
        ax.set_title(f"{axis_names[r]} - Max Bending Deflection", fontsize=12, fontweight='bold')
        
        plotted_count = 0
        for c in range(5):
            idx = r * 5 + c
            if idx >= len(instances): break
            inst = instances[idx]
            
            time_arr = np.array(inst.history.get('time', []))
            if not time_arr.size: continue

            # Try to get deflection or bending data
            defl_mm = np.array(inst.history.get('deflection', []))
            if defl_mm.size > 0 and any(defl_mm > 0):
                ax.plot(time_arr, defl_mm, color=colors[c], label=offset_labels[c], linewidth=1.5, alpha=0.9)
                plotted_count += 1
            else:
                # Fallback: maybe just show bending angle if deflection lacks
                bend_deg = np.array(inst.history.get('bending', [])) * (180.0/np.pi)
                if bend_deg.size > 0:
                    ax.plot(time_arr, bend_deg, color=colors[c], label=f"{offset_labels[c]} (deg)", linewidth=1.0, linestyle=':')
                    plotted_count += 1
            
        ax.set_ylabel("Deflection (mm) / Angle (deg)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if plotted_count > 0:
            ax.legend(loc='upper right', title="Offset", fontsize='x-small', ncol=2)
        else:
            ax.text(0.5, 0.5, "No Bending Data Recorded\n(Is Flex Mode Enabled?)", 
                    transform=ax.transAxes, ha='center', va='center', color='gray')
        
    axes[2].set_xlabel("Time (seconds)")
    fig.suptitle(f"DOE Bending Comparison (K={instances[0].flex_stiffness}, D={instances[0].flex_damping})", fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def run_detailed_singleton_analysis(original_inst, solver_mode, use_ac, use_pl):
    """선택한 박스 하나만 별도로 시뮬레이션하여 상세 데이터를 수집"""
    print(f"\n🔬 Starting Detailed Singleton Analysis for: {original_inst.label}")
    
    # Create a fresh manager for a single box at origin
    sim = SimulationManager(enable_air_cushion=use_ac, enable_plasticity=use_pl, solver_mode=solver_mode)
    
    # Clone instance properties but reset position to origin
    inst = BoxDropInstance(
        uid=0,
        position_offset=[0, 0, 0],
        drop_type=original_inst.drop_type,
        box_params={'L': original_inst.L, 'W': original_inst.W, 'H': original_inst.H, 'MASS': original_inst.MASS},
        com_offset=original_inst.CoM_offset,
        label=original_inst.label,
        enable_pad_contact=original_inst.enable_pad_contact,
        use_flex=original_inst.use_flex,
        flex_stiffness=original_inst.flex_stiffness,
        flex_damping=original_inst.flex_damping
    )
    sim.add_instance(inst)
    sim.init_simulation()
    
    # Run with Record Details enabled
    sim.run_headless(duration=2.5, record_details=True)
    
    # Plot results
    plot_detailed_singleton_results(inst)

def plot_detailed_singleton_results(inst):
    """mujoco_boxdrop_analysis.py 스타일의 상세 그래프 생성 (Center + 8 Corners)"""
    print(f"\n📈 Generating Detailed Analysis Plots for {inst.label}...")
    
    h = inst.history
    times = np.array(h['time'])
    if not times.size: return
    
    # Figure 1: 위치, 속도, 가속도 (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'Box Drop Detailed Analysis: {inst.label}', fontsize=16, fontweight='bold')
    
    labels = ['X axis', 'Y axis', 'Z axis']
    row_titles = ['Position (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s²)']
    data_keys = ['pos', 'vel', 'acc']
    corner_keys = ['corner_pos', 'corner_vel', 'corner_acc']
    
    colors = plt.cm.tab10(np.linspace(0, 1, 9)) # Center + 8 Corners colors
    
    for row in range(3):
        center_data = np.array(h[data_keys[row]])
        for col in range(3):
            ax = axes[row, col]
            
            # 1. Plot Center Data (Thick line)
            ax.plot(times, center_data[:, col] * 1000, label='Center', color='black', linewidth=2.5, zorder=10)
            
            # 2. Plot 8 Corners
            corner_data_list = h[corner_keys[row]]
            for i in range(8):
                if i < len(corner_data_list) and len(corner_data_list[i]) > 0:
                    c_arr = np.array(corner_data_list[i])
                    if c_arr.ndim > 1:
                        ax.plot(times, c_arr[:, col] * 1000, label=f'Corner {i+1}', color=colors[i], linewidth=1.0, alpha=0.6)
            
            ax.set_ylabel(row_titles[row])
            ax.set_title(f"{row_titles[row]} - {labels[col]}", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            if row == 0 and col == 2:
                ax.legend(loc='upper right', fontsize=8, ncol=2)
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show() # Removed to show all at once
    
    # Figure 2: Impact Force & Cushion Force
    plt.figure(figsize=(10, 6))
    plt.plot(times, h['impact_force'], color='red', linewidth=1.8, label='Impact Force (Normal)')
    plt.plot(times, h['cushion_force'], color='blue', linewidth=1.5, linestyle='--', label='Air Cushion Force', alpha=0.7)
    
    max_f = np.max(h['impact_force'])
    max_t = times[np.argmax(h['impact_force'])]
    plt.scatter(max_t, max_f, color='black', zorder=10)
    plt.annotate(f'Peak: {max_f:.1f} N @ {max_t:.3f} s', xy=(max_t, max_f), xytext=(max_t + 0.2, max_f),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    plt.title(f"Forces Analysis: {inst.label}", fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Force (N)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.show() # Removed to show all at once

    # Figure 3: Bending / Flexing Analysis (Deflection)
    if h.get('deflection') and any(np.array(h['deflection']) > 1e-3):
        plt.figure(figsize=(10, 5))
        defl_mm = np.array(h['deflection'])
        bending_deg = np.array(h['bending']) * (180.0 / np.pi)
        
        plt.plot(times, defl_mm, color='purple', linewidth=2.0, label='Max Tip Deflection (mm)')
        
        peak_d = np.max(defl_mm)
        peak_a = bending_deg[np.argmax(defl_mm)]
        peak_t = times[np.argmax(defl_mm)]
        
        plt.axhline(peak_d, color='gray', linestyle=':', alpha=0.5)
        plt.annotate(f'Peak Flex: {peak_d:.2f}mm ({peak_a:.2f}°)\nat {peak_t:.3f}s', 
                     xy=(peak_t, peak_d), xytext=(peak_t + 0.1, peak_d * 0.8),
                     arrowprops=dict(facecolor='purple', shrink=0.05, width=1, headwidth=5),
                     fontsize=10, fontweight='bold', color='purple')
        
        plt.title(f"Bending Analysis (Deflection): {inst.label}", fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Max Vertical Deflection (mm)")
        
        # Add Parameter Info Box
        param_text = f"Flex Parameters:\nK = {inst.flex_stiffness:.0f}\nD = {inst.flex_damping:.0f}"
        plt.gca().text(0.02, 0.95, param_text, transform=plt.gca().transAxes, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        # plt.show() # Removed to show all at once

    # Figure 4: Unified Joint Rotation Analysis (Flexible Body Only)
    if inst.use_flex and h.get('joint_angles'):
        plt.figure(figsize=(12, 6))
        
        all_angles = []
        x_angles, y_angles = [], []
        
        for key, angles in h['joint_angles'].items():
            arr = np.array(angles)
            plt.plot(times, arr, alpha=0.15, linewidth=0.8, color='gray') # Individual trances
            all_angles.append(arr)
            if key.endswith('_x'): x_angles.append(arr)
            elif key.endswith('_y'): y_angles.append(arr)
            
        if all_angles:
            # Calculate Averages
            avg_all = np.mean(all_angles, axis=0)
            plt.plot(times, avg_all, color='black', linewidth=2.5, label='Overall Average (RMS-like)')
            
            if x_angles:
                plt.plot(times, np.mean(x_angles, axis=0), color='blue', linewidth=1.5, linestyle='--', label='X-Axis Avg (NS Bending)')
            if y_angles:
                plt.plot(times, np.mean(y_angles, axis=0), color='red', linewidth=1.5, linestyle='--', label='Y-Axis Avg (EW Bending)')
                
        plt.title(f"Unified Joint Rotation Analysis: {inst.label}", fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Angle (Degrees)")
        
        # Add Parameter Info Box
        param_text = f"Flex Parameters:\nK = {inst.flex_stiffness:.0f}\nD = {inst.flex_damping:.0f}"
        plt.gca().text(0.02, 0.95, param_text, transform=plt.gca().transAxes, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize='small')
        plt.tight_layout()

    # Figure 5: Pad Compression Analysis (Time-series)
    if h.get('corner_comp'):
        plt.figure(figsize=(12, 6))
        for c_name, comp_data in h['corner_comp'].items():
            plt.plot(times, comp_data, label=f'{c_name} Compression', linewidth=1.5)
            
        plt.title(f"Dynamic Pad Compression Analysis: {inst.label}", fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Average Compression Depth (mm)")
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    # --- FINAL SHOW ---
    print("   [PLOT] Displaying all analysis windows...")
    plt.show()

if __name__ == "__main__":
    print("--- WonhoLee Multi-Box Simulation System ---")
    print("Select Testcase:")
    print("1. One Box Drop (Detail Recording)")
    print("2. CoG_Random   (Grid 4x5)")
    print("3. CoG_Mov      (Parametric Sweep 3x5)")
    tc_choice = input("Enter selection (1/2/3, default 1): ").strip()
    
    if tc_choice == "2":
        manager = run_testcase_CoM_Random()
    elif tc_choice == "3":
        manager = run_testcase_CoM_Mov()
    else:
        manager = run_testcase_OneBox()
    
    # Physics Settings storage for singleton re-run
    use_ac = manager.enable_air_cushion
    use_pl = manager.enable_plasticity
    solver_mode = manager.solver_mode 
    use_pc = ENABLE_PAD_CONTACT # Simplified for loop

    while True:
        # 1. Case List Output
        print("\n" + "="*60)
        print("📋 ANALYSIS MENU: Select Case for Detailed Mapping")
        print("-" * 60)
        for inst in manager.instances:
            # Display descriptive name along with ID
            print(f"  [{inst.uid:2d}] {inst.label}")
        
        # Add special option for Case 3 (DOE Summary)
        is_sweep_case = (len(manager.instances) == 15 and tc_choice == "3")
        if is_sweep_case:
            print(f"  [99] View DOE Bending Comparison (All {len(manager.instances)} cases)")
            
        print("="*60)
        print("  [Enter] Quit Simulation")
            
        try:
            choice = input(f"\n📊 Enter ID to analyze (0-{len(manager.instances)-1}): ").strip()
            if choice == "": break
            
            if choice == "99" and is_sweep_case:
                plot_bending_doe_comparison(manager)
                continue
                
            idx = int(choice)
            
            if 0 <= idx < len(manager.instances):
                selected = manager.instances[idx]
                run_detailed_singleton_analysis(selected, solver_mode, use_ac, use_pl)
            else:
                print("⚠️ Out of range. Please select valid ID.")
                
        except ValueError:
            print("⚠️ Invalid input. Enter numeric ID.")
    
    print("Test Complete.")
