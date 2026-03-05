import gmsh
import math
import os
import sys
# Mujoco version 3.5.0
# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    def __init__(self):
        # 1. Box (Outer Dimensions)
        self.box_width = 1500.0
        self.box_height = 900.0
        self.box_depth = 200.0
        self.box_thick = 5.0
        self.box_center = [0.0, 0.0, 0.0]

        # 2. Display
        self.disp_width = self.box_width - 100.0
        self.disp_height = self.box_height - 100.0
        self.disp_depth = 5.0 
        
        # 3. Chassis
        self.chassis_depth = 40.0
        
        # 4. Adhesive
        self.coh_width = 20.0
        self.coh_thick = 1.0
        
        # 5. Orientation
        self.mode = "parcel" # "parcel" (-Z) or "ltl" (+Z)
        
        # 6. Mesh Settings
        self.mesh_size = 400.0 
        self.output_dir = "test_box_msh"

        self.cushion_cuts = [
            ([0.0, 0.0, 0.0], [600.0, 400.0, 400.0])
        ]
        
        # 7. Tolerance/Gap
        self.gap = 3.0

        # 8. Colors (RGBA)
        self.colors = {
            "Box": "0.6 0.5 0.4 0.5",      # Cardboard
            "Cushion": "0.9 0.9 0.9 1.0",  # EPS White
            "Disp": "0.1 0.1 0.1 1.0",     # Black Screen
            "Chassis": "0.7 0.7 0.7 1.0",  # Silver Metal
            "CohDisp": "0.8 0.8 0.2 0.5"   # Yellowish Tape
        }

# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================
def create_box(occ, center, dims):
    return occ.addBox(
        center[0] - dims[0]/2,
        center[1] - dims[1]/2,
        center[2] - dims[2]/2,
        dims[0], dims[1], dims[2]
    )

def get_stack_dims(cfg):
    """Returns z-offsets and dimensions for stack parts"""
    d_disp = cfg.disp_depth
    d_coh = cfg.coh_thick
    d_chas = cfg.chassis_depth
    total_thick = d_disp + d_coh + d_chas
    z_start = -total_thick / 2.0
    
    # Chassis
    z_chas = z_start + d_chas/2.0
    
    # Coh
    z_coh = z_start + d_chas + d_coh/2.0
    
    # Disp
    z_disp = z_start + d_chas + d_coh + d_disp/2.0
    
    return {
        "z_chas": z_chas, "d_chas": d_chas,
        "z_coh": z_coh, "d_coh": d_coh,
        "z_disp": z_disp, "d_disp": d_disp,
        "total_thick": total_thick,
        "z_start": z_start
    }

def apply_orientation(occ, tags, mode):
    if mode == "parcel":
        occ.rotate([(3, t) for t in tags], 0, 0, 0, 0, 1, 0, math.pi)

def get_cushion_defs(config):
    gap = config.gap
    margin = 2 * config.box_thick + 2 * gap
    cw = config.box_width - margin
    ch = config.box_height - margin
    cd = config.box_depth - margin
    
    sw = config.disp_width + 2 * gap
    sh = config.disp_height + 2 * gap
    
    z_cush_min = -cd / 2.0
    z_cush_max = cd / 2.0
    
    s = get_stack_dims(config)
    z_cav_min = s["z_start"] - gap
    z_cav_max = s["z_start"] + s["total_thick"] + gap
    
    if z_cav_min < z_cush_min: z_cav_min = z_cush_min
    if z_cav_max > z_cush_max: z_cav_max = z_cush_max
    
    defs = {}
    
    # A. Back Plate (0)
    thick_back = z_cav_min - z_cush_min
    if thick_back > 0:
        center_z = z_cush_min + thick_back/2.0
        defs[0] = ([0, 0, center_z], [cw, ch, thick_back])
    # B. Front Plate (1)
    thick_front = z_cush_max - z_cav_max
    if thick_front > 0:
        center_z = z_cav_max + thick_front/2.0
        defs[1] = ([0, 0, center_z], [cw, ch, thick_front])
    # C. Ring Parts
    ring_depth = z_cav_max - z_cav_min
    center_z = z_cav_min + ring_depth/2.0
    # Top (2), Bottom (3), Left (4), Right (5)
    h_top = (ch - sh)/2.0
    if h_top > 0: defs[2] = ([0, sh/2.0 + h_top/2.0, center_z], [cw, h_top, ring_depth])
    h_bot = (ch - sh)/2.0
    if h_bot > 0: defs[3] = ([0, -(sh/2.0 + h_bot/2.0), center_z], [cw, h_bot, ring_depth])
    w_left = (cw - sw)/2.0
    if w_left > 0: defs[4] = ([-(sw/2.0 + w_left/2.0), 0, center_z], [w_left, sh, ring_depth])
    w_right = (cw - sw)/2.0
    if w_right > 0: defs[5] = ([sw/2.0 + w_right/2.0, 0, center_z], [w_right, sh, ring_depth])
    return defs

# ==============================================================================
# PART GENERATION LOGIC
# ==============================================================================
def build_box_only(occ, cfg):
    # Hollow Box: Outer - Inner
    b_w, b_h, b_d = cfg.box_width, cfg.box_height, cfg.box_depth
    b_t = cfg.box_thick
    
    outer = create_box(occ, cfg.box_center, [b_w, b_h, b_d])
    inner = create_box(occ, cfg.box_center, [b_w - 2*b_t, b_h - 2*b_t, b_d - 2*b_t])
    
    res = occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
    return res[0][0][1]

def build_product_part(occ, cfg, part_name):
    s = get_stack_dims(cfg)
    target_tag = None
    
    if part_name == "Chassis":
        target_tag = create_box(occ, [0,0,s["z_chas"]], [cfg.disp_width, cfg.disp_height, s["d_chas"]])
        
    elif part_name == "Disp":
        target_tag = create_box(occ, [0,0,s["z_disp"]], [cfg.disp_width, cfg.disp_height, s["d_disp"]])
        
    elif part_name == "CohDisp":
        outer = create_box(occ, [0,0,s["z_coh"]], [cfg.disp_width, cfg.disp_height, s["d_coh"]])
        inner_w = cfg.disp_width - 2 * cfg.coh_width
        inner_h = cfg.disp_height - 2 * cfg.coh_width
        
        if inner_w > 0 and inner_h > 0:
            inner = create_box(occ, [0,0,s["z_coh"]], [inner_w, inner_h, s["d_coh"]])
            res = occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
            target_tag = res[0][0][1]
        else:
            target_tag = outer
            
    if target_tag:
        apply_orientation(occ, [target_tag], cfg.mode)
        
    return target_tag

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
def export_single_tag(vol_tag_or_list, part_name, config, suffix=""):
    """Helper to export a single volume tag to OBJ/STL/RAD"""
    # Clear any previous mesh to avoid index mixing
    gmsh.model.mesh.clear()
    
    if isinstance(vol_tag_or_list, list):
        vol_tags = vol_tag_or_list
    else:
        vol_tags = [vol_tag_or_list]

    # 2. Surface Mesh
    boundaries = gmsh.model.getBoundary([(3, t) for t in vol_tags], combined=True, oriented=False, recursive=False)
    surface_tags = [b[1] for b in boundaries]
    
    p_surf = gmsh.model.addPhysicalGroup(2, surface_tags)
    gmsh.model.setPhysicalName(2, p_surf, f"{part_name}{suffix}_surf")
    
    # Mesh Settings - Adaptive sizing is critical for thin parts (5mm) vs large box (1500mm)
    # If Min is too large, elements in thin sections will be flattened (ill-shaped), causing explosion.
    gmsh.option.setNumber("Mesh.MeshSizeMin", 5.0)   # Allow small elements near thin walls
    gmsh.option.setNumber("Mesh.MeshSizeMax", 200.0) # Allow large elements in bulk
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    
    # Use Frontal-Delaunay for 3D generation (Algorithm 6) - often better quality
    gmsh.option.setNumber("Mesh.Algorithm", 6) 
    gmsh.option.setNumber("Mesh.Optimize", 1) # Additional optimization
    
    gmsh.model.mesh.generate(2)
    
    # --- Check Normal Orientation (Signed Volume) ---
    try:
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_map = {tag: (node_coords[3*i], node_coords[3*i+1], node_coords[3*i+2]) 
                    for i, tag in enumerate(node_tags)}
        
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        calc_vol = 0.0
        
        for i, etype in enumerate(elem_types):
            nodes = elem_node_tags[i]
            if etype == 2: # Triangle
                for j in range(0, len(nodes), 3):
                    n1 = node_map[nodes[j]]; n2 = node_map[nodes[j+1]]; n3 = node_map[nodes[j+2]]
                    calc_vol += n1[0]*(n2[1]*n3[2]-n2[2]*n3[1]) + n1[1]*(n2[2]*n3[0]-n2[0]*n3[2]) + n1[2]*(n2[0]*n3[1]-n2[1]*n3[0])
            elif etype == 3: # Quad
                for j in range(0, len(nodes), 4):
                    n1 = node_map[nodes[j]]; n2 = node_map[nodes[j+1]]; n3 = node_map[nodes[j+2]]; n4 = node_map[nodes[j+3]]
                    calc_vol += n1[0]*(n2[1]*n3[2]-n2[2]*n3[1]) + n1[1]*(n2[2]*n3[0]-n2[0]*n3[2]) + n1[2]*(n2[0]*n3[1]-n2[1]*n3[0])
                    calc_vol += n1[0]*(n3[1]*n4[2]-n3[2]*n4[1]) + n1[1]*(n3[2]*n4[0]-n3[0]*n4[2]) + n1[2]*(n3[0]*n4[1]-n3[1]*n4[0])
        
        if calc_vol < 0:
            print(f"    > Detected INWARD normals for {part_name}{suffix}. Reversing...")
            gmsh.model.mesh.reverse([(2, t) for t in surface_tags])
            
    except Exception as e:
        print(f"    > Warning: Orientation check failed: {e}")
    # ------------------------------------------------
    
    # Cleanup
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    # Export Surface
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.Binary", 1)
    
    stl_name = f"{part_name}{suffix}.stl"
    
    print(f"    > Exporting {stl_name}...")
    gmsh.write(os.path.join(config.output_dir, stl_name))

    # 3. Volume Mesh
    gmsh.model.removePhysicalGroups([(2, p_surf)])
    p_vol = gmsh.model.addPhysicalGroup(3, vol_tags)
    gmsh.model.setPhysicalName(3, p_vol, f"{part_name}{suffix}_vol")
    
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(3)
    
    # --- OPTIMIZATION (Crucial for concave shapes) ---
    print(f"    > Optimizing 3D Mesh for {part_name}{suffix} (Netgen)...")
    gmsh.model.mesh.optimize("Netgen")
    # Multiple passes might be needed
    for _ in range(2):
        gmsh.model.mesh.optimize("Relocate3D")
        gmsh.model.mesh.optimize("Laplace2D")
    # -------------------------------------------------
    
    # Cleanup again
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    
    # Ensure only 3D elements (Tetra) are exported by removing 2D elements
    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.removeElements(dim, tag, [])

    # Verify 3D elements
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
    tetra_count = 0
    for etype, tags in zip(elem_types, elem_tags):
        if etype == 4:  # 4-node tetrahedron
            tetra_count += len(tags)
            print(f"    > Verified: {len(tags)} Tetrahedron elements generated.")
        else:
            print(f"    > Note: Generated {len(tags)} elements of type {etype} (Not Tetra).")
            
    if tetra_count == 0:
        print(f"    ! WARNING: No tetrahedron elements found for {part_name}{suffix}!")

    print(f"    > Exporting {part_name}{suffix}.msh (Format 2.2)...")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)

    # Strategy: Remove all existing physical groups to clear any 2D surface groups
    # Then add ONLY 3D volume elements to a new physical group.
    existing_groups = gmsh.model.getPhysicalGroups()
    if existing_groups:
        gmsh.model.removePhysicalGroups(existing_groups)
        
    p_vol = gmsh.model.addPhysicalGroup(3, vol_tags)
    gmsh.model.setPhysicalName(3, p_vol, f"{part_name}{suffix}_vol")

    gmsh.write(os.path.join(config.output_dir, f"{part_name}{suffix}.msh"))
    try:
        gmsh.write(os.path.join(config.output_dir, f"{part_name}{suffix}.rad"))
    except:
        pass
        
    # Remove physical groups for next iteration
    gmsh.model.removePhysicalGroups([(3, p_vol)])

def process_part(part_name, config):
    print(f"--- Processing {part_name} ---")
    
    # Helper to ensure clean session for each part
    def start_gmsh():
        if gmsh.isInitialized(): gmsh.finalize()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        return gmsh.model.occ
    
    if part_name == "Cushion":
        # DECOMPOSED CONVEX CUSHION GENERATION
        print("  > Generating Single Convex Cushion Part...")
        
        gap = config.gap
        margin = 2 * config.box_thick + 2 * gap
        cw = config.box_width - margin
        ch = config.box_height - margin
        cd = config.box_depth - margin
        
        sw = config.disp_width + 2 * gap
        sh = config.disp_height + 2 * gap
        
        s = get_stack_dims(config)
        z_cav_min = s["z_start"] - gap
        z_cav_max = s["z_start"] + s["total_thick"] + gap

        occ = start_gmsh()
        
        outer = create_box(occ, [0, 0, 0], [cw, ch, cd])
        
        cav_h = z_cav_max - z_cav_min
        cav_center_z = (z_cav_max + z_cav_min) / 2.0
        inner = create_box(occ, [0, 0, cav_center_z], [sw, sh, cav_h])
        
        res = occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
        cushion_tags = [t[1] for t in res[0]]
        
        if config.cushion_cuts:
            print(f"  > Applying {len(config.cushion_cuts)} additional cuts...")
            for center, dims in config.cushion_cuts:
                tool = create_box(occ, center, dims)
                res_cut = occ.cut([(3, t) for t in cushion_tags], [(3, tool)], removeObject=True, removeTool=True)
                cushion_tags = [t[1] for t in res_cut[0]]
        
        occ.synchronize()
        export_single_tag(cushion_tags, "Cushion", config)
        gmsh.finalize()

    else:
        occ = start_gmsh()
        # Standard Processing for Box and Products
        vol_tag = None
        if part_name == "Box":
            vol_tag = build_box_only(occ, config)
        else:
            vol_tag = build_product_part(occ, config, part_name)
            
        if vol_tag is None:
            print(f"Error: Failed to build {part_name}")
            gmsh.finalize()
            return

        occ.synchronize()
        
        print(f"  > Exporting CAD (STEP, IGES)...")
        gmsh.write(os.path.join(config.output_dir, f"{part_name}.step"))
        gmsh.write(os.path.join(config.output_dir, f"{part_name}.iges"))
        
        export_single_tag(vol_tag, part_name, config)
        gmsh.finalize()

def generate_mujoco_xml(parts, config):
    print("--- Generating MuJoCo XMLs ---")
    
    assets_str = ""
    bodies_rigid_str = ""
    bodies_flex_str = ""
    
    # 1. Assets
    # Product Parts
    for part in ["Disp", "CohDisp", "Chassis"]:
        assets_str += f'    <mesh name="{part}_mesh" file="{part}.stl" scale="0.001 0.001 0.001"/>\n'
    
    # Cushion Part (Single)
    assets_str += f'    <mesh name="Cushion_mesh" file="Cushion.stl" scale="0.001 0.001 0.001"/>\n'
        
    # 2. Bodies
    z_pos = 0.5
    
    # --- BOX (Primitive Geoms) ---
    W, H, D = config.box_width * 0.001, config.box_height * 0.001, config.box_depth * 0.001
    T = config.box_thick * 0.001
    col_box = config.colors["Box"]

    # Box Body String
    box_str = f'\n    <body name="Box" pos="0 0 {z_pos}">\n'
    box_str += f'      <freejoint/>\n'
    # Top/Bottom
    box_str += f'      <geom type="box" size="{W/2} {H/2} {T/2}" pos="0 0 {D/2 - T/2}" rgba="{col_box}" mass="1" group="1" contype="1" conaffinity="3"/>\n'
    box_str += f'      <geom type="box" size="{W/2} {H/2} {T/2}" pos="0 0 {-D/2 + T/2}" rgba="{col_box}" mass="1" group="1" contype="1" conaffinity="3"/>\n'
    # Front/Back
    box_str += f'      <geom type="box" size="{W/2} {T/2} {D/2 - T}" pos="0 {H/2 - T/2} 0" rgba="{col_box}" mass="1" group="1" contype="1" conaffinity="3"/>\n'
    box_str += f'      <geom type="box" size="{W/2} {T/2} {D/2 - T}" pos="0 {-H/2 + T/2} 0" rgba="{col_box}" mass="1" group="1" contype="1" conaffinity="3"/>\n'
    # Left/Right
    box_str += f'      <geom type="box" size="{T/2} {H/2 - T} {D/2 - T}" pos="{W/2 - T/2} 0 0" rgba="{col_box}" mass="1" group="1" contype="1" conaffinity="3"/>\n'
    box_str += f'      <geom type="box" size="{T/2} {H/2 - T} {D/2 - T}" pos="{-W/2 + T/2} 0 0" rgba="{col_box}" mass="1" group="1" contype="1" conaffinity="3"/>\n'
    box_str += f'    </body>\n'

    bodies_rigid_str += box_str
    bodies_flex_str += box_str

    # --- CUSHION (Compound Body) ---
    col_cush = config.colors["Cushion"]
    
    # Rigid Cushion
    bodies_rigid_str += f'\n    <body name="Cushion" pos="0 0 {z_pos}">\n'
    bodies_rigid_str += f'      <freejoint/>\n'
    bodies_rigid_str += f'      <geom type="mesh" mesh="Cushion_mesh" rgba="{col_cush}" mass="0.5" group="1" contype="1" conaffinity="3"/>\n'
    bodies_rigid_str += f'    </body>\n'

    # Flex Cushion (using flexcomp)
    margin = 2 * config.box_thick + 2 * config.gap
    cw = config.box_width - margin
    ch = config.box_height - margin
    cd = config.box_depth - margin
    ox = -cw/2.0 * 0.001
    oy = -ch/2.0 * 0.001
    oz = -cd/2.0 * 0.001
    
    bodies_flex_str += f'\n    <flexcomp name="Cushion" type="gmsh" file="Cushion.msh" scale="0.001 0.001 0.001" pos="0 0 {z_pos}" rgba="{col_cush}" mass="0.5" radius="0.001" dim="3" group="1" origin="{ox} {oy} {oz}">\n'
    bodies_flex_str += f'      <elasticity young="1e5" poisson="0.1" damping="0.01"/>\n'
    bodies_flex_str += f'      <contact selfcollide="none" internal="false" contype="1" conaffinity="3"/>\n'
    bodies_flex_str += f'    </flexcomp>\n'
    
    # --- PARTS ---
    for part in ["Disp", "CohDisp", "Chassis"]:
        col = config.colors.get(part, "0.5 0.5 0.5 1")
        part_str = f'\n    <body name="{part}" pos="0 0 {z_pos}">\n'
        part_str += f'      <freejoint/>\n'
        part_str += f'      <geom type="mesh" mesh="{part}_mesh" rgba="{col}" mass="1" group="2" contype="2" conaffinity="3"/>\n'
        part_str += f'    </body>\n'
        bodies_rigid_str += part_str
        bodies_flex_str += part_str

    template = """<mujoco model=\"TV_Packaging\">\n  <compiler angle=\"radian\" meshdir=\"./\"/>\n  <option timestep=\"0.002\" gravity=\"0 0 -9.81\"/>\n\n  <default>\n    <geom solref=\".5e-4 1.0 \" solimp=\"0.9 0.99 1e-4\"/>\n  </default>\n\n  <asset>\n    <texture name=\"grid\" type=\"2d\" builtin=\"checker\" rgb1=\".1 .2 .3\" rgb2=\".2 .3 .4\" width=\"300\" height=\"300\" mark=\"edge\" markrgb=\".8 .8 .8\"/>\n    <material name=\"grid\" texture=\"grid\" texrepeat=\"1 1\" texuniform=\"true\" reflectance=\".2\"/>\n{assets}\n  </asset>\n\n  <worldbody>\n    <light pos=\"0 0 4\" dir=\"0 0 -1\" diffuse=\"1.5 1.5 1.5\"/>\n    <geom name=\"floor\" type=\"plane\" size=\"0 0 0.1\" material=\"grid\" group=\"0\" contype=\"0\" conaffinity=\"1\"/>\n{bodies}\n  </worldbody>\n</mujoco>\n"""
    
    with open(os.path.join(config.output_dir, "model_B_rigid.xml"), "w") as f:
        f.write(template.format(assets=assets_str, bodies=bodies_rigid_str))
        
    with open(os.path.join(config.output_dir, "model_A_flex.xml"), "w") as f:
        f.write(template.format(assets=assets_str, bodies=bodies_flex_str))

def generate_test_flex_cushion_xml(config):
    print("--- Generating Test Flex Cushion XML ---")
    xml_content = """<mujoco model="Test_Flex_Cushion">
  <compiler angle="radian" meshdir="./"/>
  <!-- Smaller timestep for high-frequency modes in thin parts -->
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <map stiffness="100" shadowscale="0.5"/>
  </visual>

  <default>
    <!-- Soft contact to handle initial penetration gracefully -->
    <geom solref="0.05 1" solimp="0.95 0.99 0.001" friction="0.8 0.005 0.0001"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".8 .8 .8"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="1.5 1.5 1.5"/>
    <geom name="floor" type="plane" size="0 0 0.1" material="grid"/>
    
    <!-- Lifted to 0.5m -->
    <flexcomp name="Cushion_Flex" type="gmsh" file="Cushion.msh" scale="0.001 0.001 0.001" 
              pos="0 0 0.5" rgba="0.9 0.9 0.9 1" mass="0.5" radius="0.002" dim="3">
      <!-- User tuned parameters for stability and realism -->
      <elasticity young="7e5" poisson="0.3"/>
      <contact selfcollide="narrow" internal="false"/> 
    </flexcomp>
  </worldbody>
</mujoco>
"""
    file_path = os.path.join(config.output_dir, "test_flex_cushion.xml")
    with open(file_path, "w") as f:
        f.write(xml_content)
    print(f"  > Saved to {file_path}")

if __name__ == "__main__":
    cfg = Config()
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
        
    parts_list = ["Box", "Cushion", "Disp", "CohDisp", "Chassis"]
    
    for part in parts_list:
        process_part(part, cfg)
        
    generate_mujoco_xml(parts_list, cfg)
    generate_test_flex_cushion_xml(cfg)
    print("Done.")
