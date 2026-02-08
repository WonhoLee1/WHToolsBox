import gmsh
import math
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

class BoxMeshByGmsh:
    def __init__(self, width, depth, height, thickness=2.0, 
                 elem_size_x=50.0, elem_size_y=50.0, elem_size_z=50.0, elem_size_floor=200.0,
                 chassis_dims=None, cell_dims=None, hole_dims=None):
        """
        Initialize the Box Mesh Generator.
        
        Args:
            width (float): Box Width (mm)
            depth (float): Box Depth (mm)
            height (float): Box Height (mm)
            thickness (float): Shell thickness (mm) for Radioss property
            elem_size_x (float): Mesh element size for Box X (mm)
            elem_size_y (float): Mesh element size for Box Y (mm)
            elem_size_z (float): Mesh element size for Box Z (mm)
            elem_size_floor (float): Mesh element size for Floor (mm)
            chassis_dims (tuple): (w, h, d) for SET_CHASSIS. If None, defaults to 80% Box W/H, 50mm D.
            cell_dims (tuple): (w, h, d) for SET_CELL. If None, defaults to 80% Box W/H, 50mm D.
            hole_dims (tuple): (w, h) for CUSHION HOLE. If None, defaults to 60% Box W/H.
        """
        self.width = width
        self.depth = depth
        self.height = height
        self.thickness = thickness
        
        self.elem_size_x = elem_size_x
        self.elem_size_y = elem_size_y
        self.elem_size_z = elem_size_z
        self.elem_size_floor = elem_size_floor
        
        # Dimensions for Components
        # SET_CHASSIS
        if chassis_dims:
            self.chassis_w, self.chassis_h, self.chassis_d = chassis_dims
        else:
            self.chassis_w = width * 0.8
            self.chassis_h = height * 0.8
            self.chassis_d = 50.0
            
        # SET_CELL
        if cell_dims:
            self.cell_w, self.cell_h, self.cell_d = cell_dims
        else:
            self.cell_w = width * 0.8
            self.cell_h = height * 0.8
            self.cell_d = 50.0
            
        # CUSHION HOLE
        if hole_dims:
            self.hole_w, self.hole_h = hole_dims
        else:
            self.hole_w = width * 0.6
            self.hole_h = height * 0.6
            
        self.part_info = []

    def generate_mesh(self, pose, floor_z=0.0, move_mode='box', output_path='box_mesh.rad', view=True, box_name="Box"):
        """
        Generate mesh using Gmsh and export to Radioss.
        """
        gmsh.initialize()
        gmsh.model.add("BoxMesh")
        
        # 0. Pre-calculate Transforms
        t = pose[:3]
        r_vec = pose[3:]
        r_obj = R.from_rotvec(r_vec)
        
        dx = self.width / 2.0
        dy = self.depth / 2.0
        dz = self.height / 2.0
        
        # 1. Component Creation (Local Frame)
        
        # A. BOX_PAPER (Solid Volume)
        # We need "Solid CAD" but "Shell Mesh". 
        # Strategy: Create Volume, Mesh everything, Delete 3D elements of Paper Volume.
        paper_box = gmsh.model.occ.addBox(-dx, -dy, -dz, self.width, self.depth, self.height)
        
        # B. SET_CHASSIS (Solid)
        cw, ch, cd = self.chassis_w, self.chassis_h, self.chassis_d
        chassis_vol = gmsh.model.occ.addBox(-cw/2, -cd/2, -ch/2, cw, cd, ch)
        
        # C. SET_CELL (Solid)
        clw, clh, cld = self.cell_w, self.cell_h, self.cell_d
        cell_vol = gmsh.model.occ.addBox(-clw/2, -cld/2, -clh/2, clw, cld, clh)
        
        # D. BOX_CUSHION (Solid)
        off = self.thickness / 2.0
        cushion_w = self.width - 2*off
        cushion_d = self.depth - 2*off
        cushion_h = self.height - 2*off
        cx, cy, cz = cushion_w/2, cushion_d/2, cushion_h/2
        
        cushion_base = gmsh.model.occ.addBox(-cx, -cy, -cz, cushion_w, cushion_d, cushion_h)
        
        hw = self.hole_w
        hh = self.hole_h
        hd = self.depth * 2.0
        hole_vol = gmsh.model.occ.addBox(-hw/2, -hd/2, -hh/2, hw, hd, hh)
        
        gmsh.model.occ.synchronize()
        
        # Boolean Subtraction for Cushion
        chassis_copy = gmsh.model.occ.copy([(3, chassis_vol)])
        cell_copy = gmsh.model.occ.copy([(3, cell_vol)])
        tools = chassis_copy + cell_copy + [(3, hole_vol)]
        
        out_cushion, _ = gmsh.model.occ.cut([(3, cushion_base)], tools, removeObject=True, removeTool=True)
        cushion_tags = [tag for dim, tag in out_cushion]
        
        gmsh.model.occ.synchronize()
        
        # 2. Floor Creation
        f_size = max(self.width, self.depth, self.height) * 3.0
        floor_tag = gmsh.model.occ.addRectangle(-f_size/2, -f_size/2, floor_z, f_size, f_size)
        
        gmsh.model.occ.synchronize()
        
        # Get Floor Lines for later specific sizing
        floor_lines_bnd = gmsh.model.getBoundary([(2, floor_tag)], combined=True, oriented=False, recursive=False)
        floor_lines_tags = set([tag for dim, tag in floor_lines_bnd])

        # 3. Apply Transforms
        move_objs = [(3, paper_box), (3, chassis_vol), (3, cell_vol)] + [(3, t) for t in cushion_tags]
        
        rot_angle = np.linalg.norm(r_vec)
        if rot_angle > 1e-6:
            rot_axis = r_vec / rot_angle
        else:
            rot_axis = [0, 0, 1]
            rot_angle = 0
            
        if move_mode == 'box':
            if rot_angle > 1e-6:
                gmsh.model.occ.rotate(move_objs, 0, 0, 0, rot_axis[0], rot_axis[1], rot_axis[2], rot_angle)
            gmsh.model.occ.translate(move_objs, t[0], t[1], t[2])
            
        elif move_mode == 'floor':
             r_vec_inv = -r_vec
             rot_angle_inv = np.linalg.norm(r_vec_inv)
             rot_axis_inv = r_vec_inv / rot_angle_inv if rot_angle_inv > 1e-6 else [0,0,1]
             t_inv = -r_obj.inv().apply(t)
             
             if rot_angle_inv > 1e-6:
                gmsh.model.occ.rotate([(2, floor_tag)], 0, 0, 0, rot_axis_inv[0], rot_axis_inv[1], rot_axis_inv[2], rot_angle_inv)
             gmsh.model.occ.translate([(2, floor_tag)], t_inv[0], t_inv[1], t_inv[2])
             
        gmsh.model.occ.synchronize()
        
        # 4. Physical Groups Setup
        self.part_info = []
        
        if move_mode == 'box':
            rot_mat = r_obj.as_matrix()
            center_pos = t
        else:
            rot_mat = np.eye(3)
            center_pos = np.zeros(3)
            
        x_axis = rot_mat[:, 0]
        y_axis = rot_mat[:, 1]
        z_axis = rot_mat[:, 2]
        
        # --- BOX_PAPER (Faces) ---
        paper_bnd = gmsh.model.getBoundary([(3, paper_box)], combined=True, oriented=False, recursive=False)
        paper_s_tags = [tag for dim, tag in paper_bnd]
        
        for st in paper_s_tags:
            bb = gmsh.model.getBoundingBox(2, st)
            c = np.array([(bb[0]+bb[3])/2, (bb[1]+bb[4])/2, (bb[2]+bb[5])/2])
            v = c - center_pos
            dots = [np.dot(v, x_axis), np.dot(v, y_axis), np.dot(v, z_axis)]
            a_idx = np.argmax([abs(d) for d in dots])
            val = dots[a_idx]
            
            f_name = "Unknown"
            if a_idx == 2: f_name = "Top" if val > 0 else "Bottom"
            elif a_idx == 1: f_name = "Back" if val > 0 else "Front" # Y-aligned
            elif a_idx == 0: f_name = "Right" if val > 0 else "Left"
            
            # User requested legacy naming "Box" instead of "BOX_PAPER"
            pname = f"{box_name}_{f_name}"
            try:
                pid = gmsh.model.addPhysicalGroup(2, [st], name=pname)
                self.part_info.append({'id': pid, 'name': pname, 'type': 'shell'})
            except: pass
            
        # --- BOX_CUSHION ---
        try:
            pid = gmsh.model.addPhysicalGroup(3, cushion_tags, tag=53000, name="BOX_CUSHION")
            self.part_info.append({'id': pid, 'name': "BOX_CUSHION", 'type': 'solid'})
        except: pass
        
        # --- SET_CHASSIS & CELL ---
        for vol_tag, base_name, base_id in [(chassis_vol, "SET_CHASSIS", 51000), 
                                            (cell_vol, "SET_CELL", 52000)]:
            try:
                pid = gmsh.model.addPhysicalGroup(3, [vol_tag], tag=base_id, name=base_name)
                self.part_info.append({'id': pid, 'name': base_name, 'type': 'solid'})
            except: pass
            
            # Node Sets (Standard)
            # User Request: "2D elements are unnecessary" for solid parts.
            # So we do NOT create Physical Groups for these surfaces.
            # If Node Sets are needed later, we can add them as Physical "Point" groups or rely on geometry.
            
            # bb = gmsh.model.getBoundingBox(3, vol_tag)
            # vol_ctr = np.array([(bb[0]+bb[3])/2, (bb[1]+bb[4])/2, (bb[2]+bb[5])/2])
            # surfs = gmsh.model.getBoundary([(3, vol_tag)], combined=True, oriented=False, recursive=False)
            
            # f_map = {"F": [], "B": [], "T": [], "Bot": [], "L": [], "R": []}
            # for d, st in surfs:
            #     # ... (Logic omitted to prevent 2D element generation) ...
            #     pass 
                    
            # for k, tags in f_map.items():
            #     if tags:
            #         grp_name = f"{base_name}_{k}"
            #         try:
            #             gmsh.model.addPhysicalGroup(2, tags, name=grp_name)
            #         except: pass
                    
        # Floor
        try:
            pid = gmsh.model.addPhysicalGroup(2, [floor_tag], tag=50100, name="Floor")
            self.part_info.append({'id': pid, 'name': "Floor", 'type': 'shell'})
        except: pass
        
        # 5. Meshing Setup
        
        # A. Directional Scaling (Lines)
        # Apply specific elem types to all lines based on orientation
        # X -> elem_size_x (W)
        # Y -> elem_size_y (H) (User said Y-axis is H-elem)
        # Z -> elem_size_z (D) (User said Z-axis is D-elem)
        
        all_lines = gmsh.model.getEntities(1)
        for dim, tag in all_lines:
            size = self.elem_size_x # Default
            
            if tag in floor_lines_tags:
                size = self.elem_size_floor
            else:
                # Check orientation
                bbox = gmsh.model.getBoundingBox(1, tag)
                # Simple check: (max - min)
                lx = abs(bbox[3] - bbox[0])
                ly = abs(bbox[4] - bbox[1])
                lz = abs(bbox[5] - bbox[2])
                
                # Since rotated, we need to check against axes
                # Or assume local frame? Lines are in global frame now.
                # If box rotated, X-axis is `x_axis`, etc.
                # We need vector of the line.
                # Since Bbox is AABB, it doesn't give direction for diagonals. 
                # But Box edges are straight.
                # Let's get start/end points.
                pts = gmsh.model.getAdjacencies(1, tag)[1] # returns nodes? No, getAdjacencies gives lower/upper entities.
                # Curve endpoints:
                # getFoundary gives 0-dim entities
                bnd_pts = gmsh.model.getBoundary([(1, tag)], combined=False, oriented=False, recursive=False)
                if len(bnd_pts) >= 2:
                    p1_tag = bnd_pts[0][1]
                    p2_tag = bnd_pts[1][1]
                    c1 = gmsh.model.getValue(0, p1_tag, [])
                    c2 = gmsh.model.getValue(0, p2_tag, [])
                    vec = np.array(c2) - np.array(c1)
                    length = np.linalg.norm(vec)
                    if length < 1e-6: continue
                    
                    # Dot with axes
                    dx = abs(np.dot(vec, x_axis))
                    dy = abs(np.dot(vec, y_axis))
                    dz = abs(np.dot(vec, z_axis))
                    
                    # Pick dominant axis
                    if dx >= dy and dx >= dz: size = self.elem_size_x
                    elif dy >= dx and dy >= dz: size = self.elem_size_y # H-elem
                    elif dz >= dx and dz >= dy: size = self.elem_size_z # D-elem
            
            # Compute nodes
            bnd_pts = gmsh.model.getBoundary([(1, tag)], combined=False, oriented=False, recursive=False)
            if len(bnd_pts) >= 2:
                p1_tag = bnd_pts[0][1]
                p2_tag = bnd_pts[1][1]
                c1 = gmsh.model.getValue(0, p1_tag, [])
                c2 = gmsh.model.getValue(0, p2_tag, [])
                length = np.linalg.norm(np.array(c2) - np.array(c1))
                
                n = max(1, int(round(length / size)))
                gmsh.model.mesh.setTransfiniteCurve(tag, n+1)

        # B. Surfaces Transfinite (Auto-detect logic for Box-like shapes)
        # We need Transfinite Surfaces for Transfinite Volumes (Chassis, Cell, Paper)
        transfinite_vols = [paper_box, chassis_vol, cell_vol]
        transfinite_surfs = set()
        for v in transfinite_vols:
            surfs = gmsh.model.getBoundary([(3, v)], combined=True, oriented=False, recursive=False)
            for s in surfs:
                transfinite_surfs.add(s[1])
                
        # Also include Floor
        transfinite_surfs.add(floor_tag)
        
        for t in transfinite_surfs:
            try:
                gmsh.model.mesh.setTransfiniteSurface(t)
                gmsh.model.mesh.setRecombine(2, t)
            except: pass
            
        # C. Solids Transfinite (Volumes)
        # Force structured hex mesh for Box-like components
        # We INCLUDE paper_box here so it uses Transfinite algo instead of global HXT.
        # This prevents HXT from seeing Quad boundaries on Paper.
        for v in transfinite_vols:
            try:
                gmsh.model.mesh.setTransfiniteVolume(v)
            except: pass
            
        # Cushion: Use HXT Algorithm as requested
        # Note: Standard Gmsh ID for HXT is 10.
        gmsh.option.setNumber("Mesh.Algorithm3D", 10) # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
        
        # Ensure Full Hex via Subdivision (Splits Tets -> 4 Hexes)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
        
        # D. Paper Solid Handling
        # Do NOT delete the Volume entity (User Request: "SOLID CAD는 생성하고")
        # But we don't want 3D mesh for it.
        # If we just mesh generate(3), it meshes all volumes.
        # We will Delete the Elements of the Paper Volume after generation.
        
        gmsh.model.mesh.generate(3)
        
        # Deduplicate (Global check)
        # This merges coincident nodes/elements if any exist improperly
        gmsh.model.mesh.removeDuplicateNodes()
        # gmsh.model.mesh.removeDuplicateElements() # Takes no args? No, it's complex. Nodes is safer first.
        
        # Post-Generation Cleanup for Paper
        # Get all elements in Paper Volume
        try:
             el_types, el_tags, _ = gmsh.model.mesh.getElements(3, paper_box)
             # el_tags is list of list (per type)
             to_delete = []
             for tags_ in el_tags:
                 to_delete.extend(tags_)
             
             if to_delete:
                 gmsh.model.mesh.deleteElements(to_delete) 
        except:
             pass 

        # 6. Write
        if not output_path.endswith('.rad'): output_path += '.rad'
        gmsh.write(output_path)
        
        # 7. Post-Process
        self.write_radioss_decks(output_path, floor_z=floor_z, view=view)
        
        if view:
            gmsh.fltk.run()
        gmsh.finalize()

    def write_radioss_decks(self, mesh_path, floor_z=0.0, view=False):
        """ Generate Radioss Decks """
        base_dir = os.path.dirname(mesh_path)
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        starter_path = os.path.join(base_dir, f"{base_name}_0000.rad")
        engine_path = os.path.join(base_dir, f"{base_name}_0001.rad")
        mesh_rel = os.path.basename(mesh_path)
        
        with open(starter_path, 'w') as f:
            f.write("#RADIOSS STARTER\n")
            f.write(f"/BEGIN\n{base_name}\n")
            f.write(f"      2025         0\n") 
            f.write(f"                  Mg                  mm                   s\n")
            f.write(f"                  Mg                  mm                   s\n")
            f.write(f"#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|\n")
            f.write(f"#include {mesh_rel}\n")
            
            # MAT
            f.write("/MAT/ELAST/1\nPaper_Mat\n        1E-9\n      5000.0                 0.4\n")
            f.write("/MAT/ELAST/2\nCushion_Mat\n        5E-10\n       500.0                 0.3\n")
            
            # PROP
            f.write(f"/PROP/SHELL/1\nPaper_Prop\n        24\n\n         5                    {self.thickness:10.5f}                                      -1        -1\n")
            f.write(f"/PROP/SHELL/2\nFloor_Prop\n        24\n\n         5                           1.0                                      -1        -1\n")
            f.write(f"/PROP/SOLID/3\nSolid_Prop\n        14\n")
            
            # PARTS
            f.write(f"#---1----|----2----|----3----|----4----|----5----|----6----|----7----|----8----|----9----|---10----|\n")
            
            # Sort parts by ID
            sorted_parts = sorted(self.part_info, key=lambda x: x['id'])
            
            for p in sorted_parts:
                pid = p['id']
                name = p['name']
                
                prop_id, mat_id = 1, 1
                if name == "Floor": prop_id, mat_id = 2, 1
                elif "BOX_CUSHION" in name: prop_id, mat_id = 3, 2
                elif "SET_" in name: prop_id, mat_id = 3, 1
                elif "Box" in name or "BOX" in name: prop_id, mat_id = 1, 1 # Catch-all for Box/BOX_PAPER
                
                f.write(f"/PART/{pid}\n{name}\n{prop_id:10d}{mat_id:10d}\n")
            
            # Rigid Floor
            f.write(f"/NODE\n     99999                 0.0                 0.0             {floor_z:20.5f}\n")
            f.write(f"/RBODY/1\nFloor_Rigid\n     99999         0                                             50100\n")
            f.write(f"\n\n\n")
            f.write(f"/BCS/1\nFloor_Fix\n   111 111         0     99999\n")
            
            f.write("/END\n")
            
        with open(engine_path, 'w') as f:
            f.write(f"/RUN/{base_name}/1\n1.0\n/VERS/2026\n/ANIM/DT\n0.0 0.1\n/ANIM/VECT/DISP\n/ANIM/ELEM/VONM\n/END\n")
            
        print(f"Generated {starter_path}")

if __name__ == "__main__":
    gen = BoxMeshByGmsh(1400, 200, 800, chassis_dims=(500,400,50), cell_dims=(300,200,50))
    gen.generate_mesh([0,0,500, 0,0,0], view=True)
