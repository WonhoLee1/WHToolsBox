import os
path = 'c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_discrete_builder/__init__.py'

# The file content structure after my repeated corruptions is messy.
# I will define the WHOLE content of the file here and write it.
# Since the file is 1500 lines, I will do it in chunks.

template_head = """import os
import json
import numpy as np
import io
import math

# =====================================================================
# [1] 시스템 설정 및 전역 유틸리티
# =====================================================================
def get_local_pose(vec, drop_height, rot_axis, angle_rad, corner_dist):
    v = np.array(vec)
    v_rot = v * np.cos(angle_rad) + np.cross(rot_axis, v) * np.sin(angle_rad) + rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_rad))
    return v_rot + np.array([0, 0, drop_height + corner_dist])

def calculate_solref(K, C):
    if K <= 0: raise ValueError("K > 0")
    timeconst = 1.0 / math.sqrt(K)
    if timeconst < 0.002:
        timeconst = 0.002
        K_safe = 1.0 / (timeconst**2)
    else: K_safe = K
    dampratio = C / (2.0 * math.sqrt(K_safe))
    return round(timeconst, 5), round(dampratio, 5)

def get_default_config(user_config=None):
    if user_config is None: user_config = {}
    config = {
        "drop_mode": user_config.get("drop_mode", "PARCEL"),
        "drop_direction": user_config.get("drop_direction", "front"),
        "sim_integrator": user_config.get("sim_integrator", "implicitfast"),
        "sim_timestep": user_config.get("sim_timestep", 0.001),
        "sim_iterations": user_config.get("sim_iterations", 80),
        "sim_noslip_iterations": user_config.get("sim_noslip_iterations", 5),
        "sim_tolerance": user_config.get("sim_tolerance", 1e-6),
        "sim_impratio": user_config.get("sim_impratio", 1.0),
        "sim_gravity": user_config.get("sim_gravity", [0, 0, -9.81]),
        "sim_nthread": user_config.get("sim_nthread", 4),
        "box_w": user_config.get("box_w", 2.0),
        "box_h": user_config.get("box_h", 1.4),
        "box_d": user_config.get("box_d", 0.25),
        "box_thick": user_config.get("box_thick", 0.01),
        "box_div": user_config.get("box_div", [5, 4, 3]),
        "cush_div": user_config.get("cush_div", [5, 4, 3]),
        "oc_div": user_config.get("oc_div", [5, 4, 1]),
        "occ_div": user_config.get("occ_div", [5, 4, 1]),
        "chassis_div": user_config.get("chassis_div", [5, 4, 1]),
        "box_use_weld": user_config.get("box_use_weld", True),
        "cush_use_weld": user_config.get("cush_use_weld", True),
        "oc_use_weld": user_config.get("oc_use_weld", True),
        "occ_use_weld": user_config.get("occ_use_weld", True),
        "chassis_use_weld": user_config.get("chassis_use_weld", True),
        "mass_paper": 4.0,
        "mass_cushion": user_config.get("mass_cushion", 1.0),
        "mass_oc": 5.0,
        "mass_occ": 0.1,
        "mass_chassis": 10.0,
        "cush_gap": 0.001,
        "occ_ithick": 0.050,
        "air_density": 1.225,
        "air_viscosity": 1.81e-5,
        "air_cd_drag": 1.05,
        "enable_air_drag": True,
        "enable_air_squeeze": True,
        "chassis_aux_masses": user_config.get("chassis_aux_masses", []),
        "ground_friction": user_config.get("ground_friction", 0.2),
    }
    # Material props (Simplified for restore, user can adjust later)
    config["mat_paper"] = {"rgba": "0.7 0.6 0.4 0.9", "solref": "0.01 1.0", "solimp": "0.1 0.95 0.005", "contype": "1", "conaffinity": "1"}
    config["mat_cush"] = {"rgba": "0.9 0.9 0.9 0.5", "solref": "0.03 0.8", "solimp": "0.1 0.95 0.005", "contype": "2", "conaffinity": "1"}
    config["mat_cell"] = {"rgba": "0.1 0.1 0.1 1.0", "solref": "0.01 0.3", "solimp": "0.5 0.95 0.001", "contype": "4", "conaffinity": "3"}
    config["mat_tape"] = {"rgba": "1.0 0.1 0.1 0.8", "solref": "0.01 1.0", "solimp": "0.1 0.99 0.001", "contype": "8", "conaffinity": "3"}
    config["mat_tv"] = {"rgba": "0.1 0.5 0.8 1.0", "solref": "0.002 0.5", "solimp": "0.1 0.95 0.005", "contype": "16", "conaffinity": "3"}
    config["ground_solref"] = "0.001 0.0001"
    config["ground_solimp"] = "0.9 0.99 0.01"
    
    for k, v in user_config.items(): config[k] = v
    return config

class DiscreteBlock:
    def __init__(self, idx, cx, cy, cz, dx, dy, dz, mass, material):
        self.idx, self.cx, self.cy, self.cz, self.dx, self.dy, self.dz, self.mass, self.material = idx, cx, cy, cz, dx, dy, dz, mass, material
        self.volume = (2*dx) * (2*dy) * (2*dz)

class BaseDiscreteBody:
    def __init__(self, name, width, height, depth, mass, div, material_props, use_internal_weld=True):
        self.name, self.width, self.height, self.depth, self.total_mass, self.div, self.material_props, self.use_internal_weld = name, width, height, depth, mass, div, material_props, use_internal_weld
        self.blocks, self.children, self.parent = {}, [], None
    def add_child(self, child): child.parent = self; self.children.append(child)
    def _generate_strict_grid_axis(self, length, num_div, cuts=[]):
        edges = sorted(list(set([-length/2, length/2] + [c for c in cuts if -length/2 <= c <= length/2])))
        target = length/num_div; nodes = []
        for i in range(len(edges)-1):
            s, e = round(edges[i], 5), round(edges[i+1], 5)
            if not nodes or abs(nodes[-1]-s)>1e-6: nodes.append(s)
            if (e-s) > target*1.01:
                sub = max(1, int(round((e-s)/target))); sn = np.linspace(s, e, sub+1)
                for n in sn[1:]: nodes.append(round(n, 5))
            else: nodes.append(e)
        return sorted(list(set(nodes)))
    def is_cavity(self, cx, cy, cz, dx, dy, dz): return False
    def is_edge_block(self, i, j, k): return False
    def build_geometry(self, offset=[0,0,0], rx=[], ry=[], rz=[]):
        nx, ny, nz = self._generate_strict_grid_axis(self.width, self.div[0], rx), self._generate_strict_grid_axis(self.height, self.div[1], ry), self._generate_strict_grid_axis(self.depth, self.div[2], rz)
        tmp, tv = [], 0.0
        for i in range(len(nx)-1):
            for j in range(len(ny)-1):
                for k in range(len(nz)-1):
                    cx, cy, cz = (nx[i]+nx[i+1])/2, (ny[j]+ny[j+1])/2, (nz[k]+nz[k+1])/2
                    dx, dy, dz = (nx[i+1]-nx[i])/2, (ny[j+1]-ny[j])/2, (nz[k+1]-nz[k])/2
                    if self.is_cavity(cx, cy, cz, dx, dy, dz): continue
                    blk = DiscreteBlock((i,j,k), cx+offset[0], cy+offset[1], cz+offset[2], dx, dy, dz, 0, self.material_props)
                    tmp.append(blk); tv += blk.volume
        for b in tmp: b.mass = self.total_mass * (b.volume/tv); self.blocks[b.idx] = b
    def get_weld_xml_strings(self):
        if not self.use_internal_weld: return [x for c in self.children for x in c.get_weld_xml_strings()]
        xml, keys, cls = [], set(self.blocks.keys()), f"weld_{self.__class__.__name__.lower()}"
        for (i,j,k), b1 in self.blocks.items():
            for ni, nj, nk, tag in [(i+1,j,k,'PX'), (i,j+1,k,'PY'), (i,j,k+1,'PZ')]:
                if (ni,nj,nk) in keys:
                    b2 = self.blocks[(ni,nj,nk)]
                    s1, s2 = f"s_{self.name}_{i}_{j}_{k}_{tag}", f"s_{self.name}_{ni}_{nj}_{nk}_N{tag[1]}"
                    xml.append(f'        <weld class="{cls}" site1="{s1}" site2="{s2}"/>')
        return xml + [x for c in self.children for x in c.get_weld_xml_strings()]
    def calculate_inertia(self):
        all_b = []
        def col(body):
            nonlocal all_b
            for b in body.blocks.values(): all_b.append(b)
            for c in body.children: col(c)
        col(self); tm = sum(b.mass for b in all_b)
        if tm <= 0: return 0, np.zeros(3), np.zeros(3), []
        cg = sum(b.mass * np.array([b.cx, b.cy, b.cz]) for b in all_b) / tm
        moi = np.zeros(3)
        for b in all_b:
            w, h, d = 2*b.dx, 2*b.dy, 2*b.dz
            moi += np.array([(1/12)*b.mass*(h**2+d**2), (1/12)*b.mass*(w**2+d**2), (1/12)*b.mass*(w**2+h**2)])
            dp = (np.array([b.cx, b.cy, b.cz]) - cg)**2
            moi += b.mass * np.array([dp[1]+dp[2], dp[0]+dp[2], dp[0]+dp[1]])
        return tm, cg, moi, []
    def get_worldbody_xml_strings(self, indent=2):
        if self.parent is None: return [x for c in self.children for x in c.get_worldbody_xml_strings(indent)]
        xml, ind = [], "  "*indent; xml.append(f'{ind}<body name="{self.name}">')
        if not self.use_internal_weld and self.name not in ["PackagingBox", "AssySet"]:
            xml += [f'{ind}  <joint type="slide" axis="{a}"/>' for a in ["1 0 0","0 1 0","0 0 1"]] + [f'{ind}  <joint type="ball"/>']
            for (i,j,k), b in self.blocks.items():
                gc = f"contact_{self.__class__.__name__.lower()}"; xml.append(f'{ind}  <geom name="g_{self.name.lower()}_{i}_{j}_{k}" type="box" pos="{b.cx:.5f} {b.cy:.5f} {b.cz:.5f}" size="{b.dx:.5f} {b.dy:.5f} {b.dz:.5f}" mass="{b.mass:.6f}" class="{gc}"/>')
        else:
            for (i,j,k), b in self.blocks.items():
                xml.append(f'{ind}  <body name="b_{self.name.lower()}_{i}_{j}_{k}" pos="{b.cx:.5f} {b.cy:.5f} {b.cz:.5f}">')
                xml += [f'{ind}    <joint type="slide" axis="{a}"/>' for a in ["1 0 0","0 1 0","0 0 1"]] + [f'{ind}    <joint type="ball"/>']
                gc = f"contact_{self.__class__.__name__.lower()}"
                xml.append(f'{ind}    <geom name="g_{self.name.lower()}_{i}_{j}_{k}" type="box" size="{b.dx:.5f} {b.dy:.5f} {b.dz:.5f}" mass="{b.mass:.6f}" class="{gc}"/>')
                for tag, p in [('PX',f'{b.dx} 0 0'),('NX',f'{-b.dx} 0 0'),('PY',f'0 {b.dy} 0'),('NY',f'0 {-b.dy} 0'),('PZ',f'0 0 {b.dz}'),('NZ',f'0 0 {-b.dz}')]:
                    xml.append(f'{ind}    <site name="s_{self.name}_{i}_{j}_{k}_{tag}" pos="{p}"/>')
                xml.append(f'{ind}  </body>')
        for c in self.children: xml.extend(c.get_worldbody_xml_strings(indent+1))
        xml.append(f'{ind}</body>'); return xml

class BPaperBox(BaseDiscreteBody):
    def __init__(self, n, w, h, d, m, dv, t, mp, uw=True): super().__init__(n,w,h,d,m,dv,mp,uw); self.thick = t
    def is_cavity(self, cx, cy, cz, dx, dy, dz): return abs(cx)<(self.width/2-self.thick-1e-4) and abs(cy)<(self.height/2-self.thick-1e-4) and abs(cz)<(self.depth/2-self.thick-1e-4)

class BCushion(BaseDiscreteBody):
    def __init__(self, n, w, h, d, m, dv, mp, ab, g, cc, uw=True): super().__init__(n,w,h,d,m,dv,mp,uw); self.assy_bbox, self.gap, self.cushion_cutter = ab, g, cc
    def is_cavity(self, cx, cy, cz, dx, dy, dz):
        ax_min, ax_max, ay_min, ay_max, az_min, az_max = self.assy_bbox
        if (ax_min-self.gap<=cx<=ax_max+self.gap and ay_min-self.gap<=cy<=ay_max+self.gap and az_min-self.gap<=cz<=az_max+self.gap): return True
        for v in self.cushion_cutter.values():
            if (v[0]-v[3]/2<=cx<=v[0]+v[3]/2 and v[1]-v[4]/2<=cy<=v[1]+v[4]/2 and v[2]-v[5]/2<=cz<=v[2]+v[5]/2): return True
        return False
    def is_edge_block(self, i, j, k): nx, ny, nz = self.div; bx, by, bz = (i==0 or i==nx-1), (j==0 or j==ny-1), (k==0 or k==nz-1); return (bx and by) or (by and bz) or (bz and bx)

class BOpenCellCohesive(BaseDiscreteBody):
    def __init__(self, n, w, h, d, m, dv, it, mp, uw=True): super().__init__(n,w,h,d,m,dv,mp,uw); self.ithick = it
    def is_cavity(self, cx, cy, cz, dx, dy, dz): return abs(cx)<(self.width/2-self.ithick-1e-4) and abs(cy)<(self.height/2-self.ithick-1e-4)

class BOpenCell(BaseDiscreteBody): pass
class BChassis(BaseDiscreteBody): pass
class BAuxBoxMass(BaseDiscreteBody):
    def __init__(self, n, w, h, d, m, mp=None):
        if mp is None: mp = {'rgba': '1 0 0 0.4', 'solref': '0.02 1', 'solimp': '0.1 0.95 0.005'}
        super().__init__(n,w,h,d,m,[1,1,1],mp); self.material_props['contype'], self.material_props['conaffinity'] = '0', '0'
    def build_geometry(self, lo=[0,0,0]): self.blocks[(0,0,0)] = DiscreteBlock((0,0,0), lo[0], lo[1], lo[2], self.width/2, self.height/2, self.depth/2, self.total_mass, self.material_props)

class BUnitBlock(BaseDiscreteBody): pass

def parse_drop_target(mode_str, direction_str, box_w, box_h, box_d):
    import re
    mode = str(mode_str).upper()
    direct = str(direction_str).lower().replace('face', '').replace('edge', '').replace('corner', '').strip()
    parcel_map = {1:[0,1,0], 2:[0,-1,0], 3:[0,0,1], 4:[0,0,-1], 5:[-1,0,0], 6:[1,0,0]}
    ltl_map = {1:[0,1,0], 2:[0,0,-1], 3:[0,-1,0], 4:[0,0,1], 5:[1,0,0], 6:[-1,0,0]}
    face_map = ltl_map if mode == 'LTL' else parcel_map
    vec = np.array([0.0, 0.0, 0.0])
    nums = [int(n) for n in re.findall(r'[1-6]', direct)]
    if nums:
        for n in nums: vec += np.array(face_map[n])
        mag = np.linalg.norm(vec)
        if mag > 1e-6: vec /= mag
    if np.linalg.norm(vec) < 1e-6:
        tokens = [t.strip() for t in re.split(r'[-,\s]+', direct)]
        for tk in tokens:
            if 'front' in tk: vec[2] = 1.0
            elif 'rear' in tk or 'back' in tk: vec[2] = -1.0
            elif 'top' in tk: vec[1] = 1.0
            elif 'bottom' in tk: vec[1] = -1.0
            elif 'left' in tk: vec[0] = -1.0
            elif 'right' in tk: vec[0] = 1.0
        mag = np.linalg.norm(vec)
        if mag > 1e-6: vec /= mag
    if np.linalg.norm(vec) < 1e-6:
        if mode == 'LTL': vec = np.array([0, -1, 0])
        else: vec = np.array([0, 0, 1])
    target_pt = np.array([vec[0]*box_w/2, vec[1]*box_h/2, vec[2]*box_d/2])
    if np.linalg.norm(target_pt) < 1e-6: target_pt = np.array([0, 0, box_d/2])
    return target_pt

def get_single_body_instance(body_name, config=None):
    config = get_default_config(config)
    bw, bh, bd, bt = config["box_w"], config["box_h"], config["box_d"], config["box_thick"]
    cg = config["cush_gap"]; cw, ch, cd = bw-2*bt, bh-2*bt, bd-2*bt
    aw, ah = config.get("assy_w", cw-0.3), config.get("assy_h", ch-0.3)
    od, ocd, cd_chas = config["oc_d"], config["occ_d"], config["chas_d"]
    ad, it = od+ocd+cd_chas, config["occ_ithick"]
    oc_cx = [-aw/2+it, aw/2-it]; oc_cy = [-ah/2+it, ah/2-it]
    if body_name == "BPaperBox":
        b = BPaperBox("BPaperBox", bw, bh, bd, config["mass_paper"], config["box_div"], bt, config["mat_paper"], config["box_use_weld"])
        b.build_geometry(offset=[0,0,0], rx=[-bw/2+bt, bw/2-bt], ry=[-bh/2+bt, bh/2-bt], rz=[-bd/2+bt, bd/2-bt])
        return b
    elif body_name == "BCushion":
        ab = [-aw/2, aw/2, -ah/2, ah/2, -ad/2, ad/2]
        cc = {"center":[0,0,0,cw*0.5,ch*0.5,cd*2]}
        b = BCushion("BCushion", cw, ch, cd, config["mass_cushion"], config["cush_div"], config["mat_cush"], ab, cg, cc, config["cush_use_weld"])
        rcx, rcy, rcz = [-aw/2-cg, aw/2+cg], [-ah/2-cg, ah/2+cg], [-ad/2-cg, ad/2+cg]
        b.build_geometry(offset=[0,0,0], rx=rcx, ry=rcy, rz=rcz)
        return b
    elif body_name == "BOpenCell":
        b = BOpenCell("BOpenCell", aw, ah, od, config["mass_oc"], config["oc_div"], config["mat_cell"], config["oc_use_weld"])
        b.build_geometry(offset=[0,0,0], rx=oc_cx, ry=oc_cy); return b
    elif body_name == "BOpenCellCohesive":
        b = BOpenCellCohesive("BOpenCellCohesive", aw, ah, ocd, config["mass_occ"], config["occ_div"], it, config["mat_tape"], config["occ_use_weld"])
        b.build_geometry(offset=[0,0,0], rx=oc_cx, ry=oc_cy); return b
    elif body_name == "BChassis":
        b = BChassis("BChassis", aw, ah, cd_chas, config["mass_chassis"], config["chassis_div"], config["mat_tv"], config["chassis_use_weld"])
        b.build_geometry(offset=[0,0,0], rx=oc_cx, ry=oc_cy); return b
    return None

def create_model(export_path, config=None, logger=print):
    config = get_default_config(config)
    bw, bh, bd, bt = config["box_w"], config["box_h"], config["box_d"], config["box_thick"]
    cw, ch, cd = bw-2*bt, bh-2*bt, bd-2*bt
    aw, ah = config.get("assy_w", cw-0.3), config.get("assy_h", ch-0.3)
    od, ocd, cd_chas = config["oc_d"], config["occ_d"], config["chas_d"]
    ad, it, cg = od+ocd+cd_chas, config["occ_ithick"], config["cush_gap"]
    root = BaseDiscreteBody("PackagingBox", 0,0,0, 0, [1,1,1], {})
    assy = BaseDiscreteBody("AssySet", 0,0,0, 0, [1,1,1], {})
    oz, ocz = ad/2-od/2, ad/2-od-ocd/2; chz = ocz-ocd/2-cd_chas/2
    b_oc = BOpenCell("BOpenCell", aw, ah, od, config["mass_oc"], config["oc_div"], config["mat_cell"], config["oc_use_weld"])
    b_occ = BOpenCellCohesive("BOpenCellCohesive", aw, ah, ocd, config["mass_occ"], config["occ_div"], it, config["mat_tape"], config["occ_use_weld"])
    b_ch = BChassis("BChassis", aw, ah, cd_chas, config["mass_chassis"], config["chassis_div"], config["mat_tv"], config["chassis_use_weld"])
    assy.add_child(b_oc); assy.add_child(b_occ); assy.add_child(b_ch)
    if config["include_paperbox"]: root.add_child(BPaperBox("BPaperBox", bw, bh, bd, config["mass_paper"], config["box_div"], bt, config["mat_paper"], config["box_use_weld"]))
    if config["include_cushion"]: root.add_child(BCushion("BCushion", cw, ch, cd, config["mass_cushion"], config["cush_div"], config["mat_cush"], [-aw/2,aw/2,-ah/2,ah/2,-ad/2,ad/2], cg, {"center":[0,0,0,cw*0.5,ch*0.5,cd*2]}, config["cush_use_weld"]))
    root.add_child(assy)
    # Build geometry and generate XML... (Full implementation skipped for brevity, user should re-run model build)
    return "XML"

if __name__ == "__main__":
    print("Discrete Model Builder Restored.")
"""

# Actually, I should NOT simplify it too much.
# I will read the original file and apply the patch properly.
# But wait, I have the pieces in my context.

# I'll just write a script that reads the file and performs a more robust text replacement.
"""
