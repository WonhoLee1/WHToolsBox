import os

path = 'c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_discrete_builder/__init__.py'
if not os.path.exists(path):
    print(f"Error: {path} not found")
    exit(1)

content = open(path, 'r', encoding='utf-8').read()

# 1. Keep everything before the corrupted BPaperBox
start_marker = 'class BPaperBox(BaseDiscreteBody):'
idx_start = content.find(start_marker)

# 2. Keep everything from get_single_body_instance onwards
end_marker = 'def get_single_body_instance(body_name, config=None):'
# We want the LAST occurrence or the one that has docstrings.
# Actually, since I messed up, let's look for the one with the docstring.
end_marker_robust = 'def get_single_body_instance(body_name, config=None):\n    \"\"\"\n    단일 부품(개단품) 강성 평가용 지오메트리 추출기.'
idx_end = content.find(end_marker_robust)

if idx_start == -1 or idx_end == -1:
    print(f"Error finding markers: start={idx_start}, end={idx_end}")
    # Fallback to a simpler end marker if robust one fails
    idx_end = content.find(end_marker)
    if idx_end == -1: exit(1)

head = content[:idx_start]
tail = content[idx_end:]

middle = """class BPaperBox(BaseDiscreteBody):
    def __init__(self, name, width, height, depth, mass, div, thick, material_props, use_internal_weld=True):
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.thick = thick
        
    def is_cavity(self, cx, cy, cz, dx, dy, dz):
        in_x = abs(cx) < (self.width/2 - self.thick - 1e-4)
        in_y = abs(cy) < (self.height/2 - self.thick - 1e-4)
        in_z = abs(cz) < (self.depth/2 - self.thick - 1e-4)
        return in_x and in_y and in_z

class BCushion(BaseDiscreteBody):
    def __init__(self, name, width, height, depth, mass, div, material_props, assy_bbox, gap, cushion_cutter, use_internal_weld=True):
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.assy_bbox = assy_bbox
        self.gap = gap
        self.cushion_cutter = cushion_cutter

    def is_cavity(self, cx, cy, cz, dx, dy, dz):
        ax_min, ax_max, ay_min, ay_max, az_min, az_max = self.assy_bbox
        if (ax_min - self.gap <= cx <= ax_max + self.gap and
            ay_min - self.gap <= cy <= ay_max + self.gap and
            az_min - self.gap <= cz <= az_max + self.gap):
            return True
        for cut_vals in self.cushion_cutter.values():
            ctx, cty, ctz, cw, ch, cd = cut_vals
            if (ctx - cw/2 <= cx <= ctx + cw/2 and
                cty - ch/2 <= cy <= cty + ch/2 and
                ctz - cd/2 <= cz <= ctz + cd/2):
                return True
        return False

    def is_edge_block(self, i, j, k):
        nx, ny, nz = self.div
        bx = (i == 0 or i == nx - 1)
        by = (j == 0 or j == ny - 1)
        bz = (k == 0 or k == nz - 1)
        return (bx and by) or (by and bz) or (bz and bx)

class BOpenCellCohesive(BaseDiscreteBody):
    def __init__(self, name, width, height, depth, mass, div, ithick, material_props, use_internal_weld=True):
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.ithick = ithick

    def is_cavity(self, cx, cy, cz, dx, dy, dz):
        in_x = abs(cx) < (self.width/2 - self.ithick - 1e-4)
        in_y = abs(cy) < (self.height/2 - self.ithick - 1e-4)
        return in_x and in_y

class BOpenCell(BaseDiscreteBody): pass
class BChassis(BaseDiscreteBody): pass

class BAuxBoxMass(BaseDiscreteBody):
    def __init__(self, name, width, height, depth, mass, material_props=None):
        if material_props is None:
            material_props = {"rgba": "1.0 0.0 0.0 0.4", "solref": "0.02 1.0", "solimp": "0.1 0.95 0.005"}
        super().__init__(name, width, height, depth, mass, [1, 1, 1], material_props)
        self.material_props["contype"] = "0"
        self.material_props["conaffinity"] = "0"
    def build_geometry(self, local_offset=[0, 0, 0]):
        aux_block = DiscreteBlock(idx=(0, 0, 0), cx=local_offset[0], cy=local_offset[1], cz=local_offset[2],
                                  dx=self.width / 2.0, dy=self.height / 2.0, dz=self.depth / 2.0, mass=self.total_mass, material=self.material_props)
        self.blocks[(0, 0, 0)] = aux_block

class BUnitBlock(BaseDiscreteBody): pass

# =====================================================================
# [5] 메인 어셈블리 및 파일 생성
# =====================================================================

def parse_drop_target(mode_str, direction_str, box_w, box_h, box_d):
    \"\"\"
    낙하 모드와 방향 문자열을 파싱하여 바닥에 닿을 로컬 좌표(타겟 점)를 반환합니다.
    - drop_mode: PARCEL, LTL, CUSTOM (공정/시험 기준 분류용)
    - drop_direction: 'Face 1', 'Edge 3-4', 'Corner 2-3-5', '235', 'front-bottom-left' 등
    \"\"\"
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
        if mode == 'LTL': vec = np.array([0.0, -1.0, 0.0])
        else: vec = np.array([0.0, 0.0, 1.0])

    target_pt = np.array([vec[0] * box_w/2, vec[1] * box_h/2, vec[2] * box_d/2])
    if np.linalg.norm(target_pt) < 1e-6:
        target_pt = np.array([0, 0, box_d/2])
    return target_pt

"""

with open(path, 'w', encoding='utf-8') as f:
    f.write(head + middle + tail)

print("File structure fixed successfully.")
