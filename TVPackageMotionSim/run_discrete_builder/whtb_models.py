from .whtb_base import BaseDiscreteBody, DiscreteBlock
from typing import List, Dict, Optional, Any, Tuple, Union

class BPaperBox(BaseDiscreteBody):
    """
    골판지 상자(Outer Box) 모델을 생성하는 클래스입니다.
    
    상자의 두께(thick)를 제외한 내부 공간을 공동(Cavity) 처리하여 
    껍데기만 있는 박스 형태를 이산형 블록들로 재구성합니다.
    """
    def __init__(self, name: str, width: float, height: float, depth: float, mass: float, div: List[int], thick: float, material_props: Dict[str, Any], use_internal_weld: bool = True):
        """
        Args:
            name (str): 부품 명칭.
            width, height, depth (float): 박스 외곽 치수.
            mass (float): 박스 총 질량.
            div (list): 최소 분할 수 [Nx, Ny, Nz].
            thick (float): 박스 벽면의 두께.
            material_props (dict): 재질 물성 정보.
            use_internal_weld (bool): 내부 블록 간 용접 사용 여부.
        """
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.thick = thick
        
    def is_cavity(self, cx: float, cy: float, cz: float, dx: float, dy: float, dz: float) -> bool:
        """
        현재 블록의 중심(cx, cy, cz)이 박스의 두께를 제외한 내부 '빈 공간'에 있는지 판단합니다.
        
        Algorithm:
            - 중심점이 각 축 방향으로 (전체 길이/2 - 두께) 범위 안에 모두 위치하면 Cavity(True)로 판정.
        """
        in_x = abs(cx) < (self.width/2 - self.thick - 1e-4)
        in_y = abs(cy) < (self.height/2 - self.thick - 1e-4)
        in_z = abs(cz) < (self.depth/2 - self.thick - 1e-4)
        return in_x and in_y and in_z

class BCushion(BaseDiscreteBody):
    """
    포장용 완충재(EPS, EPP 등) 모델을 생성하는 클래스입니다.
    
    제품이 안착될 공간(assy_bbox)과 추가적으로 파낼 부위(cushion_cutter)를 
    반영하여 복잡한 완충재 형상을 이산화합니다.
    """
    def __init__(self, name: str, width: float, height: float, depth: float, mass: float, div: List[int], material_props: Dict[str, Any], assy_bbox: List[float], gap: float, cushion_cutter: Dict[str, List[float]], use_internal_weld: bool = True):
        """
        Args:
            assy_bbox (list): 내부 제품의 [minX, maxX, minY, maxY, minZ, maxZ] 경계 상자.
            gap (float): 제품과 완충재 사이의 간격(Offset).
            cushion_cutter (dict): 추가로 제거할 공간들의 정보 {name: [cx, cy, cz, w, h, d]}.
        """
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.assy_bbox = assy_bbox
        self.gap = gap
        self.cushion_cutter = cushion_cutter

    def is_cavity(self, cx: float, cy: float, cz: float, dx: float, dy: float, dz: float) -> bool:
        """
        제품 안착 공간 또는 사용자 정의 커터 부위에 블록이 포함되는지 확인합니다.
        """
        # 1. 제품 안착용 공간 체크
        ax_min, ax_max, ay_min, ay_max, az_min, az_max = self.assy_bbox
        if (ax_min - self.gap <= cx <= ax_max + self.gap and
            ay_min - self.gap <= cy <= ay_max + self.gap and
            az_min - self.gap <= cz <= az_max + self.gap):
            return True
            
        # 2. 추가 커터(Cutter) 부위 체크
        for cut_vals in self.cushion_cutter.values():
            ctx, cty, ctz, cw, ch, cd = cut_vals
            if (ctx - cw/2 <= cx <= ctx + cw/2 and 
                cty - ch/2 <= cy <= cty + ch/2 and 
                ctz - cd/2 <= cz <= ctz + cd/2):
                return True
        return False

    def is_edge_block(self, i: int, j: int, k: int) -> bool:
        """
        격자 인덱스 (i, j, k)를 보고 해당 블록이 완충재의 어느 모서리(에지)든 포함되는지 판단합니다.
        (12개의 모든 모서리 포함)
        """
        nx, ny, nz = getattr(self, 'actual_div', self.div)
        bx = (i == 0 or i == nx - 1)
        by = (j == 0 or j == ny - 1)
        bz = (k == 0 or k == nz - 1)
        return (bx and by) or (by and bz) or (bz and bx)

    def is_corner_block(self, i: int, j: int, k: int) -> bool:
        """
        8개의 꼭짓점과 'Depth(Z)' 방향의 4개 모서리 기둥에 해당하는지 판단합니다.
        (ix, iy가 모두 끝단인 경우)
        """
        nx, ny, nz = getattr(self, 'actual_div', self.div)
        bx = (i == 0 or i == nx - 1)
        by = (j == 0 or j == ny - 1)
        return bx and by

    def get_weld_xml_strings(self) -> List[str]:
        """
        완충재 내부 블록들을 연결하는 Weld 구문을 생성합니다.
        특히 '모서리 용접(Corner Weld)' 물성을 별도로 분리하여 설정할 수 있는 기능을 제공합니다.
        """
        weld_xml = []
        if not self.use_internal_weld:
            return weld_xml
            
        weld_class_base = "weld_bcushion"
        # 모서리 전용 물성(solref 등)이 설정되어 있는지 확인
        has_corner_weld = "corner_weld_solref" in self.material_props and self.material_props["corner_weld_solref"]
        solref = self.material_props.get("weld_solref", "0.02 1.0")
        solimp = self.material_props.get("weld_solimp", "0.1 0.95 0.005 0.5 2")
        block_keys = set(self.blocks.keys())
        
        for (i, j, k), blk1 in self.blocks.items():
            is_c1 = self.is_edge_block(i, j, k)
            
            # X, Y, Z 양방향 인접 블록 탐색
            for di, dj, dk, suffix in [(1,0,0, "PX"), (0,1,0, "PY"), (0,0,1, "PZ")]:
                ni, nj, nk = i+di, j+dj, k+dk
                if (ni, nj, nk) in block_keys:
                    blk2 = self.blocks[(ni, nj, nk)]
                    match = False
                    if di == 1: match = abs((blk1.cx + blk1.dx) - (blk2.cx - blk2.dx)) < 1e-4
                    elif dj == 1: match = abs((blk1.cy + blk1.dy) - (blk2.cy - blk2.dy)) < 1e-4
                    elif dk == 1: match = abs((blk1.cz + blk1.dz) - (blk2.cz - blk2.dz)) < 1e-4
                    
                    if match:
                        is_c2 = self.is_edge_block(ni, nj, nk)
                        curr_solref = solref
                        if has_corner_weld and (is_c1 or is_c2):
                            curr_solref = self.material_props["corner_weld_solref"]
                            
                        site1_name = f"s_{self.name}_{i}_{j}_{k}_{suffix}"
                        opp_suffix = "NX" if suffix=="PX" else "NY" if suffix=="PY" else "NZ"
                        site2_name = f"s_{self.name}_{ni}_{nj}_{nk}_{opp_suffix}"
                        weld_xml.append(f'        <weld site1="{site1_name}" site2="{site2_name}" solref="{curr_solref}" solimp="{solimp}"/>')
                        
        # 자식 요소들의 Weld 정보도 병합
        for child in self.children:
            weld_xml.extend(child.get_weld_xml_strings())
        return weld_xml

class BOpenCellCohesive(BaseDiscreteBody):
    """
    OpenCell 구조를 가진 점착성 부재를 모델링합니다.
    특정 두께(ithick)만큼의 테두리만 남기고 가운데를 파낸 형상을 기본으로 합니다.
    """
    def __init__(self, name: str, width: float, height: float, depth: float, mass: float, div: List[int], ithick: float, material_props: Dict[str, Any], use_internal_weld: bool = True):
        super().__init__(name, width, height, depth, mass, div, material_props, use_internal_weld)
        self.ithick = ithick
        
    def is_cavity(self, cx: float, cy: float, cz: float, dx: float, dy: float, dz: float) -> bool:
        # 상하(Z) 면은 뚫지 않고 전/후/좌/우 테두리 두께만 유지
        in_x = abs(cx) < (self.width/2 - self.ithick - 1e-4)
        in_y = abs(cy) < (self.height/2 - self.ithick - 1e-4)
        return in_x and in_y

class BOpenCell(BaseDiscreteBody):
    """기타 OpenCell 구조체 (추가 로직 필요 시 구현)"""
    pass

class BChassis(BaseDiscreteBody):
    """제품의 섀시/프레임을 이산화하는 클래스 (현재는 전체 채움)"""
    pass

class BAuxBoxMass(BaseDiscreteBody):
    """
    보조 질량(Dummy Weight) 등을 표현하기 위한 단순 박스 모델입니다.
    격자 분할 없이 하나의 큰 블록으로 생성되며, 충돌 속성을 끄고 질량만 기여하도록 설정됩니다.
    """
    def __init__(self, name: str, width: float, height: float, depth: float, mass: float, material_props: Optional[Dict[str, Any]] = None):
        if material_props is None:
            material_props = {"rgba": "1.0 0.0 0.0 0.4", "solref": "0.02 1.0", "solimp": "0.1 0.95 0.005"}
        # 분할을 [1, 1, 1]로 강제함
        super().__init__(name, width, height, depth, mass, [1, 1, 1], material_props)
        # 기본적으로 충돌 회피 설정
        self.material_props["contype"] = "0"
        self.material_props["conaffinity"] = "0"
        
    def build_geometry(self, local_offset: List[float] = [0, 0, 0]) -> None:
        """단일 덩어리로 블록 생성"""
        aux_block = DiscreteBlock(idx=(0, 0, 0), cx=local_offset[0], cy=local_offset[1], cz=local_offset[2],
                                  dx=self.width / 2.0, dy=self.height / 2.0, dz=self.depth / 2.0, 
                                  mass=self.total_mass, material=self.material_props)
        self.blocks[(0, 0, 0)] = aux_block

class BUnitBlock(BaseDiscreteBody):
    """단위 테스트 또는 특수 부재용 블록 클래스"""
    pass
