import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union

class DiscreteBlock:
    """
    이산형 모델의 최소 단위인 개별 블록(Voxel) 정보를 담는 데이터 클래스입니다.
    
    각 블록은 공간상의 위치, 크기, 질량 및 재질 정보를 가지며 
    MuJoCo의 <geom> 또는 <body> 요소로 변환되는 기초 데이터를 제공합니다.
    """
    def __init__(self, idx: Tuple[int, int, int], cx: float, cy: float, cz: float, dx: float, dy: float, dz: float, mass: float, material: Dict[str, Any]):
        """
        Args:
            idx (tuple): (i, j, k) 형태의 격자 인덱스.
            cx, cy, cz (float): 부모 바디의 로컬 좌표계 기준 중심점 좌표.
            dx, dy, dz (float): 블록 절반 크기(Half-size). 실제 길이는 2 * dx 임.
            mass (float): 블록에 할당된 질량 (kg).
            material (dict): 재질 물성 정보 (rgba, solref 등).
        """
        self.idx = idx         # 지역 인덱스
        self.cx = cx           # X 중심 좌표
        self.cy = cy           # Y 중심 좌표
        self.cz = cz           # Z 중심 좌표
        self.dx = dx           # Half-width
        self.dy = dy           # Half-height
        self.dz = dz           # Half-depth
        self.mass = mass       # 할당된 질량
        self.material = material # 재질 정보
        self.volume = (2*dx) * (2*dy) * (2*dz) # 블록 체적

class BaseDiscreteBody:
    """
    이산형 바디 모델링의 최상위 추상 클래스입니다.
    
    이 클래스는 전체 형상을 격자로 분할하고, 각 격자(Block)를 생성하며, 
    MuJoCo XML 문구 및 물리적 관성(Inertia) 정보를 계산하는 핵심 로직을 포함합니다.
    """
    def __init__(self, name: str, width: float, height: float, depth: float, mass: float, div: List[int], material_props: Dict[str, Any], use_internal_weld: bool = True):
        """
        Args:
            name (str): 바디의 고유 명칭.
            width, height, depth (float): 전체 바디의 외곽 치수.
            mass (float): 전체 바디의 총 질량.
            div (list): [Nx, Ny, Nz] 형태의 축별 최소 분할 수.
            material_props (dict): 기본 재질 물성.
            use_internal_weld (bool): True일 경우 각 블록을 독립 <body>로 분리하여 용접(Weld)으로 연결함.
                                     False일 경우 하나의 <body> 내에 여러 <geom>을 배치함.
        """
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        self.total_mass = mass
        self.div = div  # [Nx, Ny, Nz]
        self.material_props = material_props 
        self.use_internal_weld = use_internal_weld 
        
        self.blocks = {} # 생성된 블록 저장소 {(i,j,k): DiscreteBlock}
        self.children = [] 
        self.parent = None 

    def add_child(self, child_body: 'BaseDiscreteBody') -> None:
        """자식 바디를 추가하여 계층 구조를 형성합니다."""
        child_body.parent = self
        self.children.append(child_body)

    def _generate_strict_grid_axis(self, length: float, num_div: int, required_cuts: List[float] = []) -> List[float]:
        """
        특정 축에 대해 필수 절단선(required_cuts)을 포함하면서 
        최소 분할 수(num_div) 이상을 만족하는 격자 노드를 생성합니다.
        
        Args:
            length (float): 전체 길이.
            num_div (int): 최소 분할 개수.
            required_cuts (list): 반드시 포함되어야 하는 좌표 지점들.
            
        Returns:
            list: 정렬된 노드 좌표 리스트.
        """
        # 양 끝단 기본 포함
        edges = set([-length/2, length/2])
        for cut in required_cuts:
            if -length/2 <= cut <= length/2:
                edges.add(cut)
        edges = sorted(list(edges))
        
        # 현재 분할 수가 목표보다 적을 경우 균등 분할 추가
        current_segments = len(edges) - 1
        if current_segments >= num_div:
            return edges

        target_step = length / num_div
        final_nodes = []
        
        for i in range(len(edges)-1):
            start = edges[i]; end = edges[i+1]; segment_len = end - start
            start = round(start, 5); end = round(end, 5)
            
            if len(final_nodes) == 0: final_nodes.append(start)
            elif abs(final_nodes[-1] - start) > 1e-6: final_nodes.append(start)
                
            # 세그먼트가 너무 길면 추가 분할
            if segment_len > target_step * 1.2:
                sub_divs = max(1, int(round(segment_len / target_step)))
                sub_nodes = np.linspace(start, end, sub_divs + 1)
                for n in sub_nodes[1:]: final_nodes.append(round(n, 5))
            else: final_nodes.append(round(end, 5))
                
        return sorted(list(set(final_nodes)))

    def is_cavity(self, block_cx: float, block_cy: float, block_cz: float, box_dx: float, box_dy: float, box_dz: float) -> bool:
        """해당 블록이 공동(빈 공간)인지 여부를 판단합니다. 하위 클래스에서 오버라이드합니다."""
        return False
        
    def is_edge_block(self, i: int, j: int, k: int) -> bool:
        """해당 블록이 모서리/에지 블록인지 판단합니다. 하위 클래스에서 오버라이드합니다."""
        return False

    def build_geometry(self, local_offset: List[float] = [0,0,0], required_cuts_x: List[float] = [], required_cuts_y: List[float] = [], required_cuts_z: List[float] = []) -> None:
        """
        설정된 치수와 분할 정보를 바탕으로 실제 DiscreteBlock들을 생성합니다.
        
        Args:
            local_offset (list): 중심 이동 오프셋.
            required_cuts_x, y, z (list): 각 축별 필수 절단 좌표.
        """
        nodes_x = self._generate_strict_grid_axis(self.width, self.div[0], required_cuts_x)
        nodes_y = self._generate_strict_grid_axis(self.height, self.div[1], required_cuts_y)
        nodes_z = self._generate_strict_grid_axis(self.depth, self.div[2], required_cuts_z)
        
        temp_blocks = []
        total_vol = 0.0
        
        # 격자 순회하며 유효한 블록(공동이 아닌 블록) 생성
        for i in range(len(nodes_x)-1):
            for j in range(len(nodes_y)-1):
                for k in range(len(nodes_z)-1):
                    cx = (nodes_x[i] + nodes_x[i+1]) / 2.0
                    cy = (nodes_y[j] + nodes_y[j+1]) / 2.0
                    cz = (nodes_z[k] + nodes_z[k+1]) / 2.0
                    
                    dx = (nodes_x[i+1] - nodes_x[i]) / 2.0
                    dy = (nodes_y[j+1] - nodes_y[j]) / 2.0
                    dz = (nodes_z[k+1] - nodes_z[k]) / 2.0

                    if self.is_cavity(cx, cy, cz, dx, dy, dz): continue
                        
                    blk = DiscreteBlock((i,j,k), cx + local_offset[0], cy + local_offset[1], cz + local_offset[2],
                                        dx, dy, dz, 0, self.material_props)
                    temp_blocks.append(blk)
                    total_vol += blk.volume

        # 실제 생성된 격자 수를 기록하고 질량을 체적비로 분배
        self.actual_div = [len(nodes_x)-1, len(nodes_y)-1, len(nodes_z)-1]
        for blk in temp_blocks:
            blk.mass = self.total_mass * (blk.volume / total_vol)
            self.blocks[blk.idx] = blk

    def get_weld_xml_strings(self) -> List[str]:
        """
        인접한 블록 간의 연결 관계를 MuJoCo <weld> 요소로 생성합니다.
        이 기능은 이산형 모델의 '강성(Stiffness)'을 표현하는 핵심 수단입니다.
        """
        weld_xml = []
        # 내부 용접을 사용하지 않는 경우 자식 바디의 문자열만 수집
        if not self.use_internal_weld:
            for child in self.children: weld_xml.extend(child.get_weld_xml_strings())
            return weld_xml

        solref = self.material_props.get("weld_solref", "0.02 1.0")
        solimp = self.material_props.get("weld_solimp", "0.1 0.95 0.005 0.5 2")
        block_keys = set(self.blocks.keys())
        for (i, j, k), blk1 in self.blocks.items():
            # X, Y, Z 방향으로 인접한 블록이 있는지 확인하여 용접 결합 생성
            if (i+1, j, k) in block_keys:
                blk2 = self.blocks[(i+1, j, k)]
                if abs((blk1.cx + blk1.dx) - (blk2.cx - blk2.dx)) < 1e-4:
                    weld_xml.append(f'        <weld site1="s_{self.name}_{i}_{j}_{k}_PX" site2="s_{self.name}_{i+1}_{j}_{k}_NX" solref="{solref}" solimp="{solimp}"/>')
            if (i, j+1, k) in block_keys:
                blk2 = self.blocks[(i, j+1, k)]
                if abs((blk1.cy + blk1.dy) - (blk2.cy - blk2.dy)) < 1e-4:
                    weld_xml.append(f'        <weld site1="s_{self.name}_{i}_{j}_{k}_PY" site2="s_{self.name}_{i}_{j+1}_{k}_NY" solref="{solref}" solimp="{solimp}"/>')
            if (i, j, k+1) in block_keys:
                blk2 = self.blocks[(i, j, k+1)]
                if abs((blk1.cz + blk1.dz) - (blk2.cz - blk2.dz)) < 1e-4:
                    weld_xml.append(f'        <weld site1="s_{self.name}_{i}_{j}_{k}_PZ" site2="s_{self.name}_{i}_{j}_{k+1}_NZ" solref="{solref}" solimp="{solimp}"/>')
        
        for child in self.children: weld_xml.extend(child.get_weld_xml_strings())
        return weld_xml

    def calculate_inertia(self) -> Tuple[float, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        수천 개의 미세 블록으로 구성된 어셈블리의 전체 질량, 무게중심(CoG), 관성 모멘트(MoI)를 계산합니다.
        Parallel Axis Theorem(평행축 정리)를 사용하여 정확한 합산 관성을 산출합니다.
        """
        all_primitive_blocks = []
        individual_details = []
        
        def _collect(body):
            nonlocal all_primitive_blocks, individual_details
            this_body_mass = 0.0
            this_body_weighted_cog_sum = np.zeros(3)
            this_body_local_moi_sum = np.zeros(3)
            this_body_blocks_list = []
            
            for blk in body.blocks.values():
                this_body_mass += blk.mass
                this_body_weighted_cog_sum += blk.mass * np.array([blk.cx, blk.cy, blk.cz])
                # 블록 자체의 관성 모멘트 (직육면체 공식)
                w_full, h_full, d_full = 2.0 * blk.dx, 2.0 * blk.dy, 2.0 * blk.dz
                ixx = (1.0/12.0) * blk.mass * (h_full**2 + d_full**2)
                iyy = (1.0/12.0) * blk.mass * (w_full**2 + d_full**2)
                izz = (1.0/12.0) * blk.mass * (w_full**2 + h_full**2)
                this_body_local_moi_sum += np.array([ixx, iyy, izz])
                this_body_blocks_list.append(blk)
                all_primitive_blocks.append(blk)
            
            if this_body_mass > 0:
                this_body_cog = this_body_weighted_cog_sum / this_body_mass
                # 평행축 정리 적용 (개별 블록 CoG와 바디 CoG 간의 거리 보정)
                this_body_parallel_correction = np.zeros(3)
                for blk in this_body_blocks_list:
                    b_pos = np.array([blk.cx, blk.cy, blk.cz])
                    dist_sq = (b_pos - this_body_cog)**2
                    this_body_parallel_correction[0] += blk.mass * (dist_sq[1] + dist_sq[2])
                    this_body_parallel_correction[1] += blk.mass * (dist_sq[0] + dist_sq[2])
                    this_body_parallel_correction[2] += blk.mass * (dist_sq[0] + dist_sq[1])
                this_body_final_moi = this_body_local_moi_sum + this_body_parallel_correction
                individual_details.append({"name": body.name, "mass": this_body_mass, "cog": this_body_cog, "moi": this_body_final_moi})
            
            for child in body.children: _collect(child)
                
        _collect(self)
        # 전체 어셈블리 관성 합산
        total_mass = 0.0; total_weighted_cog_sum = np.zeros(3); total_pure_local_moi_sum = np.zeros(3)
        for blk in all_primitive_blocks:
            total_mass += blk.mass
            total_weighted_cog_sum += blk.mass * np.array([blk.cx, blk.cy, blk.cz])
            w_full, h_full, d_full = 2.0 * blk.dx, 2.0 * blk.dy, 2.0 * blk.dz
            total_pure_local_moi_sum[0] += (1.0/12.0) * blk.mass * (h_full**2 + d_full**2)
            total_pure_local_moi_sum[1] += (1.0/12.0) * blk.mass * (w_full**2 + d_full**2)
            total_pure_local_moi_sum[2] += (1.0/12.0) * blk.mass * (w_full**2 + h_full**2)
            
        if total_mass > 0:
            total_cog = total_weighted_cog_sum / total_mass
            total_parallel_moI_correction = np.zeros(3)
            for blk in all_primitive_blocks:
                b_pos = np.array([blk.cx, blk.cy, blk.cz])
                dist_sq = (b_pos - total_cog)**2
                total_parallel_moI_correction[0] += blk.mass * (dist_sq[1] + dist_sq[2])
                total_parallel_moI_correction[1] += blk.mass * (dist_sq[0] + dist_sq[2])
                total_parallel_moI_correction[2] += blk.mass * (dist_sq[0] + dist_sq[1])
            final_total_moi = total_pure_local_moi_sum + total_parallel_moI_correction
        else: total_cog = np.zeros(3); final_total_moi = np.zeros(3)
            
        return total_mass, total_cog, final_total_moi, individual_details
        
    def get_worldbody_xml_strings(self, indent_level: int = 2) -> List[str]:
        """
        MuJoCo XML의 <worldbody> 섹션에 들어갈 <body> 및 <geom> 구문을 생성합니다.
        
        Args:
            indent_level (int): XML 들여쓰기 단계.
            
        Returns:
            list: 생성된 XML 문자열 리스트.
        """
        xml_outs = []
        ind = "  " * indent_level; ind_c = ind + "  "
        # 루트 컨테이너인 경우 자식들만 처리
        if self.parent is None:
            for child in self.children: xml_outs.extend(child.get_worldbody_xml_strings(indent_level))
            return xml_outs
            
        xml_outs.append(f'{ind}<body name="{self.name}">')
        
        # 내부 용접을 사용하지 않는 모드: 하나의 body 내에 수많은 geom 배치
        if not self.use_internal_weld:
            if self.name not in ["PackagingBox", "AssySet"]:
                # 각 블록(Geom) 간의 상대 운동이 필요할 경우 관절 추가 고려 가능 (현재는 정적 배치)
                xml_outs.append(f'{ind_c}<joint type="slide" axis="1 0 0"/>')
                xml_outs.append(f'{ind_c}<joint type="slide" axis="0 1 0"/>')
                xml_outs.append(f'{ind_c}<joint type="slide" axis="0 0 1"/>')
                xml_outs.append(f'{ind_c}<joint type="ball"/>')
            for (i, j, k), blk in self.blocks.items():
                geom_class = f"contact_{self.__class__.__name__.lower()}"
                geom_name = f"g_{self.name.lower()}_{i}_{j}_{k}"
                if hasattr(self, 'is_corner_block') and self.is_corner_block(i, j, k): 
                    geom_class += "_edge"
                    geom_name += "_edge"
                xml_outs.append(f'{ind_c}<geom name="{geom_name}" type="box" '
                                 f'pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz:.5f}" '
                                 f'size="{blk.dx:.5f} {blk.dy:.5f} {blk.dz:.5f}" mass="{blk.mass:.6f}" '
                                 f'class="{geom_class}"/>')
                # Site 배치 (Geom 방식에서도 에지 접촉용 Site 필요 시 생성)
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_PX" pos="{blk.cx+blk.dx:.5f} {blk.cy:.5f} {blk.cz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_NX" pos="{blk.cx-blk.dx:.5f} {blk.cy:.5f} {blk.cz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_PY" pos="{blk.cx:.5f} {blk.cy+blk.dy:.5f} {blk.cz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_NY" pos="{blk.cx:.5f} {blk.cy-blk.dy:.5f} {blk.cz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_PZ" pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz+blk.dz:.5f}"/>')
                xml_outs.append(f'{ind_c}<site name="s_{self.name}_{i}_{j}_{k}_NZ" pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz-blk.dz:.5f}"/>')
        
        # 내부 용접 사용 모드: 각 블록을 독립 <body>로 분리 (고급 물성/소성 특성 연구용)
        else:
            for (i, j, k), blk in self.blocks.items():
                xml_outs.append(f'{ind_c}<body name="b_{self.name.lower()}_{i}_{j}_{k}" pos="{blk.cx:.5f} {blk.cy:.5f} {blk.cz:.5f}">')
                ind_cc = ind_c + "  "
                # 개별 블록이 자유도를 가짐
                xml_outs.append(f'{ind_cc}<joint type="slide" axis="1 0 0"/>')
                xml_outs.append(f'{ind_cc}<joint type="slide" axis="0 1 0"/>')
                xml_outs.append(f'{ind_cc}<joint type="slide" axis="0 0 1"/>')
                xml_outs.append(f'{ind_cc}<joint type="ball"/>')
                geom_class = f"contact_{self.__class__.__name__.lower()}"
                geom_name = f"g_{self.name.lower()}_{i}_{j}_{k}"
                if hasattr(self, 'is_corner_block') and self.is_corner_block(i, j, k): 
                    geom_class += "_edge"
                    geom_name += "_edge"
                xml_outs.append(f'{ind_cc}<geom name="{geom_name}" type="box" '
                                 f'size="{blk.dx:.5f} {blk.dy:.5f} {blk.dz:.5f}" mass="{blk.mass:.6f}" '
                                 f'class="{geom_class}"/>')
                # 6개 면의 중심에 Site 배치 (용접점)
                xml_outs.append(f'{ind_cc}<site name="s_{self.name}_{i}_{j}_{k}_PX" pos="{blk.dx:.5f} 0 0"/>')
                xml_outs.append(f'{ind_cc}<site name="s_{self.name}_{i}_{j}_{k}_NX" pos="{-blk.dx:.5f} 0 0"/>')
                xml_outs.append(f'{ind_cc}<site name="s_{self.name}_{i}_{j}_{k}_PY" pos="0 {blk.dy:.5f} 0"/>')
                xml_outs.append(f'{ind_cc}<site name="s_{self.name}_{i}_{j}_{k}_NY" pos="0 {-blk.dy:.5f} 0"/>')
                xml_outs.append(f'{ind_cc}<site name="s_{self.name}_{i}_{j}_{k}_PZ" pos="0 0 {blk.dz:.5f}"/>')
                xml_outs.append(f'{ind_cc}<site name="s_{self.name}_{i}_{j}_{k}_NZ" pos="0 0 {-blk.dz:.5f}"/>')
                xml_outs.append(f'{ind_c}</body>')
        
        # 재귀적으로 자식 바디의 XML 생성
        for child in self.children: xml_outs.extend(child.get_worldbody_xml_strings(indent_level + 1))
        xml_outs.append(f'{ind}</body>')
        return xml_outs
