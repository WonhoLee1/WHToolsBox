"""
TV Packaging Gmsh Mesh Generator + MuJoCo XML
==============================================
ISTA 6A 낙하 규정 기반 TV 포장재 FE 메쉬 생성 스크립트.

Bodies:
  - Box      : 속이 빈 골판지 상자 (중공 쉘, 3D 볼륨 메쉬)
  - Cushion  : 쿠션 (내부 제품 형상 컷아웃 포함)
  - Chassis  : TV 샤시 (단순 육면체)
  - Display  : TV 디스플레이 패널 (단순 육면체)
  - DispCoh  : 디스플레이-샤시 점착층 (창틀 Picture-Frame 형상)

좌표계 (Parcel 기준):
  - 전면(Front) = 디스플레이 화면 방향 = -Z 방향
  - 샤시가 -Z쪽, 디스플레이가 +Z쪽
  - LTL은 반대: 디스플레이가 -Z쪽, 샤시가 +Z쪽

출력 (test_box_msh/ 디렉토리):
  - box.msh, cushion.msh, chassis.msh, display.msh, dispcoh.msh
  - packaging_A_flexcomp.xml
  - packaging_B_rigid.xml

단위: mm (MuJoCo XML에서 mesh scale="0.001 0.001 0.001"로 m 변환)
MuJoCo 3.5.0+ 호환.

주의사항 (Gmsh OCC 안정성):
  - occ.rotate()로 평면을 만드는 방식은 불안정 → 얇은 박스로 슬라이서 생성
  - 불린 연산 전후 반드시 occ.synchronize() 호출
  - 바디별 독립 Gmsh 세션으로 깨끗한 단일-Physical-Group MSH 파일 생성
"""

import gmsh
import os
import math
import struct


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ascii_stl_to_binary(ascii_path, binary_path):
    """
    Gmsh가 내보낸 ASCII STL을 MuJoCo 호환 바이너리 STL로 변환.
    MuJoCo는 바이너리 STL만 지원 (ASCII STL 미지원).
    """
    triangles = []
    normal = (0.0, 0.0, 0.0)
    verts = []
    with open(ascii_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('facet normal'):
                parts = line.split()
                try:
                    normal = (float(parts[2]), float(parts[3]), float(parts[4]))
                except (IndexError, ValueError):
                    normal = (0.0, 0.0, 0.0)
                verts = []
            elif line.startswith('vertex'):
                parts = line.split()
                try:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except (IndexError, ValueError):
                    pass
            elif line.startswith('endfacet'):
                if len(verts) == 3:
                    triangles.append((normal, verts[0], verts[1], verts[2]))
                verts = []

    with open(binary_path, 'wb') as f:
        # 80-byte header
        f.write(b'\x00' * 80)
        # number of triangles (uint32)
        f.write(struct.pack('<I', len(triangles)))
        for (nx, ny, nz), v0, v1, v2 in triangles:
            f.write(struct.pack('<fff', nx, ny, nz))
            for vx, vy, vz in (v0, v1, v2):
                f.write(struct.pack('<fff', vx, vy, vz))
            f.write(struct.pack('<H', 0))  # attribute byte count

def _fix_msh41_single_block(msh_path):
    """
    MSH 4.1 파일을 읽어서 노드/요소를 단일 블록으로 재작성.
    MuJoCo flexcomp type="gmsh"는 단일 노드 블록 + tet4(type 4) 요소만 지원.

    MSH 4.1 $Nodes 형식:
      numEntityBlocks numNodes minTag maxTag
      entityDim entityTag parametric numNodesInBlock
      nodeTag ...
      x y z
    → 단일 블록(entityDim=3, entityTag=1)으로 병합
    """
    with open(msh_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = content.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # ── $Nodes 섹션 재작성 ────────────────────────────────────────────────
        if line == '$Nodes':
            i += 1
            header = lines[i].strip().split()
            # numEntityBlocks numNodes minTag maxTag
            num_nodes = int(header[1])
            min_tag = int(header[2])
            max_tag = int(header[3])
            i += 1

            # 모든 블록에서 (tag, x, y, z) 수집
            nodes = {}  # tag → (x, y, z)
            while i < len(lines) and lines[i].strip() != '$EndNodes':
                blk_header = lines[i].strip().split()
                i += 1
                if len(blk_header) < 4:
                    continue
                n_in_blk = int(blk_header[3])
                tags = []
                for _ in range(n_in_blk):
                    tags.append(int(lines[i].strip()))
                    i += 1
                for t in tags:
                    xyz = lines[i].strip().split()
                    nodes[t] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                    i += 1
            i += 1  # $EndNodes

            # 단일 블록으로 재작성
            out.append('$Nodes')
            out.append(f'1 {num_nodes} {min_tag} {max_tag}')
            out.append(f'3 1 0 {num_nodes}')  # entityDim=3, entityTag=1, parametric=0
            sorted_tags = sorted(nodes.keys())
            for t in sorted_tags:
                out.append(str(t))
            for t in sorted_tags:
                x, y, z = nodes[t]
                out.append(f'{x} {y} {z}')
            out.append('$EndNodes')
            continue

        # ── $Elements 섹션: tet4(type 4)만 단일 블록으로 재작성 ───────────────
        elif line == '$Elements':
            i += 1
            header = lines[i].strip().split()
            # numEntityBlocks numElements minTag maxTag
            i += 1

            # 모든 블록에서 tet4 요소 수집
            tets = []  # (tag, n1, n2, n3, n4)
            while i < len(lines) and lines[i].strip() != '$EndElements':
                blk_header = lines[i].strip().split()
                i += 1
                if len(blk_header) < 4:
                    continue
                elem_type = int(blk_header[2])
                n_in_blk = int(blk_header[3])
                for _ in range(n_in_blk):
                    parts = lines[i].strip().split()
                    i += 1
                    if elem_type == 4 and len(parts) == 5:  # tet4
                        tets.append(tuple(int(p) for p in parts))
            i += 1  # $EndElements

            if not tets:
                # tet4가 없으면 원본 유지 (비정상 케이스)
                out.append('$Elements')
                out.append(f'0 0 0 0')
                out.append('$EndElements')
                continue

            min_e = tets[0][0]
            max_e = tets[-1][0]
            out.append('$Elements')
            out.append(f'1 {len(tets)} {min_e} {max_e}')
            out.append(f'3 1 4 {len(tets)}')  # entityDim=3, entityTag=1, elemType=4(tet4)
            for tet in tets:
                out.append(' '.join(str(v) for v in tet))
            out.append('$EndElements')
            continue

        else:
            out.append(lines[i])
            i += 1

    with open(msh_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


def _box_centered(occ, cx, cy, cz, w, h, d):
    """중심 (cx,cy,cz) 기준 w×h×d 육면체. 반환: tag(int)"""
    return occ.addBox(cx - w/2, cy - h/2, cz - d/2, w, h, d)


def _vol_tags(ents):
    """(dim, tag) 리스트에서 dim==3 tag 목록 반환."""
    return [tag for dim, tag in ents if dim == 3]


def _mesh_session(body_name, build_fn, elem_size, out_msh, out_stl=None):
    """
    독립 Gmsh 세션에서 단일 바디를 생성·메쉬·저장.
    build_fn(occ) → [(3, tag), ...] 형태로 볼륨 엔티티 반환.

    out_msh : 볼륨 테트라 메쉬 (MSH 4.1 ASCII) - flexcomp type="gmsh"용
    out_stl : 표면 삼각형 메쉬 (STL) - rigid geom용 (None이면 생략)
    """
    gmsh.initialize()
    gmsh.model.add(body_name)
    occ = gmsh.model.occ

    try:
        vol_ents = build_fn(occ)
        occ.synchronize()

        vol_tags = _vol_tags(vol_ents)
        if not vol_tags:
            print(f"  [WARN] {body_name}: no volumes generated.")
            gmsh.finalize()
            return False

        # 단일 Physical Group (3D 볼륨)
        gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name=body_name)

        # 메쉬 옵션
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)   # Delaunay
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", elem_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", elem_size * 2.0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)  # MuJoCo flexcomp gmsh 요구
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Binary", 0)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.removeDuplicateNodes()

        # ── MSH 저장 (볼륨 테트라, flexcomp type="gmsh"용) ─────────────────────
        # MuJoCo flexcomp는 단일 노드 블록 요구 → Physical Group 제거 후 SaveAll=1
        gmsh.model.removePhysicalGroups()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(out_msh)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        # MSH 4.1 다중 블록 → 단일 블록 후처리 (MuJoCo flexcomp 호환)
        _fix_msh41_single_block(out_msh)
        print(f"  [OK] {body_name} MSH: {out_msh}")

        # ── STL 저장 (표면 삼각형, rigid geom용) ───────────────────────────────
        if out_stl is not None:
            # STL은 표면(dim=2) Physical Group이 필요
            surf_tags = []
            for vt in vol_tags:
                bnd = gmsh.model.getBoundary([(3, vt)], oriented=False)
                surf_tags.extend([t for d, t in bnd if d == 2])
            surf_tags = list(set(surf_tags))
            if surf_tags:
                gmsh.model.addPhysicalGroup(2, surf_tags, tag=2, name=body_name + "_surf")
            # Mesh.Binary=1: 바이너리 STL (MuJoCo 요구사항)
            gmsh.option.setNumber("Mesh.Binary", 1)
            gmsh.write(out_stl)
            gmsh.option.setNumber("Mesh.Binary", 0)
            print(f"  [OK] {body_name} STL (binary): {out_stl}")

        gmsh.finalize()
        return True

    except Exception as e:
        print(f"  [ERROR] {body_name}: {e}")
        import traceback
        traceback.print_exc()
        gmsh.finalize()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────

class TVPackagingMeshGenerator:
    """
    TV 포장재 + 내용물 FE 메쉬 생성기.

    Parameters (모두 mm 단위)
    -------------------------
    boxWidth, boxHeight, boxDepth : float
        포장 상자 외부 치수. 기본값 2000 × 1400 × 250 mm.
    boxThick : float
        상자 벽 두께. 기본값 5 mm.
    boxCenter : tuple(3)
        상자 중심 좌표. 기본값 (0, 0, 0).
    dispWidth, dispHeight : float
        디스플레이/샤시 폭·높이. 기본값 boxWidth-100, boxHeight-100.
    dispDepth : float
        디스플레이 두께. 기본값 5 mm.
    chassisDepth : float
        샤시 두께. 기본값 40 mm.
    dispCohWidth : float
        점착층 폭 (창틀 두께). 기본값 20 mm.
    dispCohThick : float
        점착층 두께 (Z 방향). 기본값 2 mm.
    contentCenter : tuple(3)
        샤시/디스플레이/점착층 중심 좌표. 기본값 (0, 0, 0).
    orientation : str
        'Parcel' (전면=-Z) 또는 'LTL' (전면=+Z). 기본값 'Parcel'.
    cushionCutouts : list
        추가 쿠션 컷아웃. 각 항목: (cx, cy, cz, w, h, d).
    doSplit : bool
        True이면 제품 경계면 평면으로 쿠션 분할. 기본값 False.
    elementSize : float
        전역 테트라 요소 크기. 기본값 50 mm.
    outputDir : str
        출력 디렉토리. 기본값 'test_box_msh'.
    """

    def __init__(
        self,
        boxWidth=2000.0,
        boxHeight=1400.0,
        boxDepth=250.0,
        boxThick=5.0,
        boxCenter=(0.0, 0.0, 0.0),
        dispWidth=None,
        dispHeight=None,
        dispDepth=5.0,
        chassisDepth=40.0,
        dispCohWidth=20.0,
        dispCohThick=2.0,
        contentCenter=(0.0, 0.0, 0.0),
        orientation='Parcel',
        cushionCutouts=None,
        doSplit=False,
        elementSize=50.0,
        outputDir='test_box_msh',
    ):
        self.bW  = float(boxWidth)
        self.bH  = float(boxHeight)
        self.bD  = float(boxDepth)
        self.bT  = float(boxThick)
        self.bCx, self.bCy, self.bCz = (float(v) for v in boxCenter)

        self.dW  = float(dispWidth)  if dispWidth  is not None else self.bW - 100.0
        self.dH  = float(dispHeight) if dispHeight is not None else self.bH - 100.0
        self.dD  = float(dispDepth)
        self.chD = float(chassisDepth)
        self.cohW = float(dispCohWidth)
        self.cohT = float(dispCohThick)

        self.cCx, self.cCy, self.cCz = (float(v) for v in contentCenter)
        self.orientation    = orientation
        self.cushionCutouts = list(cushionCutouts) if cushionCutouts else []
        self.doSplit        = doSplit
        self.elemSize       = float(elementSize)
        self.outputDir      = outputDir

    # ── Z 좌표 계산 ─────────────────────────────────────────────────────────

    def _content_z(self):
        """
        샤시·DispCoh·디스플레이의 Z 중심 좌표 반환.

        Parcel (전면=-Z): 샤시 → DispCoh → 디스플레이 순 (+Z 방향)
        LTL   (전면=+Z): 디스플레이 → DispCoh → 샤시 순 (+Z 방향)

        스택 전체 중심 = contentCenter.z
        """
        total = self.chD + self.cohT + self.dD
        half  = total / 2.0
        cz    = self.cCz

        if self.orientation == 'Parcel':
            z_ch  = cz - half + self.chD / 2.0
            z_coh = cz - half + self.chD + self.cohT / 2.0
            z_dp  = cz - half + self.chD + self.cohT + self.dD / 2.0
        else:  # LTL
            z_dp  = cz - half + self.dD / 2.0
            z_coh = cz - half + self.dD + self.cohT / 2.0
            z_ch  = cz - half + self.dD + self.cohT + self.chD / 2.0

        return z_ch, z_coh, z_dp

    # ── 바디별 빌더 (독립 세션용 클로저 반환) ───────────────────────────────

    def _make_box_builder(self):
        bCx, bCy, bCz = self.bCx, self.bCy, self.bCz
        bW, bH, bD, bT = self.bW, self.bH, self.bD, self.bT

        def build(occ):
            outer = _box_centered(occ, bCx, bCy, bCz, bW, bH, bD)
            inner = _box_centered(occ, bCx, bCy, bCz,
                                  bW - 2*bT, bH - 2*bT, bD - 2*bT)
            occ.synchronize()
            result, _ = occ.cut([(3, outer)], [(3, inner)],
                                removeObject=True, removeTool=True)
            occ.synchronize()
            return result
        return build

    def _make_chassis_builder(self):
        z_ch, _, _ = self._content_z()
        cCx, cCy = self.cCx, self.cCy
        dW, dH, chD = self.dW, self.dH, self.chD

        def build(occ):
            tag = _box_centered(occ, cCx, cCy, z_ch, dW, dH, chD)
            occ.synchronize()
            return [(3, tag)]
        return build

    def _make_display_builder(self):
        _, _, z_dp = self._content_z()
        cCx, cCy = self.cCx, self.cCy
        dW, dH, dD = self.dW, self.dH, self.dD

        def build(occ):
            tag = _box_centered(occ, cCx, cCy, z_dp, dW, dH, dD)
            occ.synchronize()
            return [(3, tag)]
        return build

    def _make_dispcoh_builder(self):
        _, z_coh, _ = self._content_z()
        cCx, cCy = self.cCx, self.cCy
        dW, dH, cohT, cohW = self.dW, self.dH, self.cohT, self.cohW

        def build(occ):
            outer = _box_centered(occ, cCx, cCy, z_coh, dW, dH, cohT)
            iw = dW - 2 * cohW
            ih = dH - 2 * cohW
            if iw <= 0 or ih <= 0:
                occ.synchronize()
                return [(3, outer)]
            inner = _box_centered(occ, cCx, cCy, z_coh, iw, ih, cohT)
            occ.synchronize()
            result, _ = occ.cut([(3, outer)], [(3, inner)],
                                removeObject=True, removeTool=True)
            occ.synchronize()
            return result
        return build

    def _make_cushion_builder(self):
        """
        쿠션 빌더:
          1. 박스 내부 크기 육면체
          2. [Chassis + Display + DispCoh] 복사본으로 Subtract
          3. cushionCutouts 추가 Subtract
          4. doSplit이면 얇은 박스 슬라이서로 Fragment 후 슬라이서 삭제
        """
        bCx, bCy, bCz = self.bCx, self.bCy, self.bCz
        bW, bH, bD, bT = self.bW, self.bH, self.bD, self.bT
        cCx, cCy = self.cCx, self.cCy
        dW, dH, cohT, cohW = self.dW, self.dH, self.cohT, self.cohW
        chD, dD = self.chD, self.dD
        z_ch, z_coh, z_dp = self._content_z()
        cutouts = list(self.cushionCutouts)
        do_split = self.doSplit

        def build(occ):
            # 쿠션 초기 볼륨
            cush = _box_centered(occ, bCx, bCy, bCz,
                                 bW - 2*bT, bH - 2*bT, bD - 2*bT)
            occ.synchronize()

            # 제품 바디 생성 (쿠션 세션 내에서 직접 생성 후 subtract)
            ch_tag = _box_centered(occ, cCx, cCy, z_ch, dW, dH, chD)
            dp_tag = _box_centered(occ, cCx, cCy, z_dp, dW, dH, dD)

            # DispCoh (picture-frame)
            coh_outer = _box_centered(occ, cCx, cCy, z_coh, dW, dH, cohT)
            iw = dW - 2 * cohW
            ih = dH - 2 * cohW
            if iw > 0 and ih > 0:
                coh_inner = _box_centered(occ, cCx, cCy, z_coh, iw, ih, cohT)
                occ.synchronize()
                coh_res, _ = occ.cut([(3, coh_outer)], [(3, coh_inner)],
                                     removeObject=True, removeTool=True)
                occ.synchronize()
                coh_ents = coh_res
            else:
                occ.synchronize()
                coh_ents = [(3, coh_outer)]

            # 모든 제품 바디를 하나의 cut 연산으로 제거
            tool_ents = [(3, ch_tag), (3, dp_tag)] + coh_ents
            result, _ = occ.cut([(3, cush)], tool_ents,
                                removeObject=True, removeTool=True)
            occ.synchronize()

            # 추가 컷아웃
            for (cx, cy, cz, w, h, d) in cutouts:
                cut_tag = _box_centered(occ, cx, cy, cz, w, h, d)
                occ.synchronize()
                result, _ = occ.cut(result, [(3, cut_tag)],
                                    removeObject=True, removeTool=True)
                occ.synchronize()

            # Split (옵션)
            if do_split:
                result = _split_with_thin_boxes(
                    occ, result,
                    cCx, cCy, bCx, bCy, bCz,
                    dW, dH, bW, bH, bD
                )

            return result
        return build

    # ── 메인 실행 ───────────────────────────────────────────────────────────

    def generate(self):
        """전체 파이프라인 실행."""
        os.makedirs(self.outputDir, exist_ok=True)

        z_ch, z_coh, z_dp = self._content_z()
        print("=" * 60)
        print("TV Packaging Mesh Generator")
        print(f"  Box     : {self.bW} x {self.bH} x {self.bD} mm  (thick={self.bT})")
        print(f"  Disp    : {self.dW} x {self.dH} x {self.dD} mm")
        print(f"  Chassis : {self.dW} x {self.dH} x {self.chD} mm")
        print(f"  CohWidth: {self.cohW} mm,  CohThick: {self.cohT} mm")
        print(f"  Orient  : {self.orientation}")
        print(f"  Z pos   : Chassis={z_ch:.1f}  DispCoh={z_coh:.1f}  Display={z_dp:.1f}")
        print(f"  ElemSize: {self.elemSize} mm")
        print(f"  doSplit : {self.doSplit}")
        print(f"  Output  : {os.path.abspath(self.outputDir)}/")
        print("=" * 60)

        # 바디별 빌더 정의
        bodies = [
            ("Box",     self._make_box_builder()),
            ("Chassis", self._make_chassis_builder()),
            ("Display", self._make_display_builder()),
            ("DispCoh", self._make_dispcoh_builder()),
            ("Cushion", self._make_cushion_builder()),
        ]

        saved_msh = {}   # flexcomp용 (볼륨 테트라, MSH 4.1)
        saved_stl = {}   # rigid geom용 (표면 삼각형, STL)
        print("\nGenerating meshes (one Gmsh session per body)...")
        for body_name, build_fn in bodies:
            msh_name = body_name.lower() + ".msh"
            stl_name = body_name.lower() + ".stl"
            msh_path = os.path.join(self.outputDir, msh_name)
            stl_path = os.path.join(self.outputDir, stl_name)
            ok = _mesh_session(body_name, build_fn, self.elemSize, msh_path, stl_path)
            if ok:
                saved_msh[body_name] = msh_name
                saved_stl[body_name] = stl_name

        print(f"\nMesh export complete: {len(saved_msh)}/5 bodies saved.")

        # MuJoCo XML 생성
        print("\nGenerating MuJoCo XML files...")
        self._generate_mujoco_xml_A(saved_msh, saved_stl)
        self._generate_mujoco_xml_B(saved_stl)

        print("\n" + "=" * 60)
        print("Done! Output:", os.path.abspath(self.outputDir))
        print("=" * 60)

    # ── MuJoCo XML ──────────────────────────────────────────────────────────

    def _body_color(self, name):
        colors = {
            "Box":     "0.6 0.4 0.2 0.4",
            "Cushion": "0.2 0.7 0.3 0.6",
            "Chassis": "0.5 0.5 0.5 1.0",
            "Display": "0.1 0.1 0.2 0.9",
            "DispCoh": "1.0 0.8 0.0 0.8",
        }
        return colors.get(name, "0.8 0.8 0.8 1.0")

    def _generate_mujoco_xml_A(self, saved_msh, saved_stl):
        """
        Version A: flexcomp (연성체) 기반 XML.
        Box, Cushion → flexcomp type="gmsh" (MSH 4.1 볼륨 메쉬, deformable)
        Chassis, Display, DispCoh → rigid mesh geom (STL 표면 메쉬)
        단위: mm → m via mesh scale="0.001 0.001 0.001" (MuJoCo 3.5.0+)
        """
        out_path = os.path.join(self.outputDir, "packaging_A_flexcomp.xml")
        L = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<mujoco model="tv_packaging_flexcomp">',
            '',
            '  <!-- Unit: mm -> m via mesh scale="0.001 0.001 0.001" (MuJoCo 3.5.0+) -->',
            '  <compiler meshdir="."/>',
            '',
            '  <option timestep="0.002" gravity="0 0 -9.81">',
            '    <flag contact="enable"/>',
            '  </option>',
            '',
            '  <visual>',
            '    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>',
            '  </visual>',
            '',
            '  <asset>',
        ]
        # rigid body용 STL mesh assets (Chassis, Display, DispCoh)
        for name in ["Chassis", "Display", "DispCoh"]:
            if name in saved_stl:
                fname = saved_stl[name]
                L.append(f'    <mesh name="{name.lower()}_mesh" file="{fname}" scale="0.001 0.001 0.001"/>')
        L += [
            '  </asset>',
            '',
            '  <worldbody>',
            '    <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 -1.5"',
            '          rgba="0.8 0.8 0.8 1" condim="3"/>',
            '',
        ]

        # flexcomp bodies (Box, Cushion) - MSH 볼륨 메쉬 (type="gmsh")
        for name in ["Box", "Cushion"]:
            if name not in saved_msh:
                continue
            fname = saved_msh[name]
            rgba  = self._body_color(name)
            L += [
                f'    <!-- {name}: deformable flexcomp (gmsh volume mesh) -->',
                f'    <body name="{name.lower()}" pos="0 0 0">',
                f'      <freejoint name="{name.lower()}_joint"/>',
                f'      <flexcomp name="{name.lower()}_flex"',
                f'               type="gmsh" file="{fname}" dim="3">',
                f'        <edge stiffness="5000" damping="10"/>',
                f'        <contact condim="3" selfcollide="none"/>',
                f'      </flexcomp>',
                f'    </body>',
                '',
            ]

        # rigid bodies (Chassis, Display, DispCoh) - STL 표면 메쉬
        for name in ["Chassis", "Display", "DispCoh"]:
            if name not in saved_stl:
                continue
            rgba = self._body_color(name)
            L += [
                f'    <!-- {name}: rigid body -->',
                f'    <body name="{name.lower()}" pos="0 0 0">',
                f'      <freejoint name="{name.lower()}_joint"/>',
                f'      <inertial pos="0 0 0" mass="1.0" diaginertia="0.1 0.1 0.1"/>',
                f'      <geom type="mesh" mesh="{name.lower()}_mesh"',
                f'            rgba="{rgba}" mass="1.0"/>',
                f'    </body>',
                '',
            ]

        L += ['  </worldbody>', '', '</mujoco>']
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(L))
        print(f"  [OK] {out_path}")

    def _generate_mujoco_xml_B(self, saved_stl):
        """
        Version B: 모든 바디 rigidbody + mesh geom (STL 표면 메쉬).
        단위: mm → m via mesh scale="0.001 0.001 0.001" (MuJoCo 3.5.0+)
        """
        out_path = os.path.join(self.outputDir, "packaging_B_rigid.xml")
        body_mass = {
            "Box":     "2.0",
            "Cushion": "1.5",
            "Chassis": "5.0",
            "Display": "3.0",
            "DispCoh": "0.1",
        }
        L = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<mujoco model="tv_packaging_rigid">',
            '',
            '  <!-- Unit: mm -> m via mesh scale="0.001 0.001 0.001" (MuJoCo 3.5.0+) -->',
            '  <compiler meshdir="."/>',
            '',
            '  <option timestep="0.002" gravity="0 0 -9.81">',
            '    <flag contact="enable"/>',
            '  </option>',
            '',
            '  <visual>',
            '    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>',
            '  </visual>',
            '',
            '  <asset>',
        ]
        for name, fname in saved_stl.items():
            L.append(f'    <mesh name="{name.lower()}_mesh" file="{fname}" scale="0.001 0.001 0.001"/>')
        L += [
            '  </asset>',
            '',
            '  <worldbody>',
            '    <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 -1.5"',
            '          rgba="0.8 0.8 0.8 1" condim="3"/>',
            '',
        ]
        for name, fname in saved_stl.items():
            rgba = self._body_color(name)
            mass = body_mass.get(name, "1.0")
            L += [
                f'    <!-- {name} -->',
                f'    <body name="{name.lower()}" pos="0 0 0">',
                f'      <freejoint name="{name.lower()}_joint"/>',
                f'      <inertial pos="0 0 0" mass="{mass}"',
                f'               diaginertia="0.1 0.1 0.1"/>',
                f'      <geom type="mesh" mesh="{name.lower()}_mesh"',
                f'            rgba="{rgba}" mass="{mass}"/>',
                f'    </body>',
                '',
            ]
        L += ['  </worldbody>', '', '</mujoco>']
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(L))
        print(f"  [OK] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Split Helper (모듈 레벨 함수 - 쿠션 빌더에서 호출)
# ─────────────────────────────────────────────────────────────────────────────

def _split_with_thin_boxes(occ, vol_ents, cCx, cCy, bCx, bCy, bCz,
                           dW, dH, bW, bH, bD):
    """
    얇은 박스(eps=0.01mm)를 슬라이서로 사용해 볼륨 분할.
    분할 기준: X = cCx ± dW/2,  Y = cCy ± dH/2
    Fragment 후 슬라이서 볼륨 삭제.
    """
    big = max(bW, bH, bD) * 2.0
    eps = 0.01

    slicer_tags = []
    for x_val in [cCx - dW/2, cCx + dW/2]:
        t = occ.addBox(x_val - eps/2, bCy - big/2, bCz - big/2, eps, big, big)
        slicer_tags.append(t)
    for y_val in [cCy - dH/2, cCy + dH/2]:
        t = occ.addBox(bCx - big/2, y_val - eps/2, bCz - big/2, big, eps, big)
        slicer_tags.append(t)

    occ.synchronize()
    slicer_ents = [(3, t) for t in slicer_tags]

    frag_result, frag_map = occ.fragment(
        vol_ents, slicer_ents, removeObject=True, removeTool=True
    )
    occ.synchronize()

    # 슬라이서 결과 볼륨 삭제
    n_obj = len(vol_ents)
    slicer_result = []
    for i in range(len(slicer_ents)):
        idx = n_obj + i
        if idx < len(frag_map):
            slicer_result.extend(frag_map[idx])

    slicer_vols = [(d, t) for d, t in slicer_result if d == 3]
    if slicer_vols:
        occ.remove(slicer_vols, recursive=True)
        occ.synchronize()

    slicer_vol_set = set((d, t) for d, t in slicer_vols)
    return [(d, t) for d, t in frag_result
            if d == 3 and (d, t) not in slicer_vol_set]


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = TVPackagingMeshGenerator(
        # 포장 상자 (외부 치수, mm)
        boxWidth=2000.0,
        boxHeight=1400.0,
        boxDepth=250.0,
        boxThick=5.0,
        boxCenter=(0.0, 0.0, 0.0),

        # 디스플레이 / 샤시 치수
        dispWidth=None,       # None → boxWidth  - 100 = 1900 mm
        dispHeight=None,      # None → boxHeight - 100 = 1300 mm
        dispDepth=5.0,
        chassisDepth=40.0,

        # 점착층
        dispCohWidth=20.0,
        dispCohThick=2.0,

        # 내용물 중심
        contentCenter=(0.0, 0.0, 0.0),

        # ISTA 6A 방향
        orientation='Parcel',

        # 추가 쿠션 컷아웃: [(cx, cy, cz, w, h, d), ...]
        cushionCutouts=[],

        # 볼륨 분할 (헥사 메쉬용, 기본 False)
        doSplit=False,

        # 요소 크기 (mm)
        elementSize=50.0,

        # 출력 디렉토리
        outputDir='test_box_msh',
    )
    gen.generate()
