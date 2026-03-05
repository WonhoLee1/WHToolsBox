# TV Packaging Gmsh Mesh Generator + MuJoCo XML

ISTA 6A 낙하 규정을 고려한 TV 포장재(Box, Cushion) 및 내용물(Chassis, Display, DispCoh)의 FE 메쉬를 Gmsh로 생성하고, MuJoCo XML로 시각화하는 스크립트를 구현합니다.

## Proposed Changes

---

### Main Script

#### [NEW] [tv_packaging_gmsh.py](file:///c:/Users/GOODMAN/WHToolsBox/tv_packaging_gmsh.py)

완전히 새로운 스크립트. 기존 [gmsh_packaging_model.py](file:///c:/Users/GOODMAN/WHToolsBox/gmsh_packaging_model.py)를 대체하는 완성형 구현.

**클래스 구조: `TVPackagingMeshGenerator`**

**입력 파라미터 (모두 mm 단위):**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `boxWidth`, `boxHeight`, `boxDepth` | 필수 | 박스 외부 치수 |
| `boxThick` | 5.0 | 박스 벽 두께 |
| `boxCenter` | (0,0,0) | 박스 중심 좌표 |
| `dispWidth` | boxWidth-100 | 디스플레이/샤시 폭 |
| `dispHeight` | boxHeight-100 | 디스플레이/샤시 높이 |
| `dispDepth` | 5.0 | 디스플레이 두께 |
| `chassisDepth` | 40.0 | 샤시 두께 |
| `dispCohWidth` | 20.0 | 점착층 폭 |
| `dispCohThick` | 2.0 | 점착층 두께 |
| `contentCenter` | (0,0,0) | 샤시/디스플레이/점착층 중심 |
| `orientation` | `'Parcel'` | `'Parcel'`(-Z front) 또는 `'LTL'`(+Z front) |
| `cushionCutouts` | `[]` | 추가 컷아웃 리스트: `[(cx,cy,cz, w,h,d), ...]` |
| `doSplit` | `False` | 바운딩박스 평면으로 볼륨 분할 여부 |
| `elementSize` | 50.0 | 전역 테트라 요소 크기 |
| `outputDir` | `'test_box_msh'` | 출력 디렉토리 |

**좌표계 및 배치 규칙:**
- **전면(Front)** = 디스플레이 화면 방향
- **Parcel**: 전면이 -Z → 샤시가 -Z쪽, 디스플레이가 +Z쪽
- **LTL**: 전면이 +Z → 디스플레이가 -Z쪽, 샤시가 +Z쪽
- Z축 기준 배치 순서 (Parcel): `[샤시 | DispCoh | 디스플레이]` (음→양 방향)

**바디별 지오메트리 생성 순서:**

1. **Box** (속이 빈 쉘)
   - 외부 박스 - 내부 박스 = 두께 `boxThick`의 중공 육면체
   - Boolean: `cut(outer_box, inner_box)`

2. **Chassis** (단순 육면체)
   - `dispWidth × dispHeight × chassisDepth`
   - Z 위치: orientation에 따라 결정

3. **Display** (단순 육면체)
   - `dispWidth × dispHeight × dispDepth`
   - Z 위치: 샤시 반대편

4. **DispCoh** (창틀 모양 점착층)
   - 외부 직사각형 - 내부 직사각형 = Picture Frame
   - 외부: `dispWidth × dispHeight × dispCohThick`
   - 내부: [(dispWidth-2*dispCohWidth) × (dispHeight-2*dispCohWidth) × dispCohThick](file:///c:/Users/GOODMAN/WHToolsBox/box_mesh_generator.py#8-62)
   - Boolean: `cut(outer_coh, inner_coh)`
   - Z 위치: 샤시와 디스플레이 사이

5. **Cushion** (복잡한 불린 연산)
   - 초기 볼륨: 박스 내부 크기 (`boxWidth-2*boxThick` 등)
   - Subtract 1: `[Chassis ∪ Display ∪ DispCoh]` 의 복사본을 빼냄
   - Subtract 2: `cushionCutouts` 리스트의 각 육면체를 순서대로 빼냄

6. **Split (옵션, `doSplit=True`)**
   - 샤시/디스플레이 경계면을 연장한 평면들로 쿠션을 분할
   - `occ.fragment()` 사용

**메쉬 생성:**
- 각 바디에 대해 `Mesh.Algorithm3D = 4` (Delaunay) 또는 `10` (HXT)으로 테트라 생성
- `elementSize`로 전역 크기 제어
- Box는 내부가 빈 쉘이므로 3D 볼륨 메쉬 생성 후 내부 요소 제거 (또는 표면 메쉬만 사용)

**파일 출력 (`test_box_msh/` 디렉토리):**
- `box.msh`
- `cushion.msh`
- `chassis.msh`
- `display.msh`
- `dispcoh.msh`

---

### MuJoCo XML 생성

#### [NEW] [tv_packaging_gmsh.py](file:///c:/Users/GOODMAN/WHToolsBox/tv_packaging_gmsh.py) (동일 파일 내 함수)

**`generate_mujoco_xml_A(output_dir)` — flexcomp 버전**

```xml
<mujoco model="tv_packaging">
  <compiler meshdir="." />
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="box">
      <flexcomp name="box" file="box.msh" type="mesh" dim="3"
                scale="0.001 0.001 0.001">  <!-- mm→m 변환 -->
        <edge stiffness="1000"/>
      </flexcomp>
    </body>
    <!-- cushion, chassis, display, dispcoh 동일 패턴 -->
  </worldbody>
</mujoco>
```

> [!IMPORTANT]
> MuJoCo는 m 단위 선호. `scale="0.001 0.001 0.001"` 속성으로 mm→m 변환.
> flexcomp의 `scale` 속성이 지원되지 않는 경우 `<compiler meshdir="." scale="0.001"/>` 사용.

**`generate_mujoco_xml_B(output_dir)` — rigidbody 버전**

```xml
<mujoco model="tv_packaging">
  <compiler meshdir="." scale="0.001"/>  <!-- mm→m 변환 -->
  <asset>
    <mesh name="box_mesh" file="box.msh"/>
    <!-- 나머지 mesh 선언 -->
  </asset>
  <worldbody>
    <body name="box" pos="0 0 0">
      <geom type="mesh" mesh="box_mesh" mass="1.0"/>
    </body>
    <!-- cushion, chassis, display, dispcoh 동일 패턴 -->
  </worldbody>
</mujoco>
```

---

## Verification Plan

### Automated Tests

```powershell
# vdmc conda 환경에서 실행
conda run -n vdmc python tv_packaging_gmsh.py
```

성공 조건:
- `test_box_msh/` 디렉토리 생성
- `box.msh`, `cushion.msh`, `chassis.msh`, `display.msh`, `dispcoh.msh` 5개 파일 생성
- `packaging_A_flexcomp.xml`, `packaging_B_rigid.xml` 2개 파일 생성

### Manual Verification

```powershell
# MuJoCo viewer로 XML 확인 (B 버전 먼저 - 더 안정적)
conda run -n vdmc python -c "import mujoco; import mujoco.viewer; m=mujoco.MjModel.from_xml_path('test_box_msh/packaging_B_rigid.xml'); mujoco.viewer.launch(m)"
```

- 5개 바디가 화면에 올바른 위치에 표시되는지 확인
- 박스 내부에 쿠션, 쿠션 내부에 샤시/디스플레이/점착층이 배치되는지 확인
