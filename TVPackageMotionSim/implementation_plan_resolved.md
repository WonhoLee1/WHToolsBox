# 구현 계획 및 내역 종합: 이산형 박스 모델 생성기 (Discrete Box Model)

## 목표
주요 목표는 복잡한 다중 레이어 포장 박스를 개별적으로 분할된 블록 요소들로 구성하는 MuJoCo XML 파일을 생성하는 파이썬 스크립트(`test_box_mujoco/run_discrete_builder.py`)를 구현하는 것입니다. 이 블록들은 굽힘(Bending) 및 비틀림(Twisting)과 같은 실제 변형 거동을 낙하 테스트(예: ISTA 6A) 시뮬레이션에서 모사하기 위해 연성 조인트(Soft Welds)로 서로 연결됩니다.

모델은 다음 5가지 주요 구성 요소로 이루어집니다:
1. [BPaperBox](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#234-248): 6개의 면으로 구성된 외부 쉘(Shell)로, 각각 그리드로 분할되고 특정 두께(`_thick`)를 가집니다.
2. [BCushion](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#249-273): [BPaperBox](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#234-248) 내부 공간에서 형상된 직각육면체를 분할하는 구조. `BCushion_cutter` (dict 변수)에 기록하는 각 key값에 해당하는 value인 `[center_x, center_y, center_z, width, height, depth]`를 가지는 육면체 형상을 제거(Cavity 생성)한 나머지 영역이 됩니다. 기본값은 `{'center': [0, 0, 0, BCushion_width*0.5, BCushion_height*0.5, BCushion_depth*0.5]}` 처럼 구성됩니다.
3. [BOpenCellCohesive](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#274-287): 액자형(Picture-frame-like) 구조 레이어.
4. [BOpenCell](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#288-290): TV 전면에 위치하는 솔리드 블록 레이어.
5. [BChassis](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#291-293): 후면 섀시를 나타내는 솔리드 블록 레이어.

*참고:* [BOpenCell](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#288-290), [BOpenCellCohesive](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#274-287), [BChassis](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#291-293)는 `AssySet` 이라는 하나의 조합을 형성합니다.

## 제안하는 구현 방향

### `test_box_mujoco/run_discrete_builder.py` 핵심 로직
이 스크립트는 형상 생성을 관리하는 클래스와 함수를 포함하는 핵심 빌더 역할을 합니다.

**기본 데이터 구조 및 설정:**
- 각 Body 유형에 대한 기본 설정 딕셔너리를 정의합니다. (포함 항목: `width`, `height`, `depth`, `mass`, `div`(방향별 분할 갯수), `thick`(PaperBox용), `ithick`(OpenCellCohesive용)).
- 글로벌 설정: `AssySet_Pos` (내용물 어셈블리의 오프셋 위치), `BCushion_inner_gap` (쿠션 내부의 여유 공간).

**형상 생성 로직:**
1. **좌표계 및 분할 (Coordinate System & Subdivisions)**: 
   - 모든 형상은 초기에 회전되지 않은 로컬 기본 상태(width는 +X, height는 +Y, depth는 +Z 방향)로 정의됩니다.
   - `generate_grid_nodes` 함수가 세부 블록의 경계 좌표를 계산합니다.
   
2. **Body별 계층 구조 (Assembly Hierarchy)**:
   - 최상위 컨테이너: `BPackagingBox` (이 Body를 기준으로 ISTA 회전 적용)
   - `BPackagingBox` 하위: [BPaperBox](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#234-248), [BCushion](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#249-273), `AssySet`
   - `AssySet` 하위: [BOpenCell](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#288-290), [BOpenCellCohesive](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#274-287), [BChassis](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#291-293)
   - 위와 같은 부모-자식 관계의 `<body name="...">` 태그 구조를 생성하여 전역 좌표 변환을 단순화합니다.

3. **엄격한 격자 분할 로직 (Strict Grid Subdivision)**:
   - [BOpenCellCohesive](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#274-287)의 `ithick` (테이프 폭), [BCushion](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#249-273)의 내부 공동(Cavity) 및 갭(`gap`) 경계선들은 분할의 **필수 기준선**이 됩니다.
   - 즉, 단순히 전체 길이를 `div` 개수로 균등 분할하는 것이 아니라, 반드시 잘려야 하는 절대적 경계면 좌표 배열을 먼저 생성한 뒤, 그 사이의 남은 공간들을 기본 분할 조건(`div`)에 맞추어 추가 분할하는 방식으로 파라메트릭하게 3D 그리드(Grid)를 생성합니다.
   
4. **Body별 모델 구현 방안**:
   - `build_BPaperBox()`: 6개의 얇은 벽면 생성을 담당합니다. 박스가 겹치지 않고 완전히 닫힌 껍질 형태를 이루도록 모서리 부분의 오버랩을 적절히 계산하여 분할 블록을 생성합니다.
   - `build_AssySet()`: 핵심 부품 그룹([BOpenCell](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#288-290), [BOpenCellCohesive](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#274-287), [BChassis](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#291-293))을 생성합니다. 특히 [BOpenCellCohesive](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#274-287)는 `_ithick`에 기반하여 중앙을 비우는(Subtraction) 로직을 적용합니다.
   - `build_BCushion()`: [BPaperBox](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#234-248) 내부 치수를 바탕으로 전체 부피를 계산한 뒤, `BCushion_cutter` (dict 변수)를 순회하며 각 key의 value(`[cx, cy, cz, w, h, d]`)에 해당하는 절단(Cutter) 육면체 영역 내부에 블록이 포함되는지 확인하고, 해당 영역의 블록을 제거(생성 건너뜀)하는 방식으로 공동을 파냅니다.

4. **Soft Weld 적용 및 물성치 (Weld Application & Material Properties)**:
   - `apply_internal_welds()`: 동일 부품 내 인접 블록 연결
   - `apply_inter_component_welds()`: 부품 간 구조적 결합 처리
   - **기본 물성치 할당**: 부품별 재질(Material)에 기반하여 기본 `solref` / `solimp` 값을 정의합니다.
     - [BPaperBox](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#234-248) (종이): 비교적 높은 강성 지정
     - [BCushion](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#249-273) (EPS Foam): 발포 스티로폼에 맞는 충격 흡수(Soft) 강성 지정
     - [BOpenCellCohesive](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#274-287) (접착테이프): 두 파트를 강하게 잡아주면서도 약간의 전단(Shear) 변위를 허용하는 접착 강성 지정

5. **질량 분배 (Volumetric Mass Distribution)**:
   - 각 부품에 선언된 총 질량(`mass`)은 단순히 블록 개수(`N`)로 평등하게 나누지 않습니다.
   - 각 서브 블록(직육면체)의 개별 체적(Volume)을 계산한 뒤, 전체 체적 대비 개별 체적 비율에 비례하여 서브 블록([geom](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#100-133))의 `mass`를 할당합니다.

6. **ISTA 6A 낙하 조건 변환 (Drop Configuration)**:
   - 기본적으로 `AssySet` (OpenCell 정면)은 +Z 방향을 향합니다.
   - `PARCEL` 모드: 정면이 -Z 방향(바닥)을 향하도록 시스템을 회전시킵니다.
   - `LTL` 모드: 정면이 +Z 방향(천장)을 향하도록 유지하거나 회전시킵니다.
   - 박스의 가장 아랫부분이 바닥 기준 `Z = drop_height`에 오도록 쿼터니언(Quaternion) 회전과 평행이동(Translation)을 전역 어셈블리에 적용합니다.

## 6. Configuration Dictionary 및 확장 낙하(ISTA) 방향 제어
[generate_tv_package.py](file:///c:/Users/GOODMAN/WHToolsBox/generate_tv_package.py)의 [create_model](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/generate_tv_package.py#297-488) 함수 내부 하드코딩된 변수들을 외부 `config` 딕셔너리로 묶어, 터미널 실행 시 혹은 메인 스크립트에서 유연하게 시나리오를 주입하도록 고도화합니다.
또한 낙하 자세(Drop Orientation)를 상세히 제어하기 위해 방향 문자열을 받아 쿼터니언 회전/평행이동으로 변환합니다.

### 6.1 Drop Configuration 매핑 규격
방향 코드는 **F(Front), R(Rear), L(Left), R(Right), T(Top), B(Bottom)** 의 조합으로 구성됩니다.
- **면(Face) 낙하**: `F`, `R`, `L`, `T`, `B` -> 각 면이 바닥(-Z)을 향하도록 90도/180도 회전
- **변(Edge) 낙하**: `F-T`, `F-L` 등 2개 문자의 조합 -> 해당 변이 바닥과 일직선상을 향하도록 45도 회전
- **모서리/꼭지점(Corner) 낙하**: `F-T-L`, `F-B-R`, `R-B-L` 등 3개 문자의 조합 -> 무게 중심 기준으로 타겟 꼭지점이 중력선상에 놓이도록 회전축(Cross Product) 연산 후 동적 회전
- **낙하 높이 기준**: 회전 완료 후 바운딩 박스의 최하점(Lowest Z)을 측정하여, 이 지점이 `Z = drop_height`에 오도록 병진(Translation) 변환합니다.

## 7. 부품별 개별 강성 평가 모듈 (Body Stiffness Test Mode)
개단품(Part)의 soft weld 강성 및 거동을 정량적으로 평가하기 위해 독립된 시뮬레이션 환경(새로운 Python 클래스/파일)을 구축합니다. 터미널에서 "Body Stiffness Test Mode"와 타겟 Body를 선택하면, 각 부품은 단독 XML로 분리 생성되고 아래와 같은 가상 하중 시나리오를 겪습니다.

### 7.1 평가 항목 (동적 액츄에이터 활용)
| 타겟 Body | 기능 | 시험 작동 방식 (시간-작용 리스트 입력 제어) |
| :--- | :--- | :--- |
| **BCushion** | Compression | 전후면에 가상의 평면(Plane) 액츄에이터를 덧대어 압축 후 유지/복구 (시간-변위/하중 그래프) |
| | Twist | 양 측면을 잡고 상호 반대 방향으로 비틀기 (시간-회전각 리스트 기반) |
| | Bending | 양 측면을 고정하고 중앙을 실린더 형상의 액츄에이터로 누름 |
| **BOpenCell** | Twist/Bending | 쿠션과 동일한 메커니즘을 얇고 넓은 면적에 적용 |
| **BChassis** | Twist/Bending | 패널류의 굽힘 강성(Bending Rigidity) 검증 적용 |
| **BOpenCellCohesive** | Twist/Bending | 테이프(접착부) 자체의 인장/전단 변형 강성 모델링 검증 |

동역학 시뮬레이션 직후 `matplotlib`를 이용해 가해진 힘(Force) vs 변위/각도(Displacement) 관계를 자동 그래프화하여 터미널에 띄웁니다.

## 8. 질량 중심(CoG) 및 관성 모멘트(MoI) 최적화 제어
제품의 질량 분포 조건을 동적으로 모사하기 위한 계산 및 추가 질량 배치 로직을 개발합니다.
1. **현재 상태 출력**: 프로그램 실행 시 조립체의 전역 CoG 및 MoI가 터미널에 계산되어 출력됩니다. 낙하 방향(LTL vs PARCEL)에 따른 전면 방향 변화와 최종 CoG 변화 원리도 설명 문구로 곁들입니다.
2. **목표 질량 조준(Target Tuning)**: 사용자가 목표 CoG 오프셋(예: 상대적 `[0, +30, 0]` 증가)이나 목표 MoI를 명령어로 주입하면, 해당 값을 충족시키기 위해 보이지 않는 Dummy Mass(다수 배치 가능)를 내부 특정 좌표에 자동으로 부착하는 최적화 솔버를 구동합니다. 이 솔버는 형상의 XY 전면 기준에서 작동한 뒤, 이후 낙하 방향 회전 시 덩어리째 함께 회전하도록 설계됩니다.

## 9. 쿠션 강성(solref/solimp) 보정 최적화 모듈
모서리/변 낙하 시뮬레이션 결과(가상 센서 측정값)와 실제 드랍 시험(가속도 등 물리 시험) 결과를 최소 오차로 맞추기 위해 쿠션 접촉상수인 `solref` 및 `solimp`를 역산하는 파라미터 최적화 모듈을 신설합니다.
### 9.1 Data-Driven Tuning 
사용자가 두 가지 데이터 중 하나 이상을 입력하면 최적화가 가동됩니다.
- 모서리 낙하 시 덩어리 전체의 무게중심 Z방향 "시간-변위" 데이터
- 모서리 낙하 전/후, 임팩트 주변부 블록 서브그리드의 "최대 변형 변위(영구/탄성 변형률)" 데이터 

이 데이터를 바탕으로 MuJoCo를 루프백하면서(가벼운 Gradient/Nelder-Mead 최적화), 에러율(Loss)이 최소가 되는 Floor Contact Property 값과 쿠션의 Weld Property를 역추적합니다.




## 📌 목표 및 현재 진행 상황 (Status)
복잡한 다중 레이어 포장 박스를 개별적으로 분할된 블록(그리드) 요소들로 구성하는 MuJoCo XML 파일을 생성하는 파이썬 스크립트(`test_box_mujoco/run_discrete_builder.py`)와 시뮬레이션 인프라를 구축했습니다. 현재 Section 1~8번까지 기능 구현 및 검증이 완료되었으며, 마지막 단계인 Section 9 구성 중입니다.

## ✅ 기구현 (Implemented Features)

### 1~5. Discrete Box Builder (`run_discrete_builder.py`)
- **계층적 컴포넌트 구조화**: `BPackagingBox`를 최상위 회전축으로 하여 5가지 내부 컴포넌트 생성.
  - `BPaperBox`, `BCushion`(내부 Cavity 절단(Subtraction) 로직 적용), `BOpenCellCohesive`, `BOpenCell`, `BChassis`
- **엄격한 격자 분할 로직**: 부품 모서리와 겹치지 않게 정확한 두께(`thick`, `ithick`)를 보존하며 균등 분할.
- **체적 기반 질량 분배**: 각 블록의 체적 비율에 맞춰 사용자 입력 `mass`를 자동 분배.
- **물리 모델링 (Soft Weld)**: 개별 단위 블록의 꼭지점 사이를 잇는 1방향 배열의 Spring constraints(`weld`)를 자동 생성하여 굽힘/비틀림 강성 모사. 부품별 재질 속성에 따른 `solref`, `solimp` 할당 적용.

### 6. Config 파일 분리 및 `drop_mode` 세분화
- **`get_default_config()`**: 복잡한 인자 전달 대신, 사용자가 꼭 필요한 인자만 던져도 연관 치수를 동적 계산하여 완벽한 `config` 딕셔너리로 통합하는 시스템 구축.
- **확장 낙하 방향 제어**: `[F, B, R, L, T]` 조합형 문자열 기반(`"F-T"`, `"R-B-L"`) 벡터 타겟팅 파서로 어떠한 면, 변, 꼭지점도 바닥 방향의 수직 타겟팅이 되도록 회전 구현.

### 7. 구조 해석 지표 추적 (`run_drop_simulation.py`)
- **Headless 동역학 통합:** 뷰어 없이 Python 스크립트 기반 고속 연산 가능.
- **구조 지표(Metrics) 추출:** 컴포넌트 개단품별로 Bending(곡률 왜곡), Twist(비틀림 위상차) 등 변형 에너지의 최대 발생 인덱스(Hotspot)와 수치를 추적/로깅 및 파이썬 Plot 시각화.

### 8. 부품별 개별 강성 평가 (Body Stiffness Test Mode)
- **`run_stiffness_test.py`**: 완성된 부품 중 특정 파트 하나를 허공에 띄워 가상 클램프(Clamp)와 램(Ram) 엑츄에이터를 붙여 Force-Displacement(하중-변위 곡선)를 측정.
- BChassis(Bending), BOpenCell(Twist), BCushion(Compression) 단위 가상 테스트 시스템 가동. (물리적 접촉(Contact) 저항 지원 및 초기화 충격 방지형 바운딩 연산 적용 완료)

### 8.1 전체 어셈블리 관성 텐서(Inertia) 연산
- `calculate_inertia()`: 수백 개의 이산화된 블록들의 국소(Local) 질량/중심점 데이터를 재귀적으로 병합, 평행축 정리(Parallel Axis Theorem)를 통해 모델 생성 즉시 최종 조립체의 전역 CoG(x,y,z 위치)와 MoI(Ixx, Iyy, Izz) 터미널 출력.

---

## 🏃 진행 예정 목표 (Upcoming Task)

### 9. [✅ 완료] 쿠션 강성(solref/solimp) 보정 최적화 모듈 (Data-Driven Tuning)
실제 드랍 시험 결과(시간에 따른 Z방향 변위 데이터)와 현재 이산화 시뮬레이션 결과를 합일화하기 위해, 쿠션의 접촉/댐핑 파라미터인 `solref` 및 `solimp`를 역산하는 데이터기반(Data-Driven) 최적화 루프백 모듈이 구현 완료되었습니다.

*   `run_cushion_optimization.py`를 신설하여 `scipy.optimize.minimize` (Nelder-Mead 기법) 최적화 솔버 연동.
*   최적화 타겟인 `solref` / `solimp` 파라미터들을 독립 변수(`cush_solref_damp`, `cush_solimp_width` 등)로 분할하여 사용자가 튜닝 및 제어하기 쉽도록 재구성 반영 완료.

#### 9.1 Data-Driven Tuning Target Metrics
사용자가 두 가지 물리 시험 데이터 중 하나 이상을 입력하면 최적화 Loss 평가가 가동됩니다.
1. **Z축 궤적 기반:** 모서리 낙하 시 전체 조립체 혹은 무게중심의 Z방향 "시간-변위" 데이터 곡선 유사도 (MSE 매칭)
2. **최대 변형 방어:** 모서리 낙하 전/후, 임팩트 주변부 블록 서브그리드의 "최대 변형각 / 영구 탄성 변형률" 데이터 (단일 지점 최대 변위값 최적화)

### 10. [✅ 완료] 부품별 개별 강성(Soft Weld) 최적화 모듈 (Component Stiffness Tuning)
8번 항목(Body Stiffness Test)에서 측정된 **단품 부품(BChassis, BOpenCell 등)의 특정 하중-변위(Force-Displacement) 목표 곡선(Target Curve)**을 기반으로, 내부 블록들의 접합부(Soft Weld) 제약 조건인 `solref` / `solimp` 파라미터를 최적화합니다.

*   `run_stiffness_optimization.py` 신설.
*   `run_stiffness_test.py`의 구조 평가 로직을 캡슐화하여, 대상 부품과 실험 Force-Displacement 데이터를 입력하면 MSE 비율 기반 Loss 오차를 반환하는 목적 함수 구현. (Multi-test Joint Loss 지원)
*   사용자 데이터에 맞춰 평판 이론의 $K \propto 1/\tau^2$ 비례법칙을 활용해 **최적의 초기 파라미터를 역산하는 Heuristic 추정 로직** 추가.
*   7차원 비선형 곡선 변수(`solimp` width, midpoint, power 포함)를 최적화할 수 있도록 파라미터 영역 확장 구현 및 고차원에 적합한 `trust-constr`, `dual_annealing` 글로벌 솔버 옵션 제공.

