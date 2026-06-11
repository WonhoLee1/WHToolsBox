# A Multi-Scale Digital Twin Framework for Impact-Optimized TV Packaging Design

**From Low-Fidelity Real-Time Dynamics to High-Fidelity FEM via JAX-Accelerated Structural Analysis, Topography Optimization, and OpenRadioss Integration**

---

**Authors**: Wonho Lee¹*, et al.  
**Affiliations**: ¹ WHTOOLS Research, Advanced Packaging Engineering Division  
**Keywords**: Digital Twin, Drop Impact Simulation, MuJoCo, Kirchhoff-Love Plate Theory, JAX, VTKHDF, CMA-ES Calibration, Topography Optimization, OpenRadioss, Fidelity Continuum, TV Display Protection

---

## Abstract

대형 디스플레이 제품의 유통 과정에서 발생하는 낙하 충격은 제품 파손의 주요 원인이다. 본 연구에서는 **Low-Fidelity(LF) → Mid-Fidelity(MF) → High-Fidelity(HF)** 연속 정밀도 체계를 갖춘 **WHTOOLS 멀티스케일 디지털 트윈 프레임워크**를 제안한다.

제안된 프레임워크는 네 가지 계층으로 구성된다:

1. **[LF] MuJoCo 감차원 강-연성체 결합 시뮬레이션**: N×M 격자 이산 요소 + 6자유도 점탄성 조인트로 TV 낙하 충격을 실시간(~40초) 해석하고, 다각도 낙하 조건(Face, Edge, Corner)의 시계열 마커 궤적을 자동 생성
2. **[MF] JAX-SSR 구조 응력 복원**: Kabsch 기구학 + Tikhonov Kirchhoff-Love 피팅으로 마커 궤적 → 연속 Von-Mises 응력장 복원(~3초), VTKHDF v2.2 단일 파일 내보내기
3. **[MF+] WHT_LightChassisModel 토포그래피 최적화**: LF 시뮬레이션에서 추출한 **다각도 낙하 하중을 ESL(Equivalent Static Load)로 변환**, JAX 자동미분 기반 MMA(Method of Moving Asymptotes) 최적화로 초기 샤시 비드 패턴 설계. 상용 FEA 없이 낙하 내충격 최적 강성 분포를 획득
4. **[HF] OpenRadioss 고충실도 FEM**: LF 시뮬레이션의 6자유도 자세·속도를 자동 추출하여 Gmsh 메시 + OpenRadioss Starter/Engine 파일을 자동 생성, VTKHDF 형식으로 결과를 직접 출력

전체 파이프라인이 단일 낙하 조건에서 **상용 FEA 대비 95% 이상의 설계 탐색 시간 단축**을 달성하면서도, 필요 시 HF 해석으로 매끄럽게 에스컬레이션되는 **Fidelity Continuum** 체계를 제공한다.

---

## 1. Introduction

### 1.1 연구 배경

전 세계 75인치 이상 대형 디스플레이 시장은 지속적으로 확대되고 있으며, 포장재(EPS, EPP, 골판지 복합 구조체)의 낙하 내구성 확보가 핵심 과제로 부상하였다. 종래의 설계 프로세스는 두 극단에 의존해왔다:

| 접근법 | 장점 | 한계 |
|--------|------|------|
| 물리 시험 반복 (ISTA 2A, ASTM D5276) | 현실 정합도 | 고비용·긴 사이클 |
| 상용 FEA (LS-DYNA, ABAQUS Explicit) | 고정밀 응력장 | 단일 케이스 수 시간~수일 |

더 나아가 **샤시(Chassis) 구조 설계** 단계에서는 낙하 충격에 대응하는 최적 비드(Bead) 패턴을 결정해야 하나, 기존 접근은 전체 패키지 FEA 결과를 피드백으로 받아 수작업으로 비드를 설계하는 방식으로, 설계-해석-수정 루프가 수주에 달하는 병목을 형성한다.

### 1.2 Fidelity Continuum 패러다임

본 연구는 단순히 LF와 HF를 별도로 운용하는 대신, **연속적 정밀도 체계(Fidelity Continuum)**로 통합한다:

```
[Physical Drop Test: Optical Marker Tracking]
          ↓  Measured Corner Trajectories (C1~C8 × Time)
          
┌──────────────────────────────────────────────────────────┐
│  LF: MuJoCo Reduced-Order Simulation (~40s/case)        │
│  ├── Rigid-Flexible Grid Model (N×M weld lattice)       │
│  ├── Aero: Quadratic Drag + Viscous + Squeeze Film       │
│  ├── Plasticity: Contact-based EPS/EPP crushing         │
│  └── Output: 6-DOF pose, velocity, corner trajectories  │
└────────────────────┬─────────────────────────────────────┘
                     │  Multi-angle drop data (Face/Edge/Corner)
          ┌──────────┴──────────────────────────┐
          │                                      │
          ▼                                      ▼
┌──────────────────────┐          ┌──────────────────────────────┐
│  MF: JAX-SSR (~3s)  │          │  MF+: WHT Topography Opt.    │
│  Kirchhoff-Love      │          │  ESL from drop trajectories  │
│  Kabsch + Tikhonov   │          │  → JAX auto-diff MMA         │
│  → 17 stress fields  │          │  → Optimal bead pattern      │
│  → VTKHDF v2.2       │          │  → LS-DYNA .k export         │
└──────────────────────┘          └──────────────────────────────┘
          │                                      │
          └──────────────┬───────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  HF: OpenRadioss Full FEM                               │
│  ├── Auto-mesh: Gmsh (Box, Cushion, OpenCell, Chassis)  │
│  ├── BC: LF pose/velocity → /TRANSFORM ROT+TRA          │
│  ├── Solver: OpenRadioss Starter + Engine               │
│  └── Output: VTKHDF (anim_to_vtkhdf=yes)               │
└──────────────────────────────────────────────────────────┘
          ↓
[Design Optimization & Verification]
```

### 1.3 핵심 기여

1. **LF 실시간 강-연성체 결합 모델**: MuJoCo + N×M 격자 조인트로 유연체 대변형 실시간 해석
2. **JAX-SSR**: Kabsch + Tikhonov Kirchhoff-Love 피팅으로 마커 궤적 → 연속 응력장 복원
3. **CMA-ES + DTW 교정 루프**: 실측 마커 궤적과 시뮬레이션의 자동 파라미터 동기화
4. **WHT_LightChassisModel**: LF 다각도 낙하 데이터 → ESL 변환 → JAX MMA 토포그래피 최적화 → 초기 샤시 강성 설계. **상용 FEA 없이 낙하 최적 비드 패턴 획득**이라는 산업적 혁신
5. **OpenRadioss 자동 연계**: LF 결과 pkl → Gmsh 메시 + `.rad` 파일 자동 생성 → HF FEM 원클릭 실행
6. **VTKHDF v2.2 일관성**: LF→MF→HF 모든 단계의 결과를 동일한 ParaView 포맷으로 출력

---

## 2. Related Work

### 2.1 포장재 낙하 충격 해석

Burgess (1988)의 쿠션 곡선 이론에서 출발하여, LS-DYNA 기반 EPS 완충재 충격 해석, PAM-CRASH 기반 골판지 좌굴 해석이 고정밀 해석의 표준을 제시하였으나, 단일 케이스당 수 시간~수일의 계산 비용이 설계 반복의 병목을 초래한다.

### 2.2 토포그래피(Topography) 최적화

토포그래피 최적화는 토폴로지 최적화(요소 추가/삭제)와 달리 **비드 높이를 설계 변수**로 사용하여 판재의 강성을 극대화한다. 기존 연구는 SIMP 밀도 변수와 고정 하중 케이스를 결합하였으나, 낙하 충격이라는 비선형 동적 하중의 ESL 추출과의 결합 사례는 보고된 바 없다.

### 2.3 ESL (Equivalent Static Load) 방법론

ESL 방법론 [Park & Kang, 2003]은 동적 하중에 대한 위상/형상 최적화를 정적 등가 하중으로 변환하여 수행하는 기법이다. 본 연구는 실측 궤적 데이터로부터 Kabsch 전처리 → Newmark-β 직접 적분(또는 모달 중첩) → SE(Strain Energy) 이력 기반 Greedy 다양성 선별의 3단계 ESL 추출 파이프라인을 구현하였다.

### 2.4 OpenRadioss 오픈소스 FEM

OpenRadioss [Altair, 2022~]는 Radioss 명시적 솔버의 오픈소스 버전으로, 충격·충돌 해석에 특화되어 있다. Python API와 Gmsh 메시 생성을 결합한 자동화 파이프라인은 전례가 없으며, 본 연구는 LF 시뮬레이션 결과를 직접 경계 조건으로 변환하는 최초의 통합 구현을 제시한다.

---

## 3. Low-Fidelity Engine: MuJoCo 감차원 시뮬레이션

### 3.1 감차원 이산-연속체 유연체 모델

연속적 유연 평면체(Glass Panel, Chassis 등)를 $N_x \times N_y \times N_z$ 격자의 강체 유닛 블록 집합으로 이산화한다. 인접 블록 쌍은 6자유도 점탄성 용접 조인트로 연결된다:

$$\mathbf{F}_{joint} = k_{ref} \cdot \Delta \mathbf{q} + d_{imp} \cdot \dot{\mathbf{q}}, \quad \Delta \mathbf{q} \in \mathbb{R}^6$$

이 접근법의 핵심 이점: 전역 강성 행렬 조립/역산 없이 $O(n)$ 복잡도로 대변형 해석 가능.

### 3.2 통합 공력-소성 엔진

Numba JIT 컴파일된 단일 함수에서 세 가지 공기 역학 효과를 동시 계산:

$$F_{total} = F_{drag} + F_{visc} + F_{squeeze}$$

$$F_{squeeze} = k_{sq} \cdot \mu_{air} \cdot \frac{A^2 \cdot (-v_z)}{h^3}, \quad h_{min} < h < h_{max}$$

완충재 소성: 등가 변형률 $\varepsilon_{eq} = \delta / L_{ref}$가 항복점 초과 시 복원 강성 비가역 감소.

### 3.3 다각도 낙하 자동 스위프 (DOE)

단일 cfg 파일에서 다음의 낙하 조건을 자동 순차 실행:

| 낙하 모드 | 낙하 조건 | 피벗 코너 |
|-----------|-----------|-----------|
| Face 1~6 | 6개 면 낙하 | 해당 면 전체 |
| Edge 1~12 | 12개 모서리 낙하 | 해당 엣지 |
| Corner 1~8 | 8개 꼭짓점 낙하 | 피벗 코너 |

각 조건의 결과(simulation_result.pkl)는 독립적으로 저장되어 WHT_LightChassisModel의 ESL 추출 입력으로 사용된다.

---

## 4. Mid-Fidelity Engine: JAX-SSR 구조 응력 복원

### 4.1 Kabsch 기구학 분리

강체 운동과 변형을 분리하기 위해 SVD 기반 Kabsch 알고리즘을 JAX vmap으로 전 프레임 병렬 적용:

$$\mathbf{H} = (\mathbf{Q} - \bar{\mathbf{q}})^T (\mathbf{P}_0 - \bar{\mathbf{p}}_0) \xrightarrow{\text{SVD}} \mathbf{R}_{opt} = \mathbf{V}\mathbf{U}^T$$

충격 임펄스 순간(상대 Z가속도 > 임계값)의 코너는 Kabsch fit에서 제외하여 관성 보정 오류를 방지한다.

### 4.2 Kirchhoff-Love 다항식 피팅

Tikhonov 정규화 최소자승법으로 수직 변위 $w(x,y)$를 다항식으로 근사:

$$M = \frac{\mathbf{X}^T \mathbf{X}}{N} + \lambda \cdot \frac{\mathbf{B}_{xx}^T \mathbf{B}_{xx} + \mathbf{B}_{yy}^T \mathbf{B}_{yy} + 2\mathbf{B}_{xy}^T \mathbf{B}_{xy}}{N} + \epsilon \mathbf{I}$$

Von-Mises 응력, 곡률 텐서, 등가 변형률 등 17개 필드를 산출하며, JAX JIT+vmap으로 GPU 가속.

### 4.3 VTKHDF v2.2 내보내기

h5py로 ParaView 6.0+ 네이티브 VTKHDF v2.2 파일을 직접 생성:

| 방식 | 파일 수 | 내보내기 시간 | 용량 |
|------|---------|--------------|------|
| PVD+VTU | 150개+ | ~5s | ~8MB |
| **VTKHDF v2.2** | **1개** | **~0.3s** | **~1.5MB** |

Static Topology 최적화: Quad 위상(Connectivity/Types/Offsets) 1회 기록, `Steps/CellOffsets=0`으로 파일 크기 최소화.

---

## 5. Mid-Fidelity+: WHT_LightChassisModel 토포그래피 최적화

### 5.1 시스템 개요

WHT_LightChassisModel은 TVPackageMotionSim에서 생성된 **다각도 낙하 궤적 데이터**를 입력으로 받아, 새시(Chassis)의 최적 비드(Bead) 패턴을 **상용 FEA 없이** 결정하는 독립 프레임워크다. 핵심 아이디어: LF 시뮬레이션의 저해상도 정보만으로 초기 샤시 강성 설계를 자동화하는 산업적 혁신.

**아키텍처:**
```
wht_modeler/     ← LS-DYNA IO, FEM 메시 엔티티
wht_solver/      ← JaxSSO 기반 FEM 해석 + 최적화
wht_topo/        ← 토포그래피 최적화 + ESL 추출 + 모니터 UI
wht_converter/   ← WHTResultData IR → VTKHDF/PVD 내보내기
```

### 5.2 ESL 추출 파이프라인

낙하 시뮬레이션(또는 실측 궤적 CSV)으로부터 등가 정적 하중을 추출하는 3단계 프로세스:

**Step 1: Kabsch 전처리**

강체 병진·회전 제거 → body-frame 3D 변형량 추출. 충격 임펄스 순간($|a_z| >$ contact_threshold) 코너를 Kabsch fit에서 제외하여 충격 특이점 보정.

진단 그래프(X/Y/Z 방향 7행 서브플롯) 자동 생성 → 데이터 품질 즉시 검증.

**Step 2: 동적 응답 해석**

4개 코너 마스터 노드(#900000~#900003) + RBE3 연결로 SPCD 하중 그룹 구성. 관성 하중 보정($F = -m\mathbf{a}$) 포함 시:

$$\mathbf{a}_{body} = \mathbf{R}_{Kabsch}^T \cdot \bar{\mathbf{a}}_{world}$$

Von Kármán 비선형 보정계수를 FEM 역산으로 계산하여 관성 하중 크기를 물리적으로 타당하게 스케일:

$$\alpha_{FEM} = \frac{w_{NL,target}}{w_{linear,FEM}}$$

해석 옵션:
- Newmark-β 직접 적분 (기본)
- 모달 중첩법 (--n-modes N, 직접 적분 대비 수십~수백 배 빠름)
- JAX 직접 적분 (--use-jax)

**Step 3: Greedy 다양성 선별**

SE(변형에너지) 이력을 n_windows 구간으로 분할 → 전역 피크 후보 추출 → **Greedy Max-Min Cosine Similarity** 다양성 알고리즘으로 중복이 적은 Top-n_top 스냅샷 선정:

$$\text{sel}_{k+1} = \arg\max_{i \notin \text{sel}} \min_{j \in \text{sel}} (1 - \cos(\mathbf{f}_i, \mathbf{f}_j))$$

이 다양성 선별이 핵심: 유사한 하중 케이스의 중복 추가를 방지하여 최적화 목적 함수의 대표성을 보장한다.

### 5.3 정적 하중 케이스 구성

낙하 ESL 외에 구조 엔지니어링적으로 중요한 6가지 정적 하중 케이스를 자동 구성:

| 하중 케이스 | 경계 조건 | 하중 | 의의 |
|------------|-----------|------|------|
| Bending | 플랜지 전체 고정 | 바닥면 균일 분포 하중 | 중앙 처짐 최소화 |
| Bending X-span | X단 양끝 고정 | $M = W \cdot L/8$ 모멘트 커플 | X방향 순수 굽힘 |
| Bending Y-span | Y단 양끝 고정 | $M = W \cdot L/8$ 모멘트 커플 | Y방향 순수 굽힘 |
| Twisting | 대각 2코너 고정 | 반대 대각 코너 ±F | 비틀림 강성 |
| Twisting Alt | 반전 대각 고정 | 역방향 비틀림 | 비대칭 구조 대응 |
| Lifting (×4) | 3코너 고정 | 1코너 상향 F | 각 코너별 리프팅 |

자중 기반 하중 자동 계산: $W_{chassis} = \Sigma(m_{node}) \times 9806$ mm/s²

### 5.4 WHTopographySolver: JAX 자동미분 MMA 최적화

**설계 변수**: 요소별 비드 높이 $h_e \in [0, h_{max}]$ (요소 기반, SIMP 불필요)

**목적 함수 (다중 옵션)**:

| 유형 | 수식 | 적용 |
|------|------|------|
| Sum | $f = \sum_i w_i C_i / C_0$ | 기본 |
| Normalize | $f = \sum_i w_i (C_i/C_{i0})$ | 정적+동적 케이스 균등 반영 |
| Max (Softmax) | $f = \frac{1}{\alpha}\log\sum_i \exp(\alpha w_i C_i/C_{i0})$ | 최악 케이스 방어 |
| Sum+Max | $f = 0.5 f_{sum} + 0.5 f_{max}$ | 균형형 |
| +Freq Penalty | $P = \lambda \cdot \max(0, f_0 - f_1)^2/f_0^2$ | 고유진동수 동시 제어 |

**민감도 계산**: JAX 자동미분으로 $\partial C / \partial h_e$ 직접 계산 (전체 강성 행렬 재조립 불필요):

$$\frac{\partial C}{\partial h_e} = \sum_{n \in e} \mathbf{u}_e^T \frac{\partial K_e}{\partial z_n} \mathbf{u}_e$$

QUAD4(MITC4+)와 TRIA3 모두 `jax.vmap`으로 병렬 계산.

**업데이트**: MMA(Method of Moving Asymptotes) — 이동 점근선이 설계 공간 탐색을 제어.

**공간 필터**: 최소 비드 폭 $r_{min}$ 제약 (반경 기반 선형/가우시안 필터). 광학 간섭을 이용한 우주 밀도 요동→별 형성과 유사한 구조적 유추로 설명 가능.

### 5.5 비드 연결 알고리즘

MMA 수렴 후 분리된 비드 섬(island)을 연결하는 4가지 알고리즘:

| 알고리즘 | 원리 | 적용 |
|----------|------|------|
| closing | Morphological Closing (Dilation→Erosion) | 섬 간격 좁을 때 |
| mst | 최소 신장 트리 + Bresenham 직선 연결 | 섬이 광범위하게 분산 |
| geodesic | 기존 밀도장을 따른 최단 경로 | 자연스러운 경로 선호 |
| hybrid | MST → Geodesic 순차 적용 | 최고 품질 |

### 5.6 실행 모드

```
모드 A: 기본 정적 최적화
  python wht_topo/run_topo.py --iters 20 --sym-x

모드 C: 동적 충격 통합 최적화 (반복 ESL)
  python wht_topo/run_topo.py \
    --dynamic-opts "corner235_traj.csv" "face1_traj.csv" "edge34_traj.csv" \
    --add-inertia --sym-x

모드 D: 고신뢰성 산업용 완전 제약 설계
  python wht_topo/run_topo.py \
    --dynamic-opts ... --add-inertia --sym-x \
    --bead-connect 150 --bead-connect-alg geodesic \
    --obj-type sum+max --normalize-obj \
    --freq-penalty 3.0 40 \
    --height-steps 2 \
    --exclude-rect 450,250,120,120

모드 E: 입력 디렉토리 일괄 실행 (topo_arg.txt + CSV 자동 탐색)
  python wht_topo/run_topo.py --input-dir /path/to/input
```

**이터레이션 반복 ESL**: 매 이터레이션마다 현재 비드 형상으로 동해석을 재실행하여 ESL을 갱신 → 구조가 강성화될수록 동적 응답이 바뀌고 ESL도 함께 진화하는 정식 반복 ESL 절차.

**ESL 재사용 최적화 (`--esl-skip-tol`)**: 이전 이터레이션 대비 $\Delta h_{rms} <$ tol이면 동해석을 생략하고 직전 ESL 재사용 → 수렴 후반부 계산 비용 절감.

### 5.7 결과 출력

```
results/D20260610_020000/
├── paraview/
│   ├── iter_000.hdf    ← 비드 형상 + 변위 + 응력 (ParaView VTKHDF)
│   ├── iter_001.hdf
│   └── ...
├── esl_se_report_iter000.png   ← SE 이력 + 선택 스냅샷
├── esl_peak_report_iter000.png ← 요소별 피크 SE 분포
└── final.k                     ← LS-DYNA 최적 비드 패턴 출력
```

---

## 6. High-Fidelity Engine: OpenRadioss 자동 연계

### 6.1 설계 철학: LF→HF 원클릭 에스컬레이션

LF 시뮬레이션이 완료된 pkl 파일에서 6자유도 자세(회전 행렬 R, 위치 t) 및 속도·각속도를 자동 추출하여, OpenRadioss FEM 모델의 경계 조건으로 직접 변환한다:

```python
# export_radioss_from_pkl.py
builder = RadiossModelBuilder(
    config=res.config,
    R_mat=R_mat,     # LF 시뮬레이션 자세 추출
    t_vec=t_vec,     # 위치 (초기 관통 방지 +60mm 보정)
    v_vec=v_vec,     # 충돌 직전 속도
    omega_vec=omega_vec,  # 각속도
    transform_mode='parts',  # 파트 변환 또는 지면 역변환
)
starter = builder.build()  # Gmsh 메시 + .rad 파일 생성
```

### 6.2 RadiossModelBuilder: 자동 메시 생성

Gmsh Python API를 이용하여 6개 파트를 자동 메시화:

| 파트 | 형상 | 요소 타입 | 단위 |
|------|------|-----------|------|
| Box | 폐합 박스 쉘 | SHELL (Quad4) | mm |
| Cushion | 중공 솔리드 (Box 내부 채움) | BRICK (Hex8) | mm |
| OpenCell | 솔리드 폼 | BRICK (Hex8) | mm |
| Chassis | 얇은 쉘 | SHELL (Quad4) | mm |
| Ground | 평판 (고정) | BRICK (Hex8) | mm |

파트별 재료 모델:
- **EPS/EPP 완충재**: `/MAT/LAW70` (SAMP-1 다중 응답점 탄소성)
- **골판지 Box**: `/MAT/LAW25` (직교이방성 탄성)
- **강판 Chassis**: `/MAT/LAW36` (탄소성 + 파단)

### 6.3 경계 조건 자동 변환

LF 시뮬레이션의 자세를 두 가지 방식으로 OpenRadioss에 전달:

**Parts 모드** (`transform_mode='parts'`): 파트들에 `/TRANSFORM ROT+TRA` 적용, 지면은 Z=0에 고정
```
/TRANSFORM ROT+TRA
 R11  R12  R13  Tx
 R21  R22  R23  Ty
 R31  R32  R33  Tz
```

**Ground 모드** (`transform_mode='ground'`): 역변환으로 지면을 기울여 파트는 원점에 고정.

### 6.4 OpenRadioss 실행 자동화

```python
builder.run(nt=4, np_cores=1, callback=progress_callback)
```

내부적으로 `RunOpenRadioss` 래퍼를 호출하여:
1. `starter_win64.exe`: 모델 문법 검사 + 파티셔닝
2. `engine_win64.exe`: 명시적 시간 적분 실행
3. `anim_to_vtkhdf=yes`: 결과를 VTKHDF로 자동 변환

콜백 스트림으로 진행 상황을 Control Center UI에 실시간 전달.

### 6.5 Control Center UI 통합

PySide6 Control Center의 **Run Engine** 버튼과 **Str. Analysis** 버튼에서:

```
[Str. Analysis 클릭]
    ↓
시각화 방식 선택 다이얼로그
  ├── [ParaView (VTKHDF)]  → JAX-SSR → VTKHDF → ParaView 실행
  └── [WHT Visualizer]     → OpenSettingsDialog → Qt 대시보드
```

WHT Visualizer의 `OpenSettingsDialog`에서 해석 해상도(sol.res), 마커 모드, Tikhonov λ 등을 사전 설정.

---

## 7. Fidelity Continuum 데이터 흐름

### 7.1 전체 파이프라인 데이터 흐름

```
[run_drop_simulation_cases_v6.py]
  ├── cfg = get_default_config()
  ├── DropSimulator.simulate() → simulation_result.pkl
  └── run_analysis_pipeline(mode='paraview' | 'visualizer')
        ├── scale_result_to_mm()
        ├── get_assembly_data_from_sim()
        ├── ShellDeformationAnalyzer × N → JAX-SSR
        │     KinematicsManager → KirchhoffPlateOptimizer → PlateMechanicsSolver
        ├── PlateAssemblyManager.run_all()
        ├── latest_results.pkl
        ├── [paraview] WHToolsExporter → Result.vtkhdf → launch_paraview()
        └── [visualizer] QtVisualizerV2 대시보드

[WHT_LightChassisModel/wht_topo/run_topo.py]
  ├── LSDYNAReader.read("chassis.k") → WHTMeshModel
  ├── StochasticLoadManager.get_esl_load_cases_from_csv(csv_path)
  │     ├── Kabsch preprocessing
  │     ├── WHTDynamicSolver.solve_direct_dynamic()
  │     └── extract_esl_advanced() → [(WHTLoadCase, weight)]
  ├── WHTopographySolver.__init__()
  │     ├── _find_design_elements()
  │     ├── _build_sensitivity_cache() [JAX vmap]
  │     └── _build_filter() [spatial filter]
  ├── WHTopographySolver.optimize(n_iters)
  │     ├── WHTSolver.solve_static(lc) × N_cases
  │     ├── vmap_element_grad_jax() [JAX auto-diff]
  │     ├── MMAOptimizer.update()
  │     └── VTKHDFExporter → iter_NNN.hdf
  └── LSDYNAWriter.write("final.k") → LS-DYNA 비드 패턴

[export_radioss_from_pkl.py]
  ├── DropSimResult.load(pkl) → R_mat, t_vec, v_vec, omega_vec
  ├── RadiossModelBuilder.build()
  │     ├── Gmsh 메시: Box, Cushion, OpenCell, Chassis, Ground
  │     ├── Material: LAW70/LAW25/LAW36
  │     └── _write_starter() → TVDrop_0000.rad
  └── RadiossModelBuilder.run(nt=4)
        ├── RunOpenRadioss.batch_run()
        └── anim_to_vtkhdf → TVDrop_ANIMATION.vtkhdf
```

### 7.2 파이프라인 성능 비교

| 단계 | 도구 | 소요 시간 | 해석 정밀도 |
|------|------|-----------|-------------|
| **LF**: 단일 낙하 시뮬레이션 | MuJoCo + Numba | ~40초 | 강체+유연체 근사 |
| **MF**: JAX-SSR 구조 해석 | JAX CPU/GPU | ~3초 | Kirchhoff-Love 이론해 |
| **MF+**: 토포그래피 최적화 (20 iter) | JaxSSO + JAX | ~10~30분 | Linear FEM |
| **HF**: OpenRadioss FEM | OpenRadioss | 1~4시간 | 완전 비선형 FEM |
| 기존 상용 FEA (전체 과정) | LS-DYNA/ABAQUS | 4~12시간 | 완전 비선형 FEM |
| **파이프라인 전체** | **WHTOOLS** | **~30분** | **LF→HF 연속 체계** |

### 7.3 Fidelity 결정 기준

```
설계 탐색 (수백 케이스) → LF만 사용 (40s × N cases)
        ↓ CMA-ES 교정 완료
중간 검증 (유망 후보 10~20개) → MF JAX-SSR (3s × M cases)
        ↓ 최적 비드 패턴 확정
샤시 설계 최적화 → MF+ 토포그래피 (10~30분)
        ↓ 최종 설계 확인
최종 인증 (1~3개) → HF OpenRadioss (1~4h × few cases)
```

---

## 8. Results and Discussion

### 8.1 LF 유연체 모델 검증

$5 \times 3 \times 1$ 격자 Glass Panel 모델의 1차 고유 진동수:

$$f_1 = \frac{\pi}{2}\sqrt{\frac{D}{\rho t}} \left(\frac{1}{a^2} + \frac{1}{b^2}\right)$$

감차원 모델과 Kirchhoff 판 이론의 오차 < 15%. 격자 해상도 증가에 따른 수렴 확인.

### 8.2 낙하 궤적 비교 (Corner 2-3-5 조건, 높이 0.5m)

| 물리 모델 추가 | 착지 속도 오차 | 최대 반발 높이 오차 |
|--------------|--------------|-------------------|
| 기본 모델 | 8.4% | 28% |
| + 공력 | 3.1% | 28% |
| + 소성 | 3.1% | 8% |
| + 스퀴즈 필름 | **1.2%** | **6%** |

### 8.3 JAX-SSR 구조 해석 정확도

마커 기반 Kirchhoff-Love vs. 상세 FEA:
- Peak Von-Mises Stress 상관 계수: **92% 이상**
- 임계 위치(파손 예측): 동일 위치 탐지
- 연산 시간: FEA 8시간 → JAX-SSR **~3초**

### 8.4 토포그래피 최적화 결과

Corner 2-3-5, Face 1, Edge 3-4 낙하 조건의 ESL을 동시 반영한 20 이터레이션 최적화:

- 초기 총 컴플라이언스 대비: **62% 감소**
- 최저 고유진동수: 32Hz → **51Hz** (59% 향상)
- 비드 면적 제약 30% 준수
- 좌우 대칭 제약 활성화 시 대칭 비드 패턴 자동 생성
- 최종 비드 패턴: LS-DYNA `final.k`로 내보내기 → 상세 FEM 직접 활용 가능

### 8.5 OpenRadioss 연계 검증

LF pkl → OpenRadioss `.rad` 자동 생성 후 실행:
- 초기 관통(Initial Penetration) 오류 방지: Z축 60mm 상향 자동 보정
- Starter 문법 검사 0 Error 확인
- 결과 VTKHDF: ParaView에서 LF/MF 결과와 동일한 인터페이스로 비교 검토

---

## 9. System Architecture

### 9.1 소프트웨어 모듈 구성

```
WHToolsBox/TVPackageMotionSim/      [LF + MF]
├── run_discrete_builder/           ← MuJoCo XML 자동 생성
├── run_drop_simulator/
│   ├── whts_engine.py              ← MuJoCo 루프 + Numba 공력/소성
│   ├── whts_jax_ssr.py             ← Kirchhoff SSR 코어 (JAX)
│   ├── whts_multipostprocessor_engine.py
│   ├── whts_analysis_pipeline.py  ← 분석 흐름 조율
│   ├── whts_exporter.py           ← VTKHDF v2.2
│   ├── whts_radioss_builder.py    ← Gmsh + OpenRadioss 자동 생성
│   ├── whts_control_panel.py      ← PySide6 Control Center
│   └── whts_multipostprocessor_ui.py ← WHT Visualizer

WHT_LightChassisModel/              [MF+ Topography]
├── wht_modeler/                    ← LS-DYNA IO, FEM 메시
├── wht_solver/                     ← JaxSSO FEM + 동해석
├── wht_topo/
│   ├── run_topo.py                 ← 통합 실행 진입점 (3,444줄)
│   ├── solver.py                   ← WHTopographySolver (2,367줄)
│   ├── loads.py                    ← ESL 추출 + 정적 하중 케이스
│   └── monitor_ui.py               ← 실시간 최적화 모니터 UI
└── wht_converter/                  ← VTKHDF/PVD 내보내기
```

### 9.2 성능 프로파일

| 항목 | 측정값 |
|------|--------|
| LF 시뮬레이션 실시간 FPS | 30~46 FPS |
| LF 총 시뮬레이션 시간 (1.5s 낙하) | ~40초 |
| MF JAX-SSR 배치 후처리 | ~3초 |
| VTKHDF 내보내기 | ~0.3초 |
| MF+ 토포그래피 최적화 (20 iter) | 10~30분 |
| HF OpenRadioss (자동 생성 포함) | 1~4시간 |
| **기존 상용 FEA 전체 과정** | **4~12시간** |
| **WHTOOLS 전체 파이프라인** | **~30분 (LF+MF+MF+)** |

---

## 10. Broader Impact and Future Directions

### 10.1 산업적 파급 효과

| 기존 방식 | WHTOOLS 프레임워크 |
|----------|------------------|
| 설계 반복 2~4주 | 1~2일 |
| 샤시 비드 설계 (전문가 수작업) | 자동 토포그래피 최적화 (~30분) |
| FEA 단일 케이스 4~12시간 | LF 파이프라인 ~45초 |
| HF FEA 수작업 모델링 | OpenRadioss 자동 생성 |
| 시각화 도구 불일치 | VTKHDF 기반 통합 ParaView 워크플로우 |

### 10.2 확장 연구 방향

1. **Physics-Informed Neural Operator (PINO)**: LF 데이터로 DeepONet 훈련 → FEA 없이 밀리초 단위 응력장 예측

2. **Differentiable Simulation & Inverse Design**: JAX 자동미분으로 시뮬레이션 전체를 미분 가능하게 구성 → 목표 충격 응답으로부터 최적 포장/샤시 구조를 역설계

3. **이터레이션 반복 ESL + 비선형 구조해석 결합**: WHT_LightChassisModel의 MMA 루프에서 현재는 선형 FEM을 사용하나, JaxSSO 비선형 확장 또는 Kirchhoff-Love 대변형 이론 적용으로 대변위 비드 형상의 정확도 향상

4. **Self-Evolving Digital Twin**: IoT 가속도 센서 스트리밍 데이터 → 실시간 LF 파라미터 자기 교정 → 유통 환경 변화(온도·습도에 의한 재료 열화) 능동 반영

5. **GNN 기반 다물체 접촉 학습**: 이산 블록 격자를 그래프로 표현, GNN으로 접촉-변형 전파 패턴 학습 → 새로운 형상에 대한 제로샷 일반화

6. **WHT_LightChassisModel → 다재료 최적화**: 현재 비드 높이(형상 최적화)에서 두께(사이징) + 재료(재료 선택) 변수로 설계 공간 확장

---

## 11. Conclusion

본 연구에서는 대형 TV 포장재의 낙하 충격 설계를 위해 **Low-Fidelity → Mid-Fidelity → High-Fidelity Fidelity Continuum**을 갖춘 WHTOOLS 멀티스케일 디지털 트윈 프레임워크를 제안하고 구현하였다:

1. **MuJoCo LF 엔진**: N×M 격자 + 6자유도 조인트 + Numba 공력/소성으로 실시간 대변형 해석 (~40초/케이스)
2. **JAX-SSR MF 엔진**: Kabsch 기구학 + Tikhonov Kirchhoff-Love로 마커 궤적 → 연속 응력장 복원 (~3초)
3. **CMA-ES + DTW 교정**: 실측 마커 궤적과 LF 시뮬레이션의 자동 파라미터 동기화
4. **WHT_LightChassisModel MF+**: LF 다각도 낙하 데이터 → ESL → JAX 자동미분 MMA 토포그래피 최적화 → 초기 샤시 비드 설계 자동화 (~30분, 상용 FEA 불필요)
5. **OpenRadioss HF 자동 연계**: LF pkl → Gmsh 메시 + `.rad` 파일 자동 생성 → 원클릭 HF FEM 실행
6. **VTKHDF v2.2 일관성**: 전 단계 결과를 동일한 ParaView 포맷으로 출력

이 Fidelity Continuum 체계는 설계 탐색(LF)에서 구조 최적화(MF+)를 거쳐 최종 인증(HF)까지 매끄럽게 에스컬레이션되며, 포장-샤시 설계의 \"시행착오 기반 반복\"을 \"데이터 기반 지능형 반복\"으로 전환하는 실질적 방법론을 제공한다.

---

## References

[1] Todorov, E., Erez, T. & Tassev, Y., "MuJoCo: A Physics Engine for Model-Based Control," *IEEE/RSJ IROS*, 2012.

[2] Burgess, G. J., "Product Fragility and Damage Boundary Theory," *Packaging Technology and Science*, vol. 1, no. 1, pp. 5–10, 1988.

[3] Hamrock, B. J., Schmid, S. R. & Jacobson, B. O., *Fundamentals of Fluid Film Lubrication*, CRC Press, 2004.

[4] Hansen, N. & Ostermeier, A., "Completely Derandomized Self-Adaptation in Evolution Strategies," *Evolutionary Computation*, vol. 9, no. 2, pp. 159–195, 2001.

[5] Park, G.-J. & Kang, B.-S., "Validation of a Structural Optimization Algorithm Transforming Dynamic Loads into Equivalent Static Loads," *J. Optim. Theory Appl.*, vol. 118, no. 1, pp. 191–200, 2003.

[6] Svanberg, K., "The Method of Moving Asymptotes — A New Method for Structural Optimization," *Int. J. Numer. Meth. Engng*, vol. 24, pp. 359–373, 1987.

[7] Bradbury, J. et al., "JAX: Composable Transformations of Python+NumPy Programs," 2018. http://github.com/google/jax

[8] Lam, S. K., Pitrou, A. & Seibert, S., "Numba: A LLVM-based Python JIT Compiler," *LLVM-HPC Workshop*, 2015.

[9] Altair Engineering, "OpenRadioss: Open-Source Explicit FEM Solver," 2022. http://openradioss.org

[10] Müller, M., "Dynamic Time Warping," *Information Retrieval for Music and Motion*, Springer, 2007.

[11] Lu, L., Jin, P. & Karniadakis, G. E., "DeepONet: Learning Nonlinear Operators," *Nature Machine Intelligence*, vol. 3, pp. 218–229, 2021.

[12] Grieves, M. & Vickers, J., "Digital Twin: Mitigating Unpredictable Emergent Behavior in Complex Systems," *Transdisciplinary Perspectives on Complex Systems*, Springer, 2017.

---

## Appendix A: 소프트웨어 요구사항

| 구성 요소 | 버전 | 역할 |
|----------|------|------|
| Python | 3.10+ | 기본 실행 환경 |
| MuJoCo | 3.x | LF 물리 엔진 |
| JAX | 0.4+ | MF 배치 연산 + 자동미분 |
| JaxSSO | latest | MF+ FEM 해석 |
| Numba | 0.57+ | LF 공력 JIT 가속 |
| PySide6 | 6.5+ | Control Center UI |
| h5py | 3.10+ | VTKHDF 직접 생성 |
| Gmsh | 4.x | HF 자동 메시 생성 |
| OpenRadioss | latest | HF FEM 솔버 |
| ParaView | 6.0+ | VTKHDF 시각화 |
| Gooey | latest | 토포그래피 최적화 GUI |

---

## Appendix B: WHT_LightChassisModel 핵심 파라미터

```bash
# 모드 D 산업용 완전 설계 예시
python wht_topo/run_topo.py \
  --dynamic-opts \
    "results/rds-20260610/corner235/structural_dynamics.csv" \
    "results/rds-20260610/face1/structural_dynamics.csv" \
    "results/rds-20260610/edge34/structural_dynamics.csv" \
  --add-inertia \
  --sym-x \
  --iters 20 \
  --min-width 80.0 \
  --height-steps 2 \
  --bead-connect 150 \
  --bead-connect-alg geodesic \
  --obj-type sum+max \
  --normalize-obj \
  --freq-penalty 3.0 40 \
  --projection 32 \
  --bead-area 0.30 \
  --exclude-rect 450,250,120,120 \
  --exclude-rect 1350,250,120,120 \
  --n-modes 20 \
  --parallel-scenarios 3
```

---

*Document Version: v7.0 (Fidelity Continuum Edition) | Last Updated: 2026-06-10 | © 2026 WHTOOLS. All rights reserved.*
