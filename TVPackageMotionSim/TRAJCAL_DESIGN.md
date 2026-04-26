# TrajCal — Trajectory-Based Parameter Calibration
## Design Specification v1.0 (2026-04-26)

---

## 1. Overview

TrajCal은 실측 낙하 시험에서 취득한 코너 마커 궤적 데이터를 기준으로,
MuJoCo 시뮬레이션의 물리 파라미터(`cfg` 딕셔너리)를 자동 교정(Calibration)하는 시스템이다.

**목표**: 실측 궤적과 시뮬레이션 궤적 사이의 불일치를 최소화하는 cfg 파라미터 세트를 CMA-ES 최적화로 탐색한다.

**핵심 철학**:
- 절대 좌표가 아닌 **상대 변위(Δpos)** 와 **속도**를 비교 → 좌표계 오차 제거
- **DTW (Dynamic Time Warping)** 로 시간축 불일치에 강건한 비교
- 복합 점수로 형태·속도·이벤트 타이밍을 동시에 평가

---

## 2. Corner Naming Convention

### 2.1 C1–C8 정의 (사용자 기준)

TV 전면(화면)을 정면으로 바라볼 때:

```
전면 (Front face, z = -d/2):          후면 (Back face, z = +d/2):

  C4 ─────── C1                          C8 ─────── C5
  │           │                          │           │
  │  (front)  │    →정면 기준 시계방향    │  (back)   │
  │           │                          │           │
  C3 ─────── C2                          C7 ─────── C6
```

- C1: 전면 우상단, C2: 전면 우하단, C3: 전면 좌하단, C4: 전면 좌상단 (전면 시계방향)
- C5: C1의 뒤, C6: C2의 뒤, C7: C3의 뒤, C8: C4의 뒤

### 2.2 LTL Face 번호 → 코너 포인트 매핑

"Corner 2-3-5" 표기는 **LTL face 번호**이며, 해당 면들이 만나는 꼭지점이 pivot 코너이다.

| LTL Drop Mode | Pivot Corner Point | 설명 |
|---|---|---|
| Corner 2-3-5 | **C5** | face 2(저면), face 3(우측면), face 5(후면)이 만나는 점 = C5 |
| Corner 1-2-5 | C6 | face 1(전면), face 2(저면), face 5(후면) → C6 |
| Corner 1-2-6 | C2 | |
| Corner 1-3-6 | C3 | |

### 2.3 시뮬레이션 내부 인덱스 매핑

`compute_corner_kinematics` (whts_utils.py)는 로컬 좌표 반복 순서로 8개 코너를 생성한다:

```python
# 생성 순서: x in [-w/2, +w/2], y in [-h/2, +h/2], z in [-d/2, +d/2]
idx 0: (-w/2, -h/2, -d/2)  ← 전면 좌하단 = C3
idx 1: (-w/2, -h/2, +d/2)  ← 후면 좌하단 = C7
idx 2: (-w/2, +h/2, -d/2)  ← 전면 좌상단 = C4
idx 3: (-w/2, +h/2, +d/2)  ← 후면 좌상단 = C8
idx 4: (+w/2, -h/2, -d/2)  ← 전면 우하단 = C2
idx 5: (+w/2, -h/2, +d/2)  ← 후면 우하단 = C6
idx 6: (+w/2, +h/2, -d/2)  ← 전면 우상단 = C1
idx 7: (+w/2, +h/2, +d/2)  ← 후면 우상단 = C5
```

**C명칭 → 시뮬 인덱스 테이블**:

| C명칭 | 시뮬 인덱스 | 로컬 좌표 |
|-------|------------|---------|
| C1 | 6 | (+w/2, +h/2, -d/2) |
| C2 | 4 | (+w/2, -h/2, -d/2) |
| C3 | 0 | (-w/2, -h/2, -d/2) |
| C4 | 2 | (-w/2, +h/2, -d/2) |
| C5 | 7 | (+w/2, +h/2, +d/2) |
| C6 | 5 | (+w/2, -h/2, +d/2) |
| C7 | 1 | (-w/2, -h/2, +d/2) |
| C8 | 3 | (-w/2, +h/2, +d/2) |

> **주의**: 위 매핑은 MuJoCo 시뮬레이션에서 박스의 로컬 X축이 Width 방향, Y축이 Height 방향,
> Z축이 Depth 방향으로 정렬됐을 때 성립한다. 다른 orientation이면 재매핑 필요.

---

## 3. 입력 데이터 형식

### 3.1 파일 포맷 (실측 기준: `resources/profiles.txt`)

**탭 구분 텍스트** (`.txt` 또는 `.csv` 모두 지원).

```
열 구성 (26열):
  Frame  Time  | C1~C8 X (8열) | C1~C8 Y (8열) | C1~C8 Z (8열)
```

실제 헤더:
```
Frame	Time	C1_pos_X  C2_pos_X  C3_pos_X  C4_pos_X  C5_pos_X  C6_pos_X  C7_pos_X  C8_pos_X
                C1_pos_Y  C2_pos_Y  C3_pos_Y  C4_pos_Y  C5_pos_Y  C6_pos_Y  C7_pos_Y  C8_pos_Y
                C1_pos_Z  C2_pos_Z  C3_pos_Z  C4_pos_Z  C5_pos_Z  C6_pos_Z  C7_pos_Z  C8_pos_Z
```

실제 데이터 예시 (2행):
```
0    0.001200    0.000002  -0.483464  -0.636768  -0.153302  0.636768  0.153302  -0.000002  0.483464 \
                0.000005  -0.917819  -0.828343   0.089481  0.828343 -0.089481  -0.000005  0.917819 \
                2.687220   2.084013   2.070740   2.673948  0.916480  0.313272   0.300000  0.903207
```

### 3.2 포맷 특성

| 항목 | 값 |
|---|---|
| 구분자 | 탭 (`\t`) |
| 헤더 | 1행, 컬럼명 기준 파싱 |
| 헤더 패턴 | `C{i}_pos_X`, `C{i}_pos_Y`, `C{i}_pos_Z` (1-indexed) |
| 단위 | 미터 (m) — 시뮬레이션 내부 단위와 동일 |
| 타임스텝 | 0.0024 s (= `reporting_interval` 설정값) |
| 첫 프레임 시각 | 0.0012 s (= `sim_timestep` 기준 첫 기록 시점) |
| 코너 수 | 8개 (C1~C8) 기본; 추가 마커(C9 등)는 skip하지 않고 자동 수용 |

### 3.3 좌표계 해석

파일의 Z 값이 ~2.7 m (C1) ~ ~0.3 m (C7): Z가 **수직(높이) 방향**임을 의미.
낙하 전 박스 상단 모서리(C1)가 약 2.7 m, 하단 모서리(C7)가 약 0.3 m.

시뮬레이션 `corner_pos_hist`의 좌표계와 동일 여부를 로더에서 자동 검증:
- 초기 프레임의 C1~C8 절대 좌표 분포가 박스 크기(box_w × box_h × box_d)와 일치하는지 확인
- 불일치 시 경고 출력 (회전 변환 필요 가능성)

### 3.4 단위 변환 옵션

파일 단위가 mm인 경우 `unit_scale=0.001` 지정:
```python
meas = load_trajectory("profiles.txt", unit_scale=1.0)   # 기본: m 그대로 사용
meas = load_trajectory("test.csv",     unit_scale=0.001) # mm → m 변환
```

---

## 4. 비교 방법론

### 4.1 변위 기반 비교에 대한 의견

**사용자 제안**: 각 코너의 시간에 따른 변위 `Δpos(t) = pos(t) - pos(t₀)` 를 비교.

**동의 및 이유**:
- 실측과 시뮬의 절대 좌표계 원점이 다름 → 상대 변위로 제거 가능
- 각 코너의 독립적 운동을 표현하는 가장 직접적인 물리량
- 노이즈에 의한 절대 위치 드리프트 제거

**추가 권장사항**:
- 속도 `v(t) = dΔpos/dt` 도 함께 비교 → 충돌 순간의 동역학 민감도 향상
- 3D 궤적 통합 비교: 각 축(X, Y, Z) 독립 비교 + 3D 궤적 형태 비교 병행

### 4.2 강건한 비교 알고리즘 설계

직접 시간 정렬 비교(time-aligned MSE)의 취약점:
- 낙하 높이 오차 → 충돌 시점이 시험/해석 간에 수십ms 차이
- 센서 샘플링 지연, 프레임 드롭
- 고주파 진동 노이즈가 비용 함수를 왜곡

**해결책: 다중 스케일 DTW 기반 복합 점수**

#### 4.2.1 신호 전처리

```
원시 pos(t) 입력 (실측 또는 시뮬)
    │
    ├─ 0. [공통 시간축 리샘플링]  ← 타임스텝 불일치 처리
    │       t_common = 두 신호의 공통 시간 그리드 생성
    │       pos_meas(t_common) ← scipy interp1d (cubic)
    │       pos_sim (t_common) ← scipy interp1d (cubic)
    │
    ├─ 1. 상대 변위: Δpos(t) = pos(t) - pos(t₀)
    ├─ 2. Savitzky-Golay 평활화 (window=11, poly=3) → 노이즈 제거
    ├─ 3. 속도 산출: v(t) = Δpos'(t) (유한 차분 후 다시 평활화)
    └─ 4. 단위 정규화: signal_norm = signal / max(|signal|)  ← 척도 불변
```

**타임스텝 불일치 처리 상세**:

실측(`profiles.txt`)과 시뮬레이션의 타임스텝이 다를 수 있다:
- 실측: `profiles.txt` 기준 0.0024 s (고정)
- 시뮬: `reporting_interval` 설정에 따라 변동

공통 시간 그리드 전략:
```python
# 두 신호의 겹치는 구간 + 더 조밀한 쪽의 해상도 사용
t_start = max(t_meas[0],  t_sim[0])
t_end   = min(t_meas[-1], t_sim[-1])
dt      = min(np.mean(np.diff(t_meas)), np.mean(np.diff(t_sim)))
t_common = np.arange(t_start, t_end, dt)

# 각 신호를 cubic spline으로 보간
from scipy.interpolate import interp1d
interp_meas = interp1d(t_meas, pos_meas, axis=0, kind='cubic', fill_value='extrapolate')
interp_sim  = interp1d(t_sim,  pos_sim,  axis=0, kind='cubic', fill_value='extrapolate')
pos_meas_r  = interp_meas(t_common)   # shape (N_common, n_corners, 3)
pos_sim_r   = interp_sim (t_common)
```

보간은 로딩 시 1회 수행하므로 최적화 루프 내 비용에 포함되지 않는다.

#### 4.2.2 DTW 기반 변위 유사도 (F_disp)

**Dynamic Time Warping**은 두 시계열의 시간 워핑을 허용하여 이벤트 타이밍 차이에 강건하다.

```
F_disp = DTW_3D(Δpos_sim(t), Δpos_meas(t))

DTW_3D 구현:
  - 각 축 독립 DTW 후 가중 합산: F_x + F_y + F_z
  - 또는 3D 점 간 유클리드 거리로 DTW 경로 비용 계산 (권장)
```

- 정규화된 신호 사용 → 절대 변위 크기 차이를 흡수
- 충돌 전/충돌 중/충돌 후 세 구간에 가중치 달리 적용 가능

#### 4.2.3 DTW 기반 속도 유사도 (F_vel)

```
F_vel = DTW_1D(|v_sim(t)|, |v_meas(t)|)
```

- 속도 크기(Magnitude) 비교는 방향 오차를 흡수
- 충돌 순간의 피크 속도, 충격 감쇠율을 동시에 평가
- 변위보다 미분 신호이므로 드리프트에 강건

#### 4.2.4 주파수 영역 유사도 (F_freq) — 선택적

```
F_freq = 1 - corr(|FFT(Δpos_sim)|, |FFT(Δpos_meas)|)
```

- 충돌 후 진동 주파수와 감쇠율 캡처
- 재료 강성/감쇠 파라미터 튜닝에 특히 민감

#### 4.2.5 이벤트 타이밍 페널티 (F_event)

주요 이벤트: 최대 속도(= 충돌 순간), 첫 번째 반등(first bounce), 최대 변위

```
F_event = Σ_i (|t_event_sim_i - t_event_meas_i| / T_total)
```

- 이벤트 탐지: 속도 곡선의 극솟값, 변위 극값
- 이벤트 수가 일치하지 않으면 패널티 부여

#### 4.2.6 최종 복합 비용 함수

```
F_total = F_traj + F_reg

F_traj = w_disp * F_disp + w_vel * F_vel + w_freq * F_freq + w_event * F_event

F_reg  = Σ_i  λ_i × (θ_i - θ_i_preferred)²   ← 파라미터 우선순위 항
```

기본 궤적 가중치:
```
w_disp  = 0.40   ← 전체 궤적 형태
w_vel   = 0.35   ← 충돌 동역학
w_freq  = 0.10   ← 진동 특성 (노이즈 없는 경우 높임)
w_event = 0.15   ← 이벤트 타이밍
```

`F_reg`의 역할: 궤적 오차(F_traj)가 동등하다면 설계자가 선호하는 파라미터 조합으로 수렴.
λ 설정 기준 (정규화 공간 [0,1] 기준):

| 설계 의도 | λ 값 |
|---|---|
| 1순위 — 자유롭게 조정 | `0.0` |
| 2순위 — 필요시 조정 | `0.05 ~ 0.2` |
| 3순위 — 최후 수단 | `0.5 ~ 2.0` |
| 사실상 고정 | `5.0+` |

가중치는 cfg에서 조정 가능하며, 데이터 품질에 따라 수정한다.
(예: 고노이즈 데이터 → w_freq↓, 샘플링 불균일 → w_event↓)

---

## 5. 파라미터 공간 정의

### 5.1 최적화 대상 cfg 키 분류

최적화 대상 파라미터는 `trajcal_param_space.py`에서 `ParamDef` 리스트로 정의한다.
각 `ParamDef`는 (cfg_key_path, min, max, log_scale, **regularization_weight**) 를 가진다.

### 5.0 파라미터 우선순위 — 이탈 패널티(Regularization)

두 파라미터가 같은 방향으로 목적함수를 개선할 수 있을 때(상관된 설계 변수),
설계자는 특정 파라미터를 우선적으로 변동시키고 나머지는 초기값에 가깝게 유지하고 싶을 수 있다.

**해결책**: 비용 함수에 파라미터별 이탈 패널티를 추가한다 (Tikhonov Regularization):

```
F_total = F_traj + Σ_i  λ_i × (θ_i - θ_i_preferred)²
```

- `λ_i = 0.0`: 자유롭게 조정 (1순위 — 최우선 설계 변수)
- `λ_i = 0.05`: 약간 억제 (2순위 — 필요시 조정)
- `λ_i = 1.0+`: 사실상 고정 (3순위 — 최후 수단 또는 참고용)

`θ_preferred`는 초기 cfg값이 기본이며, 문헌값이나 실측 기준값으로도 지정 가능.

λ는 정규화된 파라미터 공간 [0,1] 기준이므로 서로 다른 파라미터 간 직접 비교 가능.

**예시**: 쿠션 접지 강성(물리적으로 불확실)과 쿠션 weld 강성(실험 데이터 있음)이
동일한 효과를 낼 때, weld 강성을 우선하려면:

```python
ParamDef('contact_solref',  ..., regularization_weight=0.5),  # 억제 — 2순위
ParamDef('weld_solref',     ..., regularization_weight=0.0),  # 자유 — 1순위
```

**수렴 후 진단**: 리포트에서 각 파라미터의 실제 이동량 `|θ_final - θ_init|`을 표시.
두 파라미터가 모두 크게 움직였다면 상관성이 높다는 신호 →
다음 캘리브레이션 시 λ를 재조정하거나 한 쪽을 고정하는 근거로 사용.

#### Group A: 쿠션 접촉 특성 (충돌 거동에 가장 직접적)

| cfg 키 경로 | 설명 | Min | Max | 로그 스케일 |
|---|---|---|---|---|
| `contacts.('ground','cushion').solref[0]` | 쿠션 접지 강성 | 0.0001 | 0.01 | Yes |
| `contacts.('ground','cushion').solref[1]` | 쿠션 접지 감쇠 | 0.1 | 2.0 | No |
| `contacts.('ground','cushion').solimp[0]` | 최소 임피던스 | 0.01 | 0.5 | No |
| `contacts.('ground','cushion').solimp[1]` | 최대 임피던스 | 0.7 | 1.0 | No |
| `contacts.('ground','cushion').solimp[2]` | 전환 폭 | 0.005 | 0.1 | Yes |

#### Group B: 용접 강성 (충돌 후 진동 거동)

| cfg 키 경로 | 설명 | Min | Max | 로그 스케일 |
|---|---|---|---|---|
| `welds.cushion.solref[0]` | 쿠션 weld 강성 | -1e6 | -1e3 | Yes (절댓값) |
| `welds.cushion.solref[1]` | 쿠션 weld 감쇠 | -2000 | -50 | Yes (절댓값) |
| `welds.opencell.solref[0]` | Open Cell 강성 | -1e6 | -1e3 | Yes |
| `welds.chassis.solref[0]` | Chassis 강성 | -1e6 | -1e3 | Yes |

#### Group C: 소성 특성

| cfg 키 경로 | 설명 | Min | Max |
|---|---|---|---|
| `cush_yield_pressure` | 쿠션 항복 압력 [Pa] | 300 | 8000 |
| `plasticity_ratio` | 소성 비율 | 0.05 | 0.8 |
| `plastic_hardening_modulus` | 가공 경화 계수 | 1000 | 200000 |

#### Group D: 마찰

| cfg 키 경로 | Min | Max |
|---|---|---|
| `contacts.('ground','cushion').friction[0]` | 0.1 | 1.5 |
| `contacts.('ground','paper').friction[0]` | 0.1 | 1.5 |

#### Group E: 질량/관성 (실측값 있으면 고정)

| cfg 키 경로 | 설명 | Min | Max |
|---|---|---|---|
| `components_balance.target_cog[0]` | CoG X 편심 [m] | -0.05 | 0.05 |
| `components_balance.target_cog[1]` | CoG Y 편심 [m] | -0.05 | 0.05 |
| `components_balance.target_cog[2]` | CoG Z 편심 [m] | -0.02 | 0.02 |
| `components_balance.target_mass` | 총 질량 [kg] | 30 | 60 |

### 5.2 파라미터 선택 전략

초기 캘리브레이션 권장 순서:
1. **Phase 1**: Group A만 활성화 (충돌 순간 거동 맞추기)
2. **Phase 2**: Group A + B 활성화 (진동 감쇠 맞추기)
3. **Phase 3**: 필요시 Group C, D, E 추가

---

## 6. 시스템 아키텍처

```
TVPackageMotionSim/
└── trajcal/
    ├── __init__.py
    ├── trajcal_loader.py          # CSV/탭 파일 로딩 + Δpos, vel 산출
    ├── trajcal_corner_mapper.py   # C1-C8 ↔ 시뮬 인덱스 매핑, LTL face→코너
    ├── trajcal_comparator.py      # DTW + 속도 + 주파수 + 이벤트 비용 함수
    ├── trajcal_param_space.py     # ParamDef 리스트 + 벡터 ↔ cfg 변환
    ├── trajcal_evaluator.py       # cfg로 시뮬 실행 → corner_hist 추출
    ├── trajcal_optimizer.py       # CMA-ES 루프
    └── trajcal_report.py          # 수렴 플롯, 궤적 오버레이 시각화
```

### 6.1 데이터 흐름

```
[CSV 파일]
    │
    ▼
trajcal_loader.py
    MeasuredTrajectory(time, corner_data[C_i][t] = np.array([x,y,z]))
    compute_delta_pos()   → Δpos[C_i][t]
    compute_velocity()    → vel[C_i][t]
    │
    ▼
trajcal_corner_mapper.py
    selected_corners = [C2, C5]  (사용자 선택)
    sim_indices = [4, 7]         (C2→4, C5→7)
    │
    ▼
[최적화 루프]
    CMA-ES 벡터 θ
         │
         ▼
    trajcal_param_space.py   theta_to_cfg(θ) → cfg_dict
         │
         ▼
    trajcal_evaluator.py     run_simulation(cfg) → SimTrajectory
         │                   sim.corner_pos_hist[:, sim_indices, :] 추출
         ▼
    trajcal_comparator.py    compute_fitness(meas, sim, weights)
         │                   → F_total (스칼라)
         ▼
    CMA-ES 업데이트
    │
    ▼ (수렴 후)
trajcal_report.py
    best_cfg 저장, 궤적 비교 플롯 출력
```

### 6.2 모듈별 핵심 인터페이스

#### `trajcal_loader.py`
```python
@dataclass
class MeasuredTrajectory:
    time: np.ndarray             # shape (T,)
    corners: Dict[str, np.ndarray]  # {'C1': (T,3), 'C5': (T,3), ...}
    delta_pos: Dict[str, np.ndarray]  # Δpos = pos - pos[0]
    velocity: Dict[str, np.ndarray]   # smoothed derivative

def load_trajectory(filepath: str, unit_scale: float = 0.001) -> MeasuredTrajectory:
    ...
```

#### `trajcal_corner_mapper.py`
```python
# 고정 테이블
C_NAME_TO_SIM_IDX = {'C1':6, 'C2':4, 'C3':0, 'C4':2, 'C5':7, 'C6':5, 'C7':1, 'C8':3}

def get_sim_corner_array(sim_result, corner_names: List[str]) -> Dict[str, np.ndarray]:
    """sim.corner_pos_hist에서 선택된 코너만 추출 → Dict[str, (T,3)]"""
    ...
```

#### `trajcal_comparator.py`
```python
@dataclass
class ComparatorConfig:
    selected_corners: List[str]         # ['C2', 'C5']
    w_disp: float = 0.40
    w_vel: float = 0.35
    w_freq: float = 0.10
    w_event: float = 0.15
    dtw_sakoe_chiba_radius: int = 20    # DTW 검색 반경 (계산 비용 제어)

def compute_fitness(
    meas: MeasuredTrajectory,
    sim_corners: Dict[str, np.ndarray],
    sim_time: np.ndarray,
    cfg: ComparatorConfig
) -> float:
    ...
```

#### `trajcal_param_space.py`
```python
@dataclass
class ParamDef:
    name: str
    cfg_path: str | tuple   # 'cush_yield_pressure' or ('contacts', ('ground','cushion'), 'solref', 0)
    min_val: float
    max_val: float
    log_scale: bool = False
    active: bool = True
    regularization_weight: float = 0.0        # λ_i: 이탈 패널티 강도 (0=자유, 높을수록 초기값에 고정)
    preferred_value: Optional[float] = None   # None이면 초기 cfg값 자동 사용

class ParamSpace:
    def __init__(self, param_defs: List[ParamDef]):
        ...
    def theta_to_cfg(self, theta: np.ndarray, base_cfg: dict) -> dict:
        """정규화된 벡터 [0,1]^n → cfg 딕셔너리 업데이트"""
        ...
    def cfg_to_theta(self, cfg: dict) -> np.ndarray:
        """현재 cfg → 정규화 벡터 (초기점 추출용)"""
        ...
    def regularization_penalty(self, theta: np.ndarray, theta_init: np.ndarray) -> float:
        """Σ_i λ_i × (θ_i - θ_i_preferred)²  — comparator가 F_traj에 합산"""
        ...
```

#### `trajcal_evaluator.py`
```python
def run_and_extract(
    base_cfg: dict,
    theta: np.ndarray,
    param_space: ParamSpace,
    selected_corners: List[str]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    cfg를 수정 → 시뮬 실행 → corner_pos_hist 추출
    Returns: (corner_dict, time_array)
    시뮬이 실패(nan, diverge)하면 None 반환
    """
    ...
```

#### `trajcal_optimizer.py`
```python
@dataclass
class TrajCalConfig:
    base_cfg: dict
    param_space: ParamSpace
    measured: MeasuredTrajectory
    comparator_cfg: ComparatorConfig
    result_dir: str = "trajcal_results"
    # CMA-ES 하이퍼파라미터
    sigma0: float = 0.3          # 초기 탐색 표준편차 (정규화 공간)
    popsize: int = 12            # 한 세대 샘플 수
    max_evals: int = 200         # 최대 시뮬 실행 횟수
    seed: int = 42

def run_calibration(cfg: TrajCalConfig) -> dict:
    """CMA-ES 루프 실행 → best_cfg 반환"""
    ...
```

---

## 7. CMA-ES 최적화 전략

### 7.1 라이브러리

`cma` Python 패키지 (`pip install cma`) 사용.
- 내장 bounds 처리, 수렴 기준 자동 관리
- 이미 프로젝트에 CMA-ES 관련 파일 존재 (cmaesdtw.py 참조)

### 7.2 최적화 설정

```python
import cma

es = cma.CMAEvolutionStrategy(
    x0=param_space.cfg_to_theta(base_cfg),   # 현재 cfg에서 초기점
    sigma0=0.3,                               # 초기 탐색 반경 (정규화 공간 기준)
    {
        'bounds': [[0]*n, [1]*n],             # 정규화 공간 [0,1]^n
        'popsize': 12,
        'maxfevals': 200,
        'seed': 42,
        'verbose': 1,
        'tolx': 1e-4,                         # 수렴 허용 오차
        'tolfun': 1e-4,
    }
)
```

### 7.3 실패 처리

시뮬레이션 발산(nan, inf) 또는 타임아웃 시:
- 비용 함수에 큰 패널티 반환 (예: 1e6)
- 해당 개체 로그에 기록

### 7.4 계산 비용 고려

시뮬 1회 ≈ 수십 초 ~ 수 분.
- 병렬화: `multiprocessing.Pool`로 세대 내 개체들을 병렬 실행
- `cfg["use_viewer"] = False`, `cfg["use_jax_reporting"] = False` 강제 설정
- DTW Sakoe-Chiba 밴드로 DTW 계산 비용 제한

---

## 8. 사용 흐름 (End-to-End)

```python
from trajcal import (
    load_trajectory, ParamSpace, TrajCalConfig, ComparatorConfig,
    run_calibration, PRESET_PARAMS_PHASE1
)

# 1. 실측 데이터 로드
meas = load_trajectory("test_data/drop_corner235.csv", unit_scale=0.001)

# 2. 비교 코너 선택
comp_cfg = ComparatorConfig(selected_corners=['C2', 'C5'])

# 3. 파라미터 공간 선택 (Phase 1: 쿠션 접촉 파라미터만)
param_space = ParamSpace(PRESET_PARAMS_PHASE1)

# 4. 기본 cfg 세팅 (기존 doe 케이스 재사용)
base_cfg = doe_modeling_case_1_setup.__wrapped__()  # 시뮬 미실행 버전

# 5. 캘리브레이션 실행
best_cfg = run_calibration(TrajCalConfig(
    base_cfg=base_cfg,
    param_space=param_space,
    measured=meas,
    comparator_cfg=comp_cfg,
    max_evals=100,
))

# 6. 결과 저장 + 리포트
# → trajcal_results/best_cfg.json, convergence.png, overlay.png
```

---

## 9. 최적화 진행 모니터링

### 9.1 구조적 제약

CMA-ES 최적화 루프의 특성:
- 시뮬 1회 = 수십 초 ~ 수 분 (blocking)
- 세대(generation) 내 개체들은 병렬 실행 → 여러 worker 프로세스에서 동시에 결과 생성
- 총 수 시간 소요 → 중간에 프로세스가 꺼져도 결과 보존 필요
- `DropSimulator`가 이미 MuJoCo 뷰어를 쓸 수 있어 GUI 충돌 주의

### 9.2 권장 방식: Rich 터미널 + 파일 로그 조합

추가 의존성 없이 실용적인 실시간 모니터링을 제공한다.
(Rich는 `whts_engine.py`에서 이미 사용 중)

#### 레이어 1: Rich 터미널 Live 대시보드 (실시간)

최적화 루프 메인 프로세스가 `rich.live.Live`로 매 세대 업데이트:

```
┌─ TrajCal Optimizer ─ Gen 12 / ∞ ─ Evals 144 / 200 ─ Elapsed 43:21 ─┐
│  Best Cost : 0.2341   │  Sigma : 0.187   │  ETA : ~1h 12m           │
├────────────────────────────────────────────────────────────────────── ┤
│  Parameter                  │  Best Value  │  Init    │  Range        │
│  cush_contact_solref[0]     │  0.00312     │  0.001   │  [1e-4,1e-2] │
│  cush_yield_pressure        │  1847        │  1500    │  [300,8000]  │
│  plasticity_ratio           │  0.31        │  0.30    │  [0.05,0.8]  │
│  target_cog[2]              │  0.008       │  0.010   │  [-0.02,0.02]│
├──────────────────────────────────────────────────────────────────────┤
│  Cost History (last 20 gen) : ▇▇▆▅▅▄▃▃▂▂▂▂▂▁▁▁▁▁▁▁                │
│  Sigma History              : ▇▆▅▅▄▄▃▃▃▂▂▂▂▂▁▁▁▁▁▁                │
└──────────────────────────────────────────────────────────────────────┘
  [Gen 12] Eval 144: cost=0.2789  [worker-3]
  [Gen 12] Eval 145: cost=0.3102  [worker-1] ← DIVERGED (penalty applied)
```

- 세대별 best cost, sigma(탐색 반경), 현재 best 파라미터값 표시
- 히스토리 바 차트: 유니코드 블록 문자로 터미널 내 인라인 렌더링
- SSH 원격 환경에서도 동작, 추가 GUI 불필요

#### 레이어 2: evallog.csv — 영속적 기록 (매 eval 즉시 append)

`result_dir/evallog.csv` 에 매 평가 결과를 기록. 프로세스 강제 종료 시에도 보존.

```csv
gen,eval_id,worker,cost,f_disp,f_vel,f_freq,f_event,sigma,param_0,param_1,...,diverged,elapsed_s
1,1,0,0.8821,0.41,0.32,0.09,0.11,0.300,0.312,1847,...,False,28.4
1,2,1,0.9103,...
...
12,144,3,0.2789,...
```

- `gen`: 세대 번호
- `eval_id`: 전체 누적 평가 횟수
- `cost`: 복합 비용 함수값
- `f_disp/f_vel/f_freq/f_event`: 세부 비용 분해 (어떤 항목이 지배적인지 진단용)
- `diverged`: 시뮬 발산 여부
- `elapsed_s`: 해당 평가 소요 시간 (성능 추적용)

병렬 worker가 동시에 쓰는 race condition 방지: 각 worker는 임시 파일에 저장 후
메인 프로세스가 세대 완료 시점에 수집하여 일괄 append.

#### 레이어 3: 주기적 PNG 저장 (매 10 세대, non-blocking)

`result_dir/` 폴더에 자동 저장:

```
trajcal_results/
├── evallog.csv
├── best_cfg.json           ← 현재 best 파라미터 (항상 최신 유지)
├── convergence_gen012.png  ← 수렴 곡선 스냅샷
└── overlay_gen012.png      ← 실측 vs 시뮬 궤적 오버레이
```

`overlay_gen{N}.png` 구성 (서브플롯):
- 좌: C별 Δpos(t) — 실측(점선) vs 시뮬(실선), 3축
- 우: C별 |vel|(t) — 실측 vs 시뮬
- 제목에 cost 분해값 표기

matplotlib을 백그라운드 스레드(`threading.Thread`)에서 실행 → 최적화 루프 블로킹 없음.

### 9.3 재시작(Resume) 지원

`evallog.csv`가 존재하면 이전 실행 결과에서 CMA-ES 상태를 복원:

```python
# trajcal_optimizer.py 내부
cma_state_file = result_dir / "cma_state.pkl"
if cma_state_file.exists():
    es = pickle.load(open(cma_state_file, 'rb'))  # CMA state 복원
    console.print(f"[green]Resuming from eval {len(existing_log)}[/green]")
else:
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
```

매 세대 완료 후 `cma_state.pkl` 저장.

### 9.4 향후 확장 옵션

현재 A+B 구조를 건드리지 않고 추가 가능:
- `trajcal_live_viewer.py`: `evallog.csv`를 주기적으로 읽어 Qt 창에 실시간 플롯
  (별도 프로세스: `python trajcal_live_viewer.py --log trajcal_results/evallog.csv`)
- Jupyter 연동: `evallog.csv`를 `pd.read_csv()`로 읽어 셀 업데이트

---

## 10. 성능 설계 — JAX vs Numba 역할 분담

### 10.1 병목 지점 분석

평가 1회당 주요 연산:

| 연산 | 입력 크기 | 복잡도 | 병목? |
|---|---|---|---|
| 보간 (interp1d) | N ≈ 800 pt | O(N) | No — 루프 밖 1회 |
| Savitzky-Golay | N × 8코너 × 3축 | O(N) | No |
| DTW (변위) | N × 8코너 × 3축 | **O(N×r)** | **Yes** |
| DTW (속도) | 동일 | **O(N×r)** | **Yes** |
| FFT 유사도 | N × 8 × 3 | O(N log N) | No |
| 이벤트 탐지 | N | O(N) | No |

DTW가 지배적. Sakoe-Chiba 밴드(반경 r) 적용 시 O(N²) → O(N×r)로 감소하지만,
여전히 내부 루프의 순차 의존성(dp[i,j] ← dp[i-1,j-1], dp[i-1,j], dp[i,j-1])이 존재.

### 10.2 Numba — DTW 핵심 루프

DTW는 **순차 의존성이 있는 DP 루프**이므로 JAX의 벡터화로 직접 가속이 어렵다.
Numba `@njit`이 가장 적합: 파이썬 오버헤드 없이 C 수준 속도로 루프 실행.

```python
# trajcal_comparator.py
from numba import njit
import numpy as np

@njit(cache=True)
def dtw_sakoe_chiba(s: np.ndarray, t: np.ndarray, radius: int) -> float:
    """
    Sakoe-Chiba 밴드 제약 DTW. 두 1D 신호 s, t의 정규화된 DTW 거리 반환.
    cache=True: 최초 컴파일 결과를 디스크에 저장 → 재실행 시 즉시 사용.
    """
    n, m = len(s), len(t)
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - radius)
        j_hi = min(m, i + radius)
        for j in range(j_lo, j_hi + 1):
            cost = abs(s[i-1] - t[j-1])
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m] / (n + m)   # 길이 정규화

@njit(cache=True, parallel=False)
def dtw_3d(pos_a: np.ndarray, pos_b: np.ndarray, radius: int) -> float:
    """(N,3) 배열 두 개의 3D DTW — 유클리드 점간 거리 기준."""
    n, m = len(pos_a), len(pos_b)
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - radius)
        j_hi = min(m, i + radius)
        for j in range(j_lo, j_hi + 1):
            dx = pos_a[i-1, 0] - pos_b[j-1, 0]
            dy = pos_a[i-1, 1] - pos_b[j-1, 1]
            dz = pos_a[i-1, 2] - pos_b[j-1, 2]
            cost = (dx*dx + dy*dy + dz*dz) ** 0.5
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m] / (n + m)
```

Sakoe-Chiba 반경 권장값: `radius = int(0.1 * N)` (신호 길이의 10%)
→ N=800일 때 r=80, 계산량 = 800×80 = 64,000 연산/호출 (O(N²)=640,000 대비 10배 절감)

### 10.3 JAX — 전처리 및 배치 집계

프로젝트에 JAX가 이미 도입되어 있으므로 벡터화 연산에 활용:

```python
# trajcal_comparator.py (JAX 파트)
import jax.numpy as jnp
from jax import jit, vmap

@jit
def preprocess_signals(pos: jnp.ndarray) -> tuple:
    """
    (N, n_corners, 3) → Δpos, |vel| 배치 계산.
    JAX jit으로 JIT 컴파일 → 두 번째 호출부터 즉시 실행.
    """
    delta_pos = pos - pos[0]                          # (N, C, 3)
    vel = jnp.gradient(delta_pos, axis=0)             # (N, C, 3)
    vel_mag = jnp.linalg.norm(vel, axis=-1)           # (N, C)
    # 정규화: 각 코너×축 독립 스케일
    scale = jnp.max(jnp.abs(delta_pos), axis=0) + 1e-9
    delta_pos_norm = delta_pos / scale
    vel_scale = jnp.max(vel_mag, axis=0) + 1e-9
    vel_mag_norm = vel_mag / vel_scale
    return delta_pos_norm, vel_mag_norm

@jit
def freq_similarity(sig_a: jnp.ndarray, sig_b: jnp.ndarray) -> float:
    """FFT 스펙트럼 코사인 유사도 (1 - corr) → [0, 1]."""
    fa = jnp.abs(jnp.fft.rfft(sig_a))
    fb = jnp.abs(jnp.fft.rfft(sig_b))
    corr = jnp.dot(fa, fb) / (jnp.linalg.norm(fa) * jnp.linalg.norm(fb) + 1e-9)
    return 1.0 - corr
```

### 10.4 역할 분담 요약

| 연산 | 도구 | 이유 |
|---|---|---|
| 보간 (interp1d) | `scipy` | 1회 실행, 구현 단순 |
| Savitzky-Golay 평활화 | `scipy.signal` | 내장 최적화 구현 |
| **DTW 핵심 루프** | **Numba @njit** | 순차 DP — JAX 벡터화 불가 |
| Δpos, vel 배치 계산 | `JAX @jit` | 이미 프로젝트 도입, GPU 가능 |
| FFT 유사도 | `JAX @jit` | 완전 벡터화 가능 |
| 이벤트 탐지 (극값) | `scipy.signal` | find_peaks 내장 |
| 비용 함수 집계 | `numpy` | 스칼라 연산, 오버헤드 무시 |

### 10.5 컴파일 전략

Numba `@njit(cache=True)`: 최초 실행 시 컴파일 (수 초 소요), 이후 캐시에서 즉시 로드.
JAX `@jit`: 최초 호출 시 XLA 컴파일 (수 초), 이후 즉시 실행.

→ **워밍업 호출**: `trajcal_optimizer.py` 초기화 단계에서 더미 데이터로 한 번씩 호출해
  최적화 루프 진입 전에 컴파일 완료.

```python
# trajcal_optimizer.py 초기화 블록
def _warmup(n: int = 100, r: int = 10):
    dummy = np.random.rand(n).astype(np.float64)
    dtw_sakoe_chiba(dummy, dummy, r)          # Numba 컴파일
    preprocess_signals(jnp.ones((n, 8, 3)))  # JAX XLA 컴파일
```

---

## 11. 구현 우선순위 (Phase별 TODO)

### Phase 1 — 데이터 인프라
- [ ] `trajcal_loader.py`: CSV/탭 파싱, Δpos 및 Savitzky-Golay 평활화 속도 산출
- [ ] `trajcal_corner_mapper.py`: C명칭↔시뮬인덱스 테이블, `get_sim_corner_array()`

### Phase 2 — 비교 엔진
- [ ] `trajcal_comparator.py`: DTW (tslearn 또는 직접 구현), 속도 DTW, 주파수 유사도, 이벤트 탐지

### Phase 3 — 파라미터 공간
- [ ] `trajcal_param_space.py`: ParamDef 정의, theta↔cfg 변환, Preset 그룹 정의

### Phase 4 — 시뮬레이션 평가자
- [ ] `trajcal_evaluator.py`: cfg 주입 → DropSimulator 실행 (viewer/ui 비활성화) → 코너 추출
- [ ] 발산 감지 및 패널티 반환

### Phase 5 — 최적화 루프
- [ ] `trajcal_optimizer.py`: CMA-ES 루프, 병렬 평가, 수렴 로그

### Phase 6 — 리포트
- [ ] `trajcal_report.py`: 수렴 곡선, 궤적 오버레이 플롯 (matplotlib)

---

## 12. 의존성

| 라이브러리 | 용도 | 설치 | 비고 |
|---|---|---|---|
| `cma` | CMA-ES 최적화 | `pip install cma` | 신규 |
| `numba` | DTW 핵심 루프 JIT | `pip install numba` | 신규 |
| `jax` | 전처리 배치 연산 | 기존 설치 | 이미 프로젝트 도입 |
| `scipy` | 보간, Savitzky-Golay, 이벤트 탐지 | 기존 설치 | |
| `matplotlib` | 리포트 플롯 | 기존 설치 | |
| `pandas` | CSV/탭 파싱 | 기존 설치 | |

---

## 11. 관련 파일 (기존 코드베이스)

| 파일 | 역할 |
|---|---|
| `run_drop_simulator/whts_engine.py` | `DropSimulator` — `corner_pos_hist`, `corner_vel_hist` 제공 |
| `run_drop_simulator/whts_utils.py:6` | `compute_corner_kinematics()` — 코너 인덱스 생성 로직 |
| `run_discrete_builder/whtb_config.py` | `get_default_config()`, 파라미터 키 정의 |
| `run_drop_simulation_cases_doe.py` | Case 1 cfg 설정 예시 + `result_base_dir` 옵션 |
