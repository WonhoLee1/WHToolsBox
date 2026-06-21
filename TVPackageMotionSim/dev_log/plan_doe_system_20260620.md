# WHToolsBox DOE System — Implementation Plan
**Status: pending approval**
**Source spec:** `.omc/specs/deep-interview-whtb-doe-system.md`
**Generated:** 2026-06-20

---

## RALPLAN-DR Summary

### Principles
1. **독립성 우선**: `whtb_doe`는 TVPackageMotionSim에 의존하지 않는 순수 독립 패키지
2. **DSL은 텍스트**: config 변수 정의는 Python 문자열로 표현 — 파일 저장/버전 관리 용이
3. **기존 코드 동결**: `run_parameter_study` DOEEngine은 건드리지 않음 — 회귀 위험 없음
4. **subprocess 격리**: 각 DOE case는 독립 OS 프로세스 — MuJoCo 메모리 누수/크래시 격리
5. **단계적 검증**: 5개 컴포넌트를 의존성 순서대로 구현, 각 단계에서 독립 테스트 가능

### Decision Drivers
1. **재사용성**: DOE Core가 TVPackageMotionSim 외 프로젝트에서도 `from whtb_doe import DOEDefinition`으로 사용 가능해야 함
2. **안전성**: subprocess 격리로 DOE case 실패가 전체 실행을 멈추지 않아야 함
3. **배포 가능성**: PySide6 UI + Gemini CLI 채팅 창이 PyInstaller로 묶이는 기존 배포 구조에 포함되어야 함

### Viable Options

#### Option A: AST 전용 파서
**Approach:** Python `ast` 모듈로 전체 DSL 파싱
- **Pros:** stdlib, 표준 Python 구문에서는 정확한 줄 번호
- **Cons:** `[10:199:15]`, `10:200:100` 등 비표준 value 구문은 `SyntaxError` 발생 → AST 도달 불가. 결국 fallback 파서 필요해져 세 개의 파서(AST + regex + value)가 됨. 오류 보고 이점이 가장 흔한 케이스에서 무효화됨

#### Option B: 정규식 기반 파서
**Approach:** regex로 `cfg['a']['b'] = ...` 패턴 추출
- **Pros:** 구현 단순, 커스텀 문법(`[10:100:4]`) 처리 유연
- **Cons:** 중첩 dict 깊이 가변 처리 취약

#### Option C: 합성 파서 — literal_eval+regex(value) + AST(key path) (채택)
**Approach:** 각 라인을 `=`로 분리. **LHS(key path)**: `ast.parse()`로 `Subscript` 체인 추출 (AST가 진정 강한 영역). **RHS(value)**: `ast.literal_eval()` 우선 시도(리스트/문자열 커버), 실패 시 colon-regex fallback(`[a:b:c]`, `a:b:c`, `norm:m:s:n` 처리)
- **Pros:** 각 도구를 강점 영역에만 사용. literal_eval이 `[10,20,30]`, `['A','B']` 무료 처리. 전체 파서 2개(AST LHS + regex RHS)
- **Cons:** 두 코드 경로 유지 (Option A보다는 단순)

> **채택: Option C (합성 파서)**. Architect 지적대로 AST만으로는 비표준 value 구문 파싱 불가. literal_eval+regex가 RHS를 커버하고 AST는 key path에만 집중.

#### Option D (병렬 실행): multiprocessing.Pool
- **Pros:** Python 레벨 통합, pool 관리 자동
- **Cons:** MuJoCo/PySide6 fork 안전성 문제, Windows에서 spawn 모드 필요, 로그 모니터링 복잡

#### Option E (병렬 실행): subprocess 1case/프로세스
- **Pros:** 최대 격리, 단순한 큐 로직
- **Cons:** JAX + MuJoCo cold-start(수 초~수십 초)를 case마다 지불. n_samples=100이면 100번 × cold-start. 단기 케이스는 초기화 시간이 시뮬레이션 시간을 초과 가능

#### Option F (병렬 실행): warm-batch subprocess worker (채택)
**Approach:** subprocess 1개가 K개 케이스를 순차 소비 후 종료. 큐 매니저가 n_parallel 슬롯을 warm-batch worker로 채움. worker가 비정상 종료(returncode != 0)하면 큐 매니저가 즉시 재스폰.
- **Pros:** 프로세스 격리 유지(crash 전파 없음), JAX+MuJoCo 초기화를 K번 상각. 실질적 cold-start 비용 1/K로 감소
- **Cons:** respawn-on-crash 로직이 E보다 약간 복잡. K 값 결정 필요(기본 K=10 권장)

> **채택: Option F (warm-batch worker)**. Architect 측정: cold-start가 수십 초이므로 100케이스에서 E 방식은 cold-start만 수백~수천 초. F 방식으로 상각 필수. K=10 기본값.

---

## Prior Art & Relationship with Existing Code

| 기존 파일 | 역할 | 신규 코드와의 관계 |
|---------|------|-----------------|
| `run_drop_simulation_cases_doe.py` | `doe_process_pipeline()` — 단일 modeling_func 실행 + 결과 저장. LHS/Random/FullFact DOE 없음, 수동 case 정의 | `DOERunner`가 이 패턴을 확장. `run_doe_case_worker` 내부에서 유사 패턴 사용. **이 파일은 동결, `DOERunner`가 canonical DOE 실행 담당** |
| `run_parameter_study/whts_optimization_engine.py` (`DOEEngine`) | LHS/Random/FullFact 샘플링 구현 (기존). GooeyParser 기반 UI | `whtb_doe`는 clean-slate 구현 (Principle 3 예외 없음). `DOEEngine` 재사용 안 함 — 이유: DSL 문자열 파싱 없음, config dict 경로 접근 없음, GooeyParser 의존성 제거 원함 |
| `run_parameter_study/whts_optimization_ui.py` | GooeyParser 기반 최적화 UI | `DOESetupDialog`가 PySide6 기반으로 대체. `run_parameter_study`는 최적화 워크플로우에 계속 사용 |
| `run_drop_simulation_cases_v4/v5/v6.py` | 시나리오별 단일 실행 진입점 | DOE 실행과 무관하게 유지. `v6.py`는 frozen 진입점이므로 `--doe-worker` 라우팅 삽입 대상 |

**결론**: `whtb_doe` + `DOERunner`가 신규 canonical DOE 시스템. 기존 `run_drop_simulation_cases_doe.py`와 `run_parameter_study`는 기존 워크플로우에서 계속 유효하나 DOE 시스템의 주 경로는 아님.

---

## Requirements Summary

| 컴포넌트 | 핵심 요구사항 |
|---------|------------|
| whtb_doe Core | DSL 파싱, LHS/FullFact/MonteCarlo 샘플링, DOEDefinition 클래스 |
| DOE Execution Engine | subprocess n개 병렬, run_doe_log.txt 모니터, done_doe_log.txt append |
| Contact Extraction | 코너 최초 접촉 시점/힘/임팩트, ContactSeqList, 캡처 |
| DOE UI Widget | PySide6 config tree, DOE table 편집, 호출자에 반환 |
| Results Analysis | pkl 로드, 그룹핑, Gemini CLI 채팅 창 |

---

## Implementation Steps

### Phase 0: DropSimResult 스키마 확장 (선행 필수)

**배경 (Architect 발견):**
- `corner_impact_hist` (shape N×8 per-corner 접촉력)는 `whts_engine.py:804-815`에서 계산됨
- 그러나 `DropSimResult` dataclass (`whts_data.py:30-67`)에 필드 없음
- `DropSimResult(...)` 생성자 (`whts_engine.py:1505-1530`)에서도 전달 안 됨
- 결과: 기존 pkl에 per-corner 접촉력 없음 → Phase 3 Contact Extraction 불가능

**`corner_impact_hist` 타입 주의 (Critic 발견):**
`whts_engine.py:336`에서 `self.corner_impact_hist: List[np.ndarray] = []`로 선언되고 `:815`에서 frame마다 append됨 — **Python list**, ndarray 아님.
기존 올바른 사용 패턴: `run_drop_simulation_cases_doe.py:130`에서 `np.array(sim.corner_impact_hist)` 변환.

**수정 대상:**
- `TVPackageMotionSim/run_drop_simulator/whts_data.py:30-67` — `DropSimResult` dataclass에 `corner_impact_hist: Optional[list] = None` 필드 추가 (list 그대로 저장)
- `TVPackageMotionSim/run_drop_simulator/whts_engine.py:1505-1530` — 생성자 호출에 `corner_impact_hist=self.corner_impact_hist` 추가 (list 그대로 전달; 변환은 소비자 측에서)
- Phase 3의 소비 코드에서: `cih = np.asarray(result.corner_impact_hist)  # shape (N, 8)`

**회귀 테스트:**
- [ ] 기존 결과 로드(기존 pkl): `corner_impact_hist` 없는 pkl 로드 시 `getattr(result, 'corner_impact_hist', None)` 로 방어적 접근
- [ ] 새 시뮬레이션: `result.corner_impact_hist.shape == (N, 8)` 확인
- **Principle 3 예외 명시**: 이 수정은 `whts_engine.py`/`whts_data.py` 변경이지만 Contact Extraction의 전제 조건이므로 불가피. 기존 pkl backward 호환성은 `getattr` 방어 처리로 유지.

---

### Phase 1: whtb_doe Core Module

**Files to create:**
- `WHToolsBox/whtb_doe/__init__.py`
- `WHToolsBox/whtb_doe/dsl_parser.py`
- `WHToolsBox/whtb_doe/sampler.py`
- `WHToolsBox/whtb_doe/definition.py`
- `WHToolsBox/whtb_doe/tests/test_dsl_parser.py`
- `WHToolsBox/whtb_doe/tests/test_sampler.py`

**Step 1.1: DSL Parser (`dsl_parser.py`) — 합성 파서 (Option C)**

```python
# 지원 형식:
# cfg['a']['b'] = [10, 20, 30]          → discrete list    (literal_eval)
# cfg['a']['f'] = [10:199:15]           → start:end:count  (colon-regex)
# cfg['d'] = 10:200:100                 → min:max:init     (colon-regex)
# cfg['x'] = norm:10.0:2.0:50          → 정규분포          (norm-regex)
# cfg['s'] = ['A', 'B', 'C']           → string list      (literal_eval)
```

구현 전략 — 합성 파서:
1. 각 라인을 최상위 `=` 기준으로 LHS / RHS 분리
2. **LHS (key path)**: `ast.parse(lhs + ' = None', mode='exec')` → `ast.Assign.targets[0]`에서 `ast.Subscript` 체인 재귀 traverse → `['a', 'b']` key path 추출. AST는 여기서만 사용.
3. **RHS (value)** — 순서대로 시도:
   - `ast.literal_eval(rhs)` 성공 → list/string/number 처리 (대부분의 표준 Python 리터럴 커버)
   - regex `r'\[(\-?\d+\.?\d*):(\-?\d+\.?\d*):(\d+)\]'` 매칭 → `start:end:count` linspace
   - regex `r'^(\-?\d+\.?\d*):(\-?\d+\.?\d*):(\-?\d+\.?\d*)$'` → `min:max:init` 연속형
   - regex `r'^norm:(\-?\d+\.?\d*):(\-?\d+\.?\d*)(?::(\d+))?$'` → 정규분포
   - 모두 실패 → 파싱 오류 (줄 번호 포함 메시지)

`parse_dsl(dsl_string, base_config)` → `List[VarSpec]`
`validate_dsl(dsl_string, base_config)` → `(bool, str)`
- key path가 base_config에 존재하는지 확인
- value 형식 유효성 검사
- FullFact 예상 케이스 수 계산 → 1000 초과 시 경고 (hard cap 옵션 제공)

**VarSpec dataclass:**
```python
@dataclass
class VarSpec:
    key_path: List[str]       # ['a', 'b']
    var_name: str             # 'a.b' (DOE table 열 이름)
    var_type: str             # 'discrete_list', 'linspace', 'continuous', 'normal', 'string_list'
    values: Any               # 타입별 파라미터
    original_value: Any       # base_config에서의 원래 값
```

**Step 1.2: Sampler (`sampler.py`)**

```python
class DOESampler:
    def sample(self, var_specs: List[VarSpec], method: str,
               n_samples: int, seed: int) -> pd.DataFrame:
```

- `'lhs'`: `scipy.stats.qmc.LatinHypercube(d=n_vars, seed=seed).random(n_samples)` → [0,1] 샘플을 변수별로 스케일
  - **연속형** (`continuous`, `linspace`): `sample * (max - min) + min`
  - **이산형 리스트** (`discrete_list`, `string_list`): `int(sample * len(levels))` → levels 인덱스. 연속 [0,1]을 floor-index로 매핑
  - **정규분포** (`normal`): [0,1] 샘플을 `scipy.stats.norm.ppf(sample, loc=mean, scale=std)` 역변환 (QMC-정규분포 매핑)
- `'fullfact'`: `itertools.product(*level_lists)`
  - 연속형 → `np.linspace(min, max, n_levels)` (n_levels 기본 5, 또는 `[a:b:c]`의 c)
  - 이산형/문자열 → 레벨 목록 그대로
  - 정규분포 → `fullfact` 시 `norm:m:s:n`의 n을 레벨 수로 사용, `np.linspace(m-3σ, m+3σ, n)`
- `'montecarlo'`: `rng.uniform(0,1)` → 각 타입별 동일한 스케일 로직 적용 (LHS와 동일, 단 stratified 없음)
- 반환: `pd.DataFrame` — 열: `case_number, var_name1, var_name2, ...`

**Step 1.3: DOEDefinition class (`definition.py`)**

```python
class DOEDefinition:
    def __init__(self, dsl_string: str, base_config: dict,
                 output_dir: str = None):
        ...

    def validate(self) -> Tuple[bool, str]:
        ...

    def generate(self, method: str = 'lhs',
                 n_samples: int = 100,
                 seed: int = 42) -> Tuple[pd.DataFrame, List[dict]]:
        # returns (doe_table, config_list)
        ...

    def regenerate(self, modified_doe_table: pd.DataFrame
                   ) -> Tuple[pd.DataFrame, List[dict]]:
        # modified_doe_table의 값을 그대로 사용해 config_list 재생성
        ...

    def set_output_dir(self, output_dir: str): ...
```

내부:
- `generate()`: `validate()` → `DOESampler.sample()` → config_list 생성
  - config_list: `[copy.deepcopy(base_config)]` × n, 각 케이스에 VarSpec key_path로 값 주입
- `regenerate(modified_doe_table)`: modified_doe_table의 각 행을 직접 config_list에 매핑 (새로 샘플링 없이)

---

### Phase 2: DOE Execution Engine

**Files to create:**
- `TVPackageMotionSim/run_doe_runner.py`
- `TVPackageMotionSim/run_doe_case_worker.py`  ← subprocess 진입점

**Step 2.1: Case Worker 진입점 — PyInstaller 호환 설계**

PyInstaller 동결 빌드에서 `python run_doe_case_worker.py` 방식은 `.py` 파일이 없고 `python` 인터프리터도 없어서 작동 안 함. 대신 `sys.executable` + 모드 플래그 방식 사용:

```python
# DOERunner에서 worker 스폰:
import sys, subprocess
worker_args = [
    sys.executable,          # 동결 빌드: 자기 자신의 실행파일. dev: python
    '--doe-worker',          # 모드 플래그 (argparse로 감지)
    '--batch-start', str(batch_start_idx),
    '--batch-size', str(K),
    '--run-dir', run_dir,
    '--slot-id', str(slot_id),
]
proc = subprocess.Popen(worker_args, ...)
```

```python
# run_drop_simulation_cases_v6.py 상단 (PyInstaller Analysis 진입점, drop_simulator_v6.spec:90):
# 반드시 JAX/PySide6/MuJoCo import 이전에 위치해야 cold-start 최소화
import sys
if '--doe-worker' in sys.argv:
    from run_doe_worker_impl import doe_worker_main
    doe_worker_main()
    sys.exit(0)
# ... 이하 기존 import 및 코드 ...
```

> **Critic 발견**: PyInstaller frozen 진입점은 `run_drop_simulation_cases_v6.py` (`drop_simulator_v6.spec:90` → `Analysis(['run_drop_simulation_cases_v6.py'])`). `run_drop_simulator/__main__.py`는 frozen exe에서 argv를 받지 않으므로 여기에 라우팅하면 안 됨. 반드시 `v6.py` 상단에 삽입.

**PyInstaller spec 추가 필요:**
```python
# drop_simulator_v6.spec 수정:
hiddenimports=['run_doe_worker_impl', 'whtb_doe', 'whtb_doe.dsl_parser',
               'whtb_doe.sampler', 'whtb_doe.definition']
```

동작 (warm-batch, K cases/process):
1. `--run-dir`에서 `batch_start` ~ `batch_start+K` 케이스의 config.pkl 순차 로드
2. 각 케이스: `DropSimulator(config=cfg).simulate()` 실행
3. 매 reporting_interval마다 `{run_dir}/monitor/slot_{slot_id}.json` 업데이트: `{pid, case_idx, target_time, current_time, frame}`
4. 케이스 완료 시 result pkl 저장 → `{run_dir}/case_{n:03d}/result.pkl`
5. K케이스 모두 완료 후 정상 종료 (returncode=0)
6. 어떤 케이스에서 예외 발생 시: 오류 로그 저장, returncode=1 종료 → 큐 매니저가 재스폰

**sys.path 주입 (동결/개발 양쪽 호환):**
```python
import sys, os
if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
else:
    base = os.path.dirname(os.path.abspath(__file__))
    # WHToolsBox 루트까지 올라가기
    base = os.path.dirname(os.path.dirname(base))
if base not in sys.path:
    sys.path.insert(0, base)
```

**Step 2.2: DOE Runner (`run_doe_runner.py`) — warm-batch 큐 매니저**

```python
class DOERunner:
    def __init__(self, doe_definition: DOEDefinition,
                 n_parallel: int = 4,
                 output_root: str = None,
                 camera_angle: Tuple[float, float] = None):
        ...

    def run(self, doe_table: pd.DataFrame,
            config_list: List[dict]) -> str:
        # output_dir 생성: results/DOE_D{날짜}_{시간}/
        # case별 config.pkl 저장
        # subprocess 큐 실행
        # run_doe_log.txt 모니터링 루프
        # 반환: output_dir 경로
        ...
```

**warm-batch subprocess 큐 매니저:**
```
K = batch_size (기본 10)  ← 1 worker가 처리할 케이스 수
slots: [None] × n_parallel  ← 실행 중인 worker subprocess
queue: [(0,K), (K,2K), ...]  ← batch 단위 대기열 (start_idx, size)

loop:
  1. 완료된 slot 확인 (process.poll() is not None)
  2a. returncode == 0: 완료 batch를 done_doe_log.txt에 append
  2b. returncode != 0: 실패 케이스 기록, 해당 batch를 재시도 큐에 추가 (최대 1회 재시도)
  3. 빈 slot에 다음 batch subprocess 스폰 (sys.executable --doe-worker ...)
  4. run_doe_log.txt 생성 (전체 덮어쓰기):
     각 slot의 monitor/slot_{id}.json 읽어서 4열 표시
     실행 중 slot만 표시, 빈 slot은 "---"
  5. 0.5초 sleep 후 반복
```

**run_doe_log.txt 형식:**
```
=== DOE RUN MONITOR [2026-06-20 15:30:01] ===
SLOT | PID   | TARGET_T | CURRENT_T | FRAME
  0  | 12345 |   2.000s |    0.450s |   375
  1  | 12346 |   2.000s |    1.200s |  1000
  2  |  ---  |    ---   |     ---   |   ---
  3  | 12348 |   2.000s |    0.100s |    83
```

**done_doe_log.txt 형식 (append):**
```
[DONE] case_003 | PID 12345 | elapsed 45.2s | 2026-06-20 15:30:46
```

---

### Phase 3: Drop Contact Extraction

**Files to create:**
- `TVPackageMotionSim/whts_contact_extractor.py`

**알고리즘:**
```python
def extract_contact_sequence(result) -> dict:
    """
    Parameters: result — DropSimulator.result 인스턴스
    Returns: ContactSeqList dict
    """
```

**전제 조건 (Phase 0 완료 필수):**
`result.corner_impact_hist` — Phase 0에서 `DropSimResult`에 추가됨 (list 타입).
기존 pkl 방어: `raw = getattr(result, 'corner_impact_hist', None)`이 None이면 명확한 오류 메시지 출력 후 조기 반환.

**접촉 감지 로직:**
```
raw = getattr(result, 'corner_impact_hist', None)
if raw is None:
    raise ValueError("corner_impact_hist 없음 — Phase 0 적용 후 재시뮬레이션 필요")
cih = np.asarray(raw)  # List[ndarray] → ndarray, shape (N, 8)
# 참조: run_drop_simulation_cases_doe.py:130 의 np.array() 패턴

for each corner C_i (i=0..7):
    force_series = cih[:, i]     # shape (N,)
    first_contact_frame = first index where force_series > CONTACT_THRESHOLD (e.g. 1.0 N)
    if first_contact_frame is None: skip
    contact_time = result.time_history[first_contact_frame - 1]  # 직전 프레임
    peak_window = force_series[first_contact_frame : first_contact_frame + PEAK_WINDOW]
    contact_peak_force = max(peak_window)
    # 접촉 유지 구간: force > CONTACT_THRESHOLD 연속 구간
    contact_end = first_contact_frame + next index where force drops below threshold
    dt = result.time_history[1] - result.time_history[0]
    impulse = np.trapz(force_series[first_contact_frame:contact_end], dx=dt)
```

**ContactSeqList 구조:**
```python
ContactSeqList = {
    'ContactSeqStr': 'C4-C3-C2-C1',
    'ContactSeqTimeList':   [['C4', 10.1], ['C3', 10.3], ...],
    'ContactSeqForceList':  [['C4', 245.3], ['C3', 180.1], ...],   # N
    'ContactSeqImpactList': [['C4', 12.5], ['C3', 8.2], ...],      # N·s
}
```

**추가 기능:**
- `result.contact_seq = ContactSeqList` 속성 추가 후 pkl 재저장
- 각 접촉 직전 프레임에서 MuJoCo 오프스크린 렌더 → `{result_dir}/contacts/C4_pre.png`
  - 렌더 방식: `whts_engine.py`의 기존 캡처 패턴 참조 (`mujoco.Renderer`)
- **독립성**: `TVPackageMotionSim/run_drop_simulator/` 내 파일 import 최소화 — `result` 인스턴스만 인자로 받음

**주의사항 (고민사항 처리):**
- 튀는 현상: `first_contact_frame` = 최초 접촉만 기준 (이후 반복 접촉 무시)
- 여러 코너가 같은 프레임에 접촉: 접촉력이 큰 코너를 먼저 기록

---

### Phase 4: DOE UI Widget

**Files to create:**
- `WHToolsBox/whtb_doe/doe_widget.py`

**클래스 구조:**
```python
class DOESetupDialog(QDialog):
    """호출: table, config_list = DOESetupDialog.run_dialog(base_config, parent)"""

    def __init__(self, base_config: dict, parent=None): ...

    @classmethod
    def run_dialog(cls, base_config, parent=None
                   ) -> Optional[Tuple[pd.DataFrame, List[dict]]]:
        dlg = cls(base_config, parent)
        if dlg.exec() == QDialog.Accepted:
            return dlg.doe_table, dlg.config_list
        return None
```

**UI 구성:**
```
┌─────────────────────────────────────────────────────┐
│  DOE Setup                                          │
├────────────────────────┬────────────────────────────┤
│  Variables             │  DOE Settings              │
│  [Add Variable]        │  Method: [LHS ▼]           │
│  ┌──────┬────────────┐ │  N samples: [100]          │
│  │ Path │ Definition │ │  Seed: [42]                │
│  │ a.b  │ [10,20,30] │ │                            │
│  │ d    │ 10:200:100 │ │  [Validate] [Make DOE List]│
│  └──────┴────────────┘ │                            │
├────────────────────────┴────────────────────────────┤
│  DOE Table (editable QTableWidget)                  │
│  case | a.b | d  |                                  │
│   0   |  10 | 50 |                                  │
│   1   |  30 | 75 |                                  │
├─────────────────────────────────────────────────────┤
│                         [Cancel]  [OK]              │
└─────────────────────────────────────────────────────┘
```

**Add Variable 트리:**
- `QTreeWidget`으로 `base_config` dict를 재귀 탐색하여 leaf 노드 표시
- 더블클릭 → 변수 목록에 추가

**유효성 검사:**
- "Validate" 버튼 → `DOEDefinition(dsl, base_config).validate()` → 결과를 상태 레이블로 표시
- DOE table 편집 후 OK → "변경 사항을 적용하시겠습니까?" QMessageBox → `regenerate(modified_table)`

---

### Phase 5: DOE Results Analysis

**Files to create:**
- `TVPackageMotionSim/run_doe_analysis.py`
- `TVPackageMotionSim/run_doe_analysis_widget.py` (PySide6 메인 UI)

**Step 5.1: 데이터 로드 및 그룹핑 (`run_doe_analysis.py`)**

```python
def load_doe_results(doe_output_dir: str) -> List[dict]:
    """DOE 결과 폴더 스캔, case별 result pkl 로드"""
    # DOE_D*/case_*/result_*.pkl 패턴으로 glob
    # result.contact_seq['ContactSeqStr'] 기준 그룹핑

def plot_group_bar(results: List[dict], output_dir: str):
    """그룹별/코너별 최대 접촉력 bar chart (matplotlib)"""
    # x축: 그룹(ContactSeqStr), 색상: 코너
    # 에러 바: 1σ
```

**Step 5.2: Gemini CLI 채팅 창 (`run_doe_analysis_widget.py`)**

```python
class GeminiDOEChatDialog(QDialog):
    """Gemini CLI 기반 DOE 결과 분석 채팅 창"""
```

구현:
- 초기화 시 결과 요약 컨텍스트 생성:
  ```
  # DOE Results Summary
  - total_cases: N
  - groups: {'C4-C3-C2-C1': 45 cases, ...}
  - pkl fields: corner_pos_hist shape (F,8,3), contact_seq keys ...
  - pkl load: import pickle; r = pickle.load(open('case_003/result_*.pkl','rb'))
  ```
- 사용자 입력 → `gemini` CLI를 **QThread**에서 호출 (UI 스레드 블록 방지):
  ```python
  class GeminiWorker(QThread):
      result_ready = Signal(str)
      def __init__(self, context, user_input): ...
      def run(self):
          proc = subprocess.run(
              ['gemini', '-p', f"{context}\n\nUser: {self.user_input}"],
              capture_output=True, text=True, encoding='utf-8'
          )
          self.result_ready.emit(proc.stdout)
  # GeminiDOEChatDialog에서:
  # self._worker = GeminiWorker(context, user_input)
  # self._worker.result_ready.connect(self._on_response)
  # self._worker.start()
  ```
- 응답에서 Python 코드 블록 감지 (```` ```python ... ``` ````) → "Run this code" 버튼 표시
- 코드 실행: "Run this code" 버튼 클릭 전 코드 내용 표시 → 사용자 확인 후 실행
  - `exec(code, {'results': results, 'plt': plt, 'np': np})` — **완전한 사용자 권한으로 실행됨** (sandbox 아님)
  - 로컬 단일 사용자 도구이므로 허용 가능한 수준. 단, UI에 "이 코드는 전체 사용자 권한으로 실행됩니다" 경고 레이블 표시
- Gemini CLI 경로: `whtb_doe_config.json`에 저장된 경로 우선, 없으면 `PATH` 탐색
- Gemini 없을 때 graceful 저하: 채팅 창 열기 시 "gemini CLI를 찾을 수 없습니다. PATH 확인 또는 설정 파일에 경로 지정" 메시지 표시 후 비활성화
- PyInstaller: gemini 바이너리는 번들에 포함하지 않음 (라이선스/재배포 문제). 별도 설치 안내.

---

## Acceptance Criteria

### whtb_doe Core
- [ ] `from whtb_doe import DOEDefinition` 작동 (WHToolsBox 루트 `sys.path` 추가 시)
- [ ] `cfg['a']['b'] = [10,20,30]` → 이산형 3레벨 DOE table 생성
- [ ] `cfg['a']['f'] = [10:199:15]` → 15개 등간격 연속값
- [ ] `cfg['d'] = 10:200:100` → 연속형 변수 (init=100은 base_config 값으로 사용)
- [ ] `norm:10.0:2.0:50` → 정규분포 50샘플
- [ ] `validate()` — 존재하지 않는 key path에서 줄 번호 포함 오류 메시지
- [ ] `generate('lhs', 30, seed=42)` → (30행 DataFrame, 30개 config dict) 반환
- [ ] `generate('fullfact')` → 모든 이산 조합의 케이스 수 정확
- [ ] `regenerate(modified_df)` → 수정된 값이 config_list에 반영됨
- [ ] `scipy` 외 외부 라이브러리 의존 없음

### Phase 0: DropSimResult 스키마 확장
- [ ] `whts_data.py` — `DropSimResult` dataclass에 `corner_impact_hist: Optional[np.ndarray] = None` 추가
- [ ] `whts_engine.py:1505-1530` — 생성자 호출에 `corner_impact_hist=self.corner_impact_hist` 추가
- [ ] 신규 시뮬레이션 result pkl 로드 시 `np.asarray(result.corner_impact_hist).shape == (N, 8)` 확인 (list→ndarray 변환)
- [ ] 기존 pkl 로드 시 `getattr(result, 'corner_impact_hist', None)` → None 반환 (crash 없음)

### DOE Execution Engine
- [ ] `DOERunner(doe_def, n_parallel=4, batch_size=10).run(table, configs)` — 4개 warm-batch worker 병렬 시작
- [ ] `results/DOE_D{날짜}_{시간}/case_{n:03d}/` 폴더 구조 생성
- [ ] worker 스폰: `sys.executable --doe-worker ...` (PyInstaller 동결 빌드 및 dev 환경 양쪽 동작)
- [ ] `run_doe_log.txt` — 실행 중 slot만 표시, 0.5초 주기 갱신
- [ ] `done_doe_log.txt` — 완료 batch append
- [ ] worker returncode != 0 시 해당 batch 1회 재시도, 나머지 계속 실행
- [ ] monitor JSON 쓰기: temp 파일 → atomic rename (`os.replace`) 으로 partial-read 방지
- [ ] `camera_angle` 설정 시 case별 mp4 생성

### Contact Extraction
- [ ] Phase 0 완료 후 진행 (corner_impact_hist 필드 존재 전제)
- [ ] `extract_contact_sequence(result)` — ContactSeqList dict 반환
- [ ] `result.corner_impact_hist` 없으면 명확한 오류 메시지 + 조기 반환
- [ ] `ContactSeqStr` 형식 `'C4-C3-C1'` (접촉 순서대로)
- [ ] 튀는 현상: 최초 접촉 프레임만 기록
- [ ] `result.contact_seq` 속성 추가 후 `result.pkl` 재저장 확인
- [ ] `contacts/C4_pre.png` 등 캡처 파일 생성
- [ ] 독립 import: `python -c "from whts_contact_extractor import extract_contact_sequence"` 성공

### DOE UI Widget
- [ ] `DOESetupDialog.run_dialog(base_config)` — None 또는 `(df, configs)` 반환
- [ ] Add Variable 트리에서 선택 후 변수 목록 추가
- [ ] Validate 버튼 → 성공/실패 메시지 표시
- [ ] DOE table 편집 후 OK → 수정된 값이 config_list에 반영
- [ ] whts_control_panel.py에서 호출 테스트 통과

### Results Analysis
- [ ] `load_doe_results('DOE_D.../') ` → case별 result dict 리스트
- [ ] `ContactSeqStr` 기준 그룹핑 정확
- [ ] bar chart — 그룹별/코너별 최대 접촉력 + 1σ 에러 바
- [ ] Gemini CLI 채팅 창 — 사용자 입력 전송 → 응답 표시
- [ ] Python 코드 블록 감지 → "Run this code" 버튼 표시 및 실행

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|---------|-----------|
| JAX+MuJoCo cold-start 오버헤드 (case당 수~수십 초) | High | warm-batch worker (K=10): 초기화 비용 K로 상각. Phase 2 전 단일 케이스 cold-start 실측 후 K 조정 |
| `corner_impact_hist` pkl에 없음 (기존 pkl 호환) | High | Phase 0에서 `DropSimResult` 확장. 기존 pkl은 `getattr(..., None)` 방어 처리 + 명확한 오류 안내 |
| PyInstaller 동결 빌드에서 worker 스폰 실패 | High | `sys.executable --doe-worker` 패턴으로 설계. dev/frozen 양쪽 테스트 필수 |
| DSL `[10:199:15]` 비표준 syntax — AST SyntaxError | Resolved | 합성 파서(Option C): RHS는 `literal_eval` → colon-regex fallback, AST는 LHS key path 전용 |
| Gemini CLI 없을 때 Phase 5 전체 비활성 | Medium | graceful 저하: CLI 없으면 채팅 창 비활성화 + 설치 안내. 나머지 분석 기능(그래프)은 정상 동작 |
| `exec()` 사용 — 의도치 않은 파일/프로세스 접근 | Medium | 로컬 단일 사용자 도구; sandbox 없음을 UI에 명시. 코드 실행 전 미리보기 + 사용자 확인 버튼 |
| FullFact 케이스 수 폭발 | Low | `validate()` — 예상 케이스 수 계산, >1000 케이스 시 경고 + 사용자 확인 요구 |

---

## Verification Steps

1. **Unit tests** (각 Phase 완료 시):
   - `python -m pytest whtb_doe/tests/ -v`
   - `python -c "from whtb_doe import DOEDefinition; d = DOEDefinition(\"cfg['x'] = [1,2,3]\", {'x':1}); print(d.validate())"`

2. **Integration test** (Phase 2 완료 시):
   - `python run_doe_runner.py` — 2케이스 × 1 parallel로 실제 실행 후 결과 폴더 확인

3. **Contact extraction** (Phase 3):
   - 기존 result pkl 로드 → `extract_contact_sequence(result)` 실행 → ContactSeqStr 출력 확인

4. **UI smoke test** (Phase 4):
   - `python -c "from whtb_doe.doe_widget import DOESetupDialog; ..."`
   - Control panel에서 DOE 버튼 클릭 → 다이얼로그 표시 확인

5. **Gemini 채팅** (Phase 5):
   - `gemini --version` 확인 후 채팅 창 실행
   - "Show groups" 입력 → Python 코드 생성 확인

---

## ADR (Architecture Decision Record)

**Decision:** 합성 DSL 파서(literal_eval+colon-regex for value / AST for key path) + warm-batch subprocess worker 병렬 실행 + Phase 0 DropSimResult 스키마 확장

**Drivers:**
- Windows 환경에서 MuJoCo 프로세스 안정성 (subprocess 격리)
- 추가 라이브러리 의존성 최소화 (stdlib + scipy)
- PyInstaller 동결 배포 호환성 (`sys.executable --doe-worker` 패턴)
- 100+ 케이스 DOE에서 JAX cold-start 비용 상각 필요

**Alternatives Considered:**
- AST 전용 파서: `[10:199:15]` 등 비표준 value에서 `SyntaxError`, 결국 세 개의 파서 필요
- regex 전용 파서: 가변 깊이 중첩 dict key path 처리 취약
- subprocess 1case/프로세스: JAX+MuJoCo cold-start × N회 → 100케이스에서 수천 초 낭비
- multiprocessing.Pool: Windows+MuJoCo+PySide6 fork 불안정

**Why Chosen:**
- **합성 파서**: AST는 key path 추출에서 강점(가변 깊이 Subscript 체인). literal_eval은 표준 Python 리터럴 무료 처리. colon-regex는 비표준 value 형식 전담. 각 도구를 강점 영역에만 사용.
- **warm-batch worker**: K=10으로 cold-start 비용 1/10 상각하면서 프로세스 격리 유지. crash 시 해당 batch만 재시도.
- **Phase 0 선행**: `corner_impact_hist`가 pkl에 없으면 Phase 3가 런타임에 실패. 스키마 확장이 Contact Extraction의 전제 조건.

**Consequences:**
- Principle 3("기존 코드 동결") 예외: `whts_engine.py`, `whts_data.py` 최소 수정 불가피. 기존 pkl backward 호환 (`Optional` 필드, `getattr` 방어)으로 회귀 최소화.
- subprocess IPC 없음 → 진행률은 파일 기반 모니터링 (`monitor/slot_{id}.json`)
- exec() 비샌드박스 — UI 경고 + 사용자 확인으로 완화

**Follow-ups:**
- Phase 2 전: 단일 케이스 cold-start 실측 → K 값 조정 (기본 K=10)
- Phase 5 전: Gemini CLI 설치 요구사항 배포 문서화
- FullFact hard cap 값 결정 (현재: >1000 경고, 사용자 확인 요구)

---

## File Structure Summary

```
WHToolsBox/
├── whtb_doe/
│   ├── __init__.py              # DOEDefinition export
│   ├── dsl_parser.py            # 합성 파서: AST(key path) + literal_eval+regex(value)
│   ├── sampler.py               # DOESampler (LHS/FullFact/MonteCarlo, scipy.stats.qmc)
│   ├── definition.py            # DOEDefinition class
│   ├── doe_widget.py            # PySide6 DOESetupDialog
│   └── tests/
│       ├── test_dsl_parser.py
│       └── test_sampler.py
│
TVPackageMotionSim/
├── run_drop_simulator/
│   ├── whts_data.py             # [Phase 0] corner_impact_hist 필드 추가
│   └── whts_engine.py           # [Phase 0] DropSimResult 생성자에 corner_impact_hist 전달
├── run_doe_runner.py            # DOERunner + warm-batch subprocess 큐 매니저
├── run_doe_worker_impl.py       # doe_worker_main() — K케이스 배치 실행 로직
├── whts_contact_extractor.py    # extract_contact_sequence() — 독립 파일
├── run_doe_analysis.py          # load_doe_results(), plot_group_bar()
└── run_doe_analysis_widget.py   # GeminiDOEChatDialog (PySide6)

# __main__.py 또는 메인 진입점:
# if '--doe-worker' in sys.argv: doe_worker_main(); sys.exit(0)
```

## Changelog
- **Planner initial draft**: 5개 컴포넌트 구현 계획, AST 파서, subprocess 1case/프로세스
- **Architect review → REQUEST CHANGES**: 3개 blocking 이슈 발견
  1. `corner_contact_hist` 존재하지 않음 → Phase 0 추가, `corner_impact_hist` 사용
  2. AST 파서가 비표준 value 구문 파싱 불가 → 합성 파서(Option C)로 교체
  3. subprocess cold-start 과도 → warm-batch worker(Option F, K=10)로 교체
  4. `exec()` "sandbox" 주장 제거, UI 경고로 대체
  5. PyInstaller worker: `sys.executable --doe-worker` 패턴으로 재설계
- **Revised (Architect)**: 위 5개 사항 반영
- **Critic review → REVISE**: 추가 3개 major 이슈 발견
  1. frozen 진입점 오류 → `run_drop_simulation_cases_v6.py` 상단에 `--doe-worker` 라우팅, PyInstaller spec `hiddenimports` 추가
  2. `corner_impact_hist`가 List → `np.asarray()` 변환, dataclass 타입 `Optional[list]`으로 수정
  3. 기존 DOE 선행 코드 관계 미설명 → "Prior Art & Relationship" 섹션 추가
  - Minor: Gemini CLI 동기 호출 → QThread 비동기 처리, discrete-var LHS 매핑 명세, monitor JSON atomic write 추가, 중복 File Structure 블록 제거
- **Revised (Critic)**: 위 모든 사항 반영 → 최종 승인 대기
