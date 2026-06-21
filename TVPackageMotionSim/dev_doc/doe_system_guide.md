# WHToolsBox DOE System — 구현 가이드

> 작성일: 2026-06-20  
> 구현 버전: v1.0  
> 관련 스펙: `.omc/specs/deep-interview-whtb-doe-system.md`

---

## 개요

WHToolsBox에 범용 **DOE(Design of Experiments, 실험계획법)** 시스템을 추가했다.  
핵심 패키지(`whtb_doe`)는 TVPackageMotionSim과 독립적으로 import 가능하며,  
TVPackageMotionSim에서는 이 패키지를 이용해 낙하 시뮬레이션 DOE 파이프라인을 실행한다.

```
WHToolsBox/
├── whtb_doe/               ← 범용 DOE 패키지 (신규)
│   ├── dsl_parser.py
│   ├── sampler.py
│   ├── definition.py
│   ├── doe_widget.py
│   └── tests/
└── TVPackageMotionSim/
    ├── whts_contact_extractor.py   ← 낙하 접촉 추출 (신규)
    ├── run_doe_runner.py           ← DOE 병렬 실행 엔진 (신규)
    ├── run_doe_worker_impl.py      ← subprocess 배치 워커 (신규)
    ├── run_doe_analysis.py         ← 결과 분석 (신규)
    └── run_doe_analysis_widget.py  ← Gemini CLI 채팅 UI (신규)
```

---

## 1. whtb_doe 패키지

### 1-1. DSL 문법

`DOEDefinition`은 다음 형식의 문자열로 파라미터 공간을 정의한다.

```python
# 이산형 리스트
cfg['a']['b'] = [10, 20, 30]

# 등간격 (start:end:count) — 15개 값
cfg['a']['f'] = [10:199:15]

# 연속형 (min:max:init)
cfg['d'] = 10:200:100

# 정규분포 (norm:mean:std:n) — n 생략 가능
cfg['mass'] = norm:1.5:0.2:50

# 문자열 이산형
cfg['material'] = ['foam_A', 'foam_B', 'foam_C']
```

| 형식 | var_type | 비고 |
|------|----------|------|
| `[v1, v2, v3]` | `discrete_list` | 정수/실수 이산값 |
| `['A', 'B']` | `string_list` | 문자열 이산값 |
| `[s:e:n]` | `linspace` | n개 등간격 |
| `min:max:init` | `continuous` | LHS/MC 연속 샘플링 |
| `norm:mean:std:n` | `normal` | 정규분포, n 생략 시 기본 10 |

### 1-2. DOEDefinition 사용법

```python
from whtb_doe import DOEDefinition

# base_config: 시뮬레이션 기본 설정 dict
base_config = {
    'drop_height': 1.0,
    'material': {'density': 1200, 'stiffness': 5000},
    'mass': 1.5,
}

dsl = """
cfg['drop_height'] = [0.5, 1.0, 1.5]
cfg['material']['stiffness'] = [3000:8000:6]
cfg['mass'] = norm:1.5:0.2:20
"""

doe = DOEDefinition(dsl, base_config)

# 유효성 검사
ok, msg = doe.validate()
print(ok, msg)

# DOE 생성
#   method: 'lhs' (기본) | 'fullfact' | 'montecarlo'
#   n_samples: LHS/MC 케이스 수 (fullfact는 무시)
#   seed: 재현성을 위한 난수 시드
doe_table, config_list = doe.generate(method='lhs', n_samples=100, seed=42)

# doe_table: pandas DataFrame
#   columns = case_number, drop_height, material.stiffness, mass
# config_list: List[dict] — 각 행에 해당하는 완성된 config

# DOE table 수동 수정 후 재생성
doe_table.loc[0, 'mass'] = 1.8
_, new_configs = doe.regenerate(doe_table)
```

### 1-3. DOE UI Widget (PySide6)

```python
from whtb_doe.doe_widget import DOESetupDialog

# 다이얼로그 실행 — (doe_table, config_list) 반환, 취소 시 None
result = DOESetupDialog.run_dialog(base_config, parent=None)
if result:
    doe_table, config_list = result
```

UI 조작 순서:
1. **Add Variable** 클릭 → config tree에서 변수 선택
2. Definition 열에 DSL 값 입력 (`[10,20,30]`, `[s:e:n]`, `norm:…` 등)
3. **Validate** 클릭 → 오류 위치 확인
4. Method / N samples / Seed 설정
5. **Make DOE List** 클릭 → DOE table 확인 및 직접 편집
6. **OK** → `(doe_table, config_list)` 반환

---

## 2. DOE 실행 엔진

### 2-1. DOERunner 사용법

```python
from whtb_doe import DOEDefinition
from run_doe_runner import DOERunner

doe = DOEDefinition(dsl, base_config)
doe_table, config_list = doe.generate(method='lhs', n_samples=50)

runner = DOERunner(
    n_parallel=4,       # 동시 실행 subprocess 수
    batch_size=10,      # 워커당 처리 케이스 수 (JAX cold-start 분산)
    output_root='results',
)

run_dir = runner.run(doe_table, config_list)
# 결과: results/DOE_D20260620_143000/
#   ├── case_000/ config.pkl, result.pkl
#   ├── case_001/ ...
#   ├── run_doe_log.txt   (실행 중 상태, 매 주기 덮어쓰기)
#   └── done_doe_log.txt  (완료 케이스 append)
```

### 2-2. 실행 구조

```
DOERunner.run()
  └── subprocess × n_parallel
        └── python run_drop_simulation_cases_v6.py --doe-worker
              --batch-start N --batch-size K --run-dir <path> --slot-id S
              └── doe_worker_main()  (run_doe_worker_impl.py)
                    └── DropSimulator(config).simulate() × K cases
```

- **warm-batch**: 워커 하나가 K=10 케이스를 연속 실행 → JAX/MuJoCo cold-start 비용 분산
- **모니터**: `monitor/slot_{id}.json` atomic write (`os.replace`) → 부분 읽기 방지
- **재시도**: 실패 시 1회 자동 재시도, 2회 실패 시 스킵 + done_doe_log에 FAILED 기록
- **`run_doe_log.txt`** 포맷:

```
=== DOE RUN MONITOR [2026-06-20 14:30:05] ===
SLOT |    PID | TARGET_T | CURRENT_T |  FRAME | STATUS
  0  | 12345  |   2.000s |    1.234s |   1234 | running
  1  | 12346  |   2.000s |    0.567s |    567 | running
  2  |    --- |      --- |       --- |    --- | idle
  3  | 12347  |   2.000s |    2.000s |   2000 | done
```

---

## 3. 낙하 접촉 추출 (whts_contact_extractor)

### 사용법

```python
from whts_contact_extractor import extract_contact_sequence

# result: DropSimResult 인스턴스 (corner_impact_hist 필드 필요)
contact_seq = extract_contact_sequence(result)

# 반환값 예시
# {
#   'ContactSeqStr':      'C4-C3-C2-C1',
#   'ContactSeqTimeList': [['C4', 0.123], ['C3', 0.145], ...],
#   'ContactSeqForceList':[['C4', 85.3],  ['C3', 120.1], ...],
#   'ContactSeqImpactList':[['C4', 0.42], ['C3', 0.67], ...],
# }

# result.contact_seq 에도 자동 저장되고 pkl 재저장됨
# result.output_dir/contacts/ 에 접촉 직전 프레임 캡처 저장 (MuJoCo 렌더 가능 시)
```

### 알고리즘

1. `corner_impact_hist` (N_frames × 8 ndarray)에서 각 코너별 첫 접촉 탐지
   - 임계값: `CONTACT_THRESHOLD = 1.0 N`
   - 튀는 현상 처리: 시뮬레이션 전체에서 **최초** 접촉 기준
2. 접촉 직전 프레임 시간 → `ContactSeqTimeList`
3. 접촉 후 50 프레임 구간 최대 힘 → `ContactSeqForceList`
4. 접촉 유지 구간 `trapz` 적분 → `ContactSeqImpactList` (impulse)
5. 접촉 순서 문자열 생성: `C4-C3-C2-C1`

### 전제 조건

`DropSimResult`에 `corner_impact_hist` 필드가 있어야 한다.  
`whts_data.py`의 `DropSimResult` dataclass에 이미 추가되어 있다:

```python
# whts_data.py
corner_impact_hist: Optional[list] = None
```

`whts_engine.py`의 `DropSimResult(...)` 생성자에도 전달된다:

```python
corner_impact_hist=self.corner_impact_hist
```

---

## 4. DOE 결과 분석

### 4-1. 스크립트 분석

```python
from run_doe_analysis import load_doe_results, group_by_contact_seq, plot_group_bar

# DOE 결과 폴더 스캔 → result pkl 로드
results = load_doe_results('results/DOE_D20260620_143000')

# ContactSeqStr 기준 그룹핑
groups = group_by_contact_seq(results)
# {'C4-C3-C2-C1': [...], 'C3-C4-C2-C1': [...], ...}

# 그룹별/코너별 최대 접촉력 막대 그래프 (1σ 오차 막대 포함)
plot_group_bar(results, output_dir='results/charts', show=True)
# → results/charts/doe_group_bar.png 저장
```

### 4-2. Gemini CLI 채팅 UI

```python
from run_doe_analysis_widget import open_doe_analysis

# DOE 결과 폴더를 인자로 → 채팅 다이얼로그 실행
open_doe_analysis('results/DOE_D20260620_143000', parent=None)
```

UI 기능:
- pkl 구조 요약 자동 context 첨부
- 사용자 질문 입력 → `gemini -p "..."` CLI 호출 (QThread, UI 비차단)
- Python 코드 블록 자동 감지 → **"Run this code"** 버튼 제공
  - 실행 전 코드 미리보기 표시
  - ⚠ 사용자 전체 권한으로 실행 (검토 후 클릭)

Gemini CLI 설정:
- PATH에 `gemini` 실행파일이 있으면 자동 감지
- 없으면 `whtb_doe_config.json`의 `gemini_path` 키 참조

```json
{ "gemini_path": "C:/Users/.../gemini.exe" }
```

---

## 5. 전체 워크플로 예시

```python
# 1. DOE 정의
from whtb_doe import DOEDefinition

dsl = """
cfg['drop_height'] = [0.5, 0.9, 1.2]
cfg['tv']['mass'] = norm:25.0:2.0:30
cfg['cushion']['stiffness'] = [3000:8000:5]
"""
base_config = load_base_config('my_tv_config.json')
doe = DOEDefinition(dsl, base_config, output_dir='results')

# 2. (선택) UI로 검토 및 편집
from whtb_doe.doe_widget import DOESetupDialog
result = DOESetupDialog.run_dialog(base_config)
if result:
    doe_table, config_list = result
else:
    doe_table, config_list = doe.generate(method='lhs', n_samples=50, seed=0)

# 3. DOE 실행
from run_doe_runner import DOERunner
runner = DOERunner(n_parallel=4, batch_size=10, output_root='results')
run_dir = runner.run(doe_table, config_list)

# 4. 접촉 추출 (실행 후 개별 케이스)
import pickle
from whts_contact_extractor import extract_contact_sequence
from pathlib import Path

for case_dir in sorted(Path(run_dir).glob('case_*')):
    pkl = case_dir / 'result.pkl'
    if pkl.exists():
        result = pickle.load(open(pkl, 'rb'))
        extract_contact_sequence(result)  # result.pkl 재저장 포함

# 5. 결과 분석
from run_doe_analysis import load_doe_results, plot_group_bar
results = load_doe_results(run_dir)
plot_group_bar(results, output_dir=run_dir + '/charts')

# 6. Gemini 채팅 분석
from run_doe_analysis_widget import open_doe_analysis
open_doe_analysis(run_dir)
```

---

## 6. 의존성

| 패키지 | 용도 | 비고 |
|--------|------|------|
| `scipy.stats.qmc` | LHS 샘플링 | vdmc 환경에 설치됨 |
| `pandas` | DOE table | vdmc 환경에 설치됨 |
| `numpy` | 수치 계산 | vdmc 환경에 설치됨 |
| `matplotlib` | 결과 차트 | vdmc 환경에 설치됨 |
| `PySide6` | UI 위젯 | vdmc 환경에 설치됨 |
| `mujoco` | 접촉 캡처 (선택) | 없으면 캡처 스킵 |
| `gemini` CLI | LLM 채팅 | PATH 또는 config 설정 |

---

## 7. 파일 목록

### 신규 생성
| 파일 | 설명 |
|------|------|
| `whtb_doe/__init__.py` | 패키지 진입점 |
| `whtb_doe/dsl_parser.py` | DSL 파서 (VarSpec, parse_dsl, validate_dsl) |
| `whtb_doe/sampler.py` | DOESampler (LHS / FullFact / MonteCarlo) |
| `whtb_doe/definition.py` | DOEDefinition API 클래스 |
| `whtb_doe/doe_widget.py` | PySide6 DOESetupDialog |
| `whtb_doe/tests/test_dsl_parser.py` | DSL 파서 단위 테스트 |
| `whtb_doe/tests/test_sampler.py` | Sampler + DOEDefinition 단위 테스트 |
| `TVPackageMotionSim/whts_contact_extractor.py` | 낙하 접촉 추출 알고리즘 |
| `TVPackageMotionSim/run_doe_runner.py` | DOERunner 병렬 실행 엔진 |
| `TVPackageMotionSim/run_doe_worker_impl.py` | subprocess 배치 워커 |
| `TVPackageMotionSim/run_doe_analysis.py` | 결과 로드 및 그래프 |
| `TVPackageMotionSim/run_doe_analysis_widget.py` | Gemini CLI 채팅 UI |

### 수정된 기존 파일
| 파일 | 변경 내용 |
|------|----------|
| `TVPackageMotionSim/run_drop_simulator/whts_data.py` | `DropSimResult`에 `corner_impact_hist: Optional[list] = None` 추가 |
| `TVPackageMotionSim/run_drop_simulator/whts_engine.py` | `DropSimResult(...)` 생성자에 `corner_impact_hist=self.corner_impact_hist` 전달 |
| `TVPackageMotionSim/run_drop_simulation_cases_v6.py` | `--doe-worker` 조기 진입점 삽입 (JAX/PySide6 import 전) |
