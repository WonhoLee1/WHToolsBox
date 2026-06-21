# Deep Interview Spec: WHToolsBox DOE System

## Metadata
- Interview ID: whtb-doe-system-20260620
- Rounds: 7
- Final Ambiguity Score: 17.4%
- Type: brownfield
- Generated: 2026-06-20
- Threshold: 0.20
- Threshold Source: default
- Initial Context Summarized: no
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.87 | 0.35 | 0.3045 |
| Constraint Clarity | 0.82 | 0.25 | 0.2050 |
| Success Criteria | 0.80 | 0.25 | 0.2000 |
| Context Clarity | 0.78 | 0.15 | 0.1170 |
| **Total Clarity** | | | **0.826** |
| **Ambiguity** | | | **17.4%** |

## Topology
| Component | Status | Description | Coverage |
|-----------|--------|-------------|---------|
| DOE Core Module | active | `WHToolsBox/whtb_doe/` 신규 패키지. 문자열 DSL 파싱, sampling, DOEDefinition 클래스 | 구현 1순위 |
| DOE Execution Engine | active | TVPackageMotionSim 내 subprocess 기반 병렬 DOE 실행 + 로그 모니터링 | 구현 2순위 |
| Drop Contact Extraction | active | 신규 독립 파일. 코너 접촉 시점/힘/임팩트 계산, ContactSeqList | 구현 3순위 |
| DOE UI Widget | active | PySide6 다이얼로그. config tree 탐색, DOE table 편집 | 구현 4순위 |
| DOE Results Analysis | active | DOE 결과 폴더 pkl 로드, 그룹핑, Gemini CLI 채팅 창 | 구현 5순위 |

## Goal
WHToolsBox에 범용 DOE(실험계획법) 시스템을 구축한다. 핵심은 `whtb_doe` 패키지(문자열 DSL 기반 파라미터 정의 → sampling → table 생성)이며, TVPackageMotionSim에서 이를 활용하여 n개 subprocess 병렬 낙하 시뮬레이션을 실행하고 결과를 분석할 수 있는 파이프라인을 제공한다. 별도로 낙하 접촉 시점 추출 알고리즘과 Gemini CLI 기반 대화형 결과 분석 채팅 창을 포함한다.

## Constraints
- **DOE Core 위치**: `WHToolsBox/whtb_doe/` — 독립 패키지, TVPackageMotionSim 외 프로젝트에서도 import 가능
- **샘플링 라이브러리**: `scipy.stats.qmc` (vdmc 환경에 이미 설치됨, 추가 의존성 없음)
- **기존 DOEEngine**: `run_parameter_study/whts_optimization_engine.py`의 기존 `DOEEngine`은 무시(동결). `whtb_doe`는 clean-slate 구현
- **병렬 실행**: subprocess 방식 — 각 DOE case를 독립 OS 프로세스로 실행 (dev_plan.md: "독립적으로 배치로 실행")
- **UI 프레임워크**: PySide6 (기존 앱과 동일)
- **LLM 통합**: Gemini CLI 사용. local MCP / API key 방식 불가. Python 배포에 포함된 채팅 창 UI로 제공
- **모니터링 로그**: `run_doe_log.txt` (매번 덮어쓰기, 실행 중 케이스 표시), `done_doe_log.txt` (append, 완료 케이스 기록)
- **결과 폴더**: 기본 `results/DOE_D{날짜}_{시간}/`, DOEDefinition 생성자 또는 멤버 함수로 변경 가능
- **Contact Extraction**: 현재 구현된 py 파일과 별개의 신규 파일, 다른 PC 브랜치 이식 용이하게 독립적으로 작성
- **Python 환경**: vdmc conda 환경

## Non-Goals
- 기존 `run_parameter_study/whts_optimization_engine.py` 리팩토링 (동결)
- local MCP 또는 Claude API key 기반 LLM 통합
- DOE UI Widget이 DOE Core 완성 전에 구현됨
- Results Analysis가 Contact Extraction 전에 구현됨

## Acceptance Criteria

### Component 1: whtb_doe Core Module
- [ ] `WHToolsBox/whtb_doe/__init__.py` 존재, `from whtb_doe import DOEDefinition` 작동
- [ ] DSL 문자열 파싱 지원:
  - `cfg['a']['b'] = [10,20,30]` — 이산형 리스트
  - `cfg['a']['f'] = [10:199:15]` — `start:end:count` (15개 등간격)
  - `cfg['d'] = 10:200:100` — `min:max:init` 연속형
  - `norm:mean:std:number` — 정규분포 (number 생략 가능)
  - 문자열 값 (`cfg['x'] = ['A','B','C']`) 지원
- [ ] `DOEDefinition(dsl_string, base_config)` 생성자
- [ ] `validate()` → `(bool, message_str)` 반환, 오류 위치 명시
- [ ] `generate(method, n_samples, seed)` → `(doe_table: List[Dict], config_list: List[Dict])` 반환
  - method: `'lhs'`, `'fullfact'`, `'montecarlo'`
  - doe_table 열: `case_number, varname1, varname2, ...`
- [ ] `regenerate(modified_doe_table)` → 수정된 table 기반 config_list 재생성
- [ ] sampling 백엔드: `scipy.stats.qmc.LatinHypercube` (LHS), `itertools.product` (FullFact), `scipy.stats.qmc.MultivariateNormalQMC` (정규분포)
- [ ] `cfg['a']['b']` 형태의 중첩 dict 경로 접근 파싱 및 적용

### Component 2: DOE Execution Engine (TVPackageMotionSim)
- [ ] `TVPackageMotionSim/run_doe_runner.py` (또는 `run_drop_simulator/whts_doe_runner.py`) 신규 파일
- [ ] `DOERunner(doe_definition, n_parallel, output_root)` 클래스
- [ ] `run()` — n개 subprocess 병렬 실행, queue로 슬롯 관리
- [ ] `run_doe_log.txt`: 4열(process_id, target_time, current_time, frame) × n행 모니터 박스, 매 주기 덮어쓰기, 완료 케이스는 표시 안 됨
- [ ] `done_doe_log.txt`: 완료 케이스 append 기록
- [ ] 결과 폴더: `results/DOE_D{YYYYMMDD}_{HHMMSS}/case_{n:03d}/`
- [ ] DOE 진행 시 카메라 각도 설정 + mp4 캡처 기능 (`sim.config['doe_camera_angle']`)
- [ ] 단일 DOE case 실행 진입점: `python -m run_doe_case --case 3 --config doe_config.json`

### Component 3: Drop Contact Extraction
- [ ] `TVPackageMotionSim/whts_contact_extractor.py` 신규 독립 파일
- [ ] `extract_contact_sequence(result_instance)` 함수
- [ ] 각 코너의 최초 지면 접촉 시점 검출 (튀는 현상 처리: 최초 접촉 기준)
- [ ] `ContactSeqStr`: 접촉 순서 문자열, 예) `"C4-C3-C2-C1"`
- [ ] `ContactSeqTimeList`: `[['C4', 10.1], ...]` — 접촉 직전 프레임 시간
- [ ] `ContactSeqForceList`: `[['C4', force_max], ...]` — 접촉 전후 구간 최대 접촉력
- [ ] `ContactSeqImpactList`: `[['C4', impulse], ...]` — 접촉력×시간 적분값
- [ ] `ContactSeqList`: 위 3개를 통합한 dict 구조 반환
- [ ] result instance에 `contact_seq` 속성 추가 저장 후 pkl 재저장
- [ ] 접촉 직전 프레임 MuJoCo 렌더 캡처 → `{result_dir}/contacts/` 폴더에 저장
- [ ] 독립 import 가능 (TVPackageMotionSim 내 다른 파일 의존 최소화)

### Component 4: DOE UI Widget
- [ ] `WHToolsBox/whtb_doe/doe_widget.py` — PySide6 QDialog 기반
- [ ] "Add Variable" 버튼 → config dict를 tree형 테이블로 표시, 변수 선택
- [ ] 선택 변수별 DSL 입력 필드 (이산/연속/정규분포 형식)
- [ ] 입력 형식 유효성 검사 실시간 표시
- [ ] "Make DOE List" 버튼 → DOE table을 QTableWidget으로 표시
- [ ] DOE table 직접 편집 → "Apply changes?" 확인 다이얼로그
- [ ] "OK" 버튼 → 호출자에게 `(doe_table, config_list)` 반환
- [ ] TVPackageMotionSim Control UI에서 호출 가능

### Component 5: DOE Results Analysis
- [ ] `TVPackageMotionSim/run_doe_analysis.py` 신규 파일
- [ ] DOE 결과 폴더 스캔 → case별 result pkl 로드
- [ ] `ContactSeqStr` 기준 case 그룹핑
- [ ] 그룹별 / 코너별 최대 접촉력 중심-산포 막대 그래프 (matplotlib)
- [ ] Gemini CLI 채팅 창 UI (PySide6 QDialog):
  - pkl 구조 요약을 context로 자동 첨부
  - 사용자 질문 입력 → `gemini` CLI 호출 (subprocess)
  - 응답 표시
  - Python 코드 블록 자동 감지 → "Run this code" 버튼 제공
- [ ] Python 배포(PyInstaller spec)에 포함 가능한 구조

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| 병렬 실행이 필요 | MuJoCo가 이미 4스레드 — 병렬 중복 아닌가? | 필요 확인: DOE case 수십~수백 개, case별 병렬이 별도 |
| DOEEngine 재활용 | 기존 run_parameter_study에 이미 DOEEngine 존재 | whtb_doe에서 clean-slate 구현, 기존 코드 동결 |
| LLM = Claude API | local API key 사용 가능 | Gemini CLI 방식으로 확정 (API key 불가 환경) |
| DSL 형식 `[10:100:4]` = 4개 값 | start:end:count인지 start:end:step인지? | `start:end:count` (개수 기준) 로 확정 |

## Technical Context
- **기존 DOEEngine**: `TVPackageMotionSim/run_parameter_study/whts_optimization_engine.py:41` — LHS/Random/FullFact 구현. whtb_doe와 무관하게 동결
- **기존 doe 파이프라인**: `TVPackageMotionSim/run_drop_simulation_cases_doe.py` — `doe_process_pipeline()` 함수, modeling_func 패턴 존재. DOERunner가 이 패턴을 확장
- **DropSimulator**: `TVPackageMotionSim/run_drop_simulator/__init__.py` — DOE case 실행 단위
- **result pkl**: `sim.result` 인스턴스, `corner_pos_hist`, `corner_vel_hist`, `ground_impact_hist` 등 포함
- **MuJoCo 렌더**: mp4 캡처 기능은 기존 `whts_engine.py`에 구현되어 있음 (참조 필요)
- **PySide6**: 기존 Control Panel(`whts_control_panel.py`)에서 동일 패턴 사용
- **배포**: `TVPackageMotionSim/drop_simulator_v6.spec` PyInstaller spec 기존 존재

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| DOEDefinition | core domain | dsl_string, base_config, output_dir, method, seed | generates DOETable, generates config_list |
| DOETable | core domain | case_number, variable columns, rows | generated by DOEDefinition |
| DOERunner | supporting | doe_definition, n_parallel, output_root | uses DOEDefinition, spawns subprocesses |
| DropSimulator | external system | config, result, output_dir | used by DOERunner per case |
| ContactSeqList | core domain | ContactSeqStr, TimeList, ForceList, ImpactList | computed from result instance |
| DOEWidget | supporting | base_config, selected_vars, doe_table | wraps DOEDefinition, returns to caller |
| GeminiChatWidget | supporting | pkl_context, chat_history | calls gemini CLI subprocess |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 4 | 4 | - | - | N/A |
| 7 | 7 | 3 | 0 | 4 | 57% → converging |

## Interview Transcript
<details>
<summary>Full Q&A (7 rounds)</summary>

### Round 0 (Topology)
**Q:** 5개 컴포넌트 topology 확인
**A:** 맞아요 (5개 모두 진행)

### Round 1
**Q:** DOE Core Module 위치?
**A:** WHToolsBox/whtb_doe/ 신규 패키지

### Round 2
**Q:** 산출물 형태?
**A:** 구현 코드 작성까지 (자동화 실행)

### Round 3
**Q:** DOE 샘플링 라이브러리?
**A:** 상관없음, 추천대로 → scipy.stats.qmc 선택

### Round 4 [Contrarian]
**Q:** 기존 DOEEngine 재활용 여부?
**A:** whtb_doe에서 완전히 새로 구현, 기존 무시

### Round 5
**Q:** 병렬 실행 방식? (1/2번 차이 설명 후)
**A:** dev_plan.md대로 병렬 필요 → subprocess 방식 확정

### Round 6
**Q:** 구현 순서?
**A:** DOE Core → Execution Engine → Contact → UI → Analysis

### Round 7
**Q:** LLM 활용 방식?
**A:** Gemini CLI, Python 배포 포함 채팅 창 UI

</details>
