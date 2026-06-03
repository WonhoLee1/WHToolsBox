# WHTOOLS 최적화 및 DOE 프레임워크 UI/엔진 설계 및 구현 계획 (2026-06-03)

본 계획서는 GooeyParser를 적극 활용한 범용 DOE 입력기 UI와 해석 결과를 비교 분석하여 최적 설계안(Best Parameter Set)을 도출하는 최적화 대시보드(DOE Monitor UI) 및 엔진의 구체적인 구현 방안을 다룹니다.

## User Review Required

> [!IMPORTANT]
> **1. Gooey 기반의 동적 Argument 생성 및 탭 구성**
> - Gooey는 CLI argparse의 스펙을 기반으로 GUI를 자동 빌드하므로, `navigation="Tabbed"` 옵션을 설정하여 **입력 위젯을 탭 단위로 분리**합니다.
> - 입력 변수들은 `GooeyParser`의 `add_argument_group`을 사용하여 파트/성격별로 그룹화 및 탭 분리합니다:
>   - **[Tab 1] Base Config**: JSON 설정 파일 경로 로드.
>   - **[Tab 2] Geometry Bounds**: `box_w`, `box_h`, `box_d` 등의 변수 최소/최대/초기값 범위 지정.
>   - **[Tab 3] Physics Bounds**: `cush_friction`, `drop_height` 등의 물리 범위 지정.
>   - **[Tab 4] DOE Strategy**: 샘플링 방법(LHS, Random, Full Fact), 샘플 수 지정.
>
> **2. 실시간 진행 상황 터미널 피드백**
> - Gooey의 내장 리다이렉션 메커니즘을 활성화하여 시뮬레이션의 `sys.stdout` 로그 및 진행률을 Gooey 내장 터미널 뷰에 실시간으로 표시합니다.
>
> **3. PySide6 기반 DOE Monitor UI (최적화 대시보드)**
> - 해석이 완료된 후, 또는 별도로 실행 가능한 대시보드를 구축하여 다음을 지원합니다:
>   - **결과 보기**: DOE 번호별 개별 해석 이력 분석.
>   - **결과 비교하기**: 다수 DOE 케이스를 다중 선택하여 하나의 차트에 궤적을 겹쳐서 플로팅(Overlay).
>   - **최적안 선택**: 사용자가 제안한 구문(예: `Max VM Stress < 200 and Ground Force -> Min`)에 만족하는 베스트 피쳐 파라미터 추출.

---

## Open Questions

> [!NOTE]
> - **Gooey의 정적 argparse 특성 극복 방안**:
>   Gooey는 스크립트 로드 시점에 Argument 목록을 확정해야 합니다. 따라서 사용자가 임의의 JSON을 올릴 때마다 GUI 인자가 동적으로 바뀌는 구조를 위해, 1단계에서 JSON을 선택하여 로컬 캐시에 등록하면 2단계에서 캐싱된 JSON의 properties 필드를 파싱하여 최적화된 Gooey UI를 보여주는 **2단계 실행 방식(또는 미리 정의된 대표 파라미터 셋 구성 방식)**을 제안합니다. 이 방식이 합당할지 검토 부탁드립니다.

---

## Proposed Changes

### [whtb_physics] (물리 안정성 패치)
`target_inertia`가 3성분으로 입력될 경우 발생하는 브로드캐스팅 에러(`ValueError: operands could not be broadcast together with shapes (3,) (6,)`)를 방지하기 위해 6성분 자동 보정 코드를 추가합니다.

#### [MODIFY] [whtb_physics.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_physics.py)
* `analyze_and_balance_components` 함수 내 `t_moi` 추출 부근에 3성분 검출 시 `[Ixx, Iyy, Izz, 0.0, 0.0, 0.0]`으로 패딩하는 로직 추가.

---

### [whts_optimization_engine] (DOE 테이블 생성 및 해석 관리 엔진)
Gooey로 입력받은 범위 정보를 바탕으로 샘플링을 진행하고 시뮬레이션을 배치 실행하는 역할을 담당합니다.

#### [NEW] [whts_optimization_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_optimization_engine.py)
* **DOE Generator**:
  * Latin Hypercube Sampling(LHS): Scipy가 없을 때를 대비한 numpy 구현 포함.
  * Random, Full Factorial 구현.
  * Discrete(이산형)는 지정된 스텝 간격으로 버림/올림 처리하고, Continuous(연속형)는 실수형 분포로 생성.
* **Batch Runner**:
  * Base JSON 설정을 파싱하여 각 DOE 조건별 설정 파일을 동적으로 생성.
  * `DropSimulator`를 백그라운드에서 기동하며 순차적으로 해석 수행.
  * 진행 현황을 `[Progress] 10%`, `[Progress] 20%` 등 Gooey가 파싱 가능한 포맷으로 터미널에 출력하여 Gooey Progress Bar 연동.
  * 결과를 `results/DOE/case_{id}/` 아래에 `config.json` 및 결과 바이너리 형태로 격리 보존.

---

### [whts_optimization_ui] (Gooey 입력기 및 DOE Monitor UI)
사용자 친화적인 입력기 UI와 결과 비교/최적안 추출용 모니터 UI를 제공합니다.

#### [NEW] [whts_optimization_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_optimization_ui.py)
* **Gooey GUI Setup (`@Gooey`)**:
  * 탭(`navigation="Tabbed"`) 기반 설계.
  * `GooeyParser`를 활용하여 Base JSON 파일 경로 인풋 및 튜닝 변수별 최소/최대/초기값/간격을 입력하는 위젯 그룹 구성.
  * 실행 시 실시간 로그를 터미널 창에 출력.
* **PySide6 Monitor Dashboard (DOE Monitor UI)**:
  * **DOE Case Selector**: 해석이 완료된 Case 목록을 표시하고, 여러 Case를 체크박스로 다중 선택.
  * **Overlay Graph Area**: matplotlib 피팅 모듈을 이용하여 Z-displacement, Ground Force, Max Von-Mises Stress를 중첩하여 다채로운 컬러로 가시화 (`koreanize-matplotlib` 9pt 필수 적용).
  * **Optimization Rules & Solver**: 사용자가 코딩 규칙이나 조건(예: `Max_VM_Stress < 150` 및 `Minimize Ground_Force`)을 기입하면 이를 충족하는 최고의 Case와 파라미터 조합을 한눈에 표로 추출하여 출력.

---

### [run_optimization_framework] (실행 진입점)
전체 프레임워크를 조율하는 메인 파일입니다.

#### [NEW] [run_optimization_framework.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_optimization_framework.py)
* Gooey가 래핑된 스크립트 실행 및 해석 완료 후 `whts_optimization_ui` 대시보드를 실행하도록 유도하는 런처.

---

## Verification Plan

### Automated Tests
1. **LHS 및 샘플링 분포 체크**:
   - LHS 샘플이 겹치지 않고 다차원 공간에 균일하게 퍼져나가는지 테스트 코드로 시각화 및 수치적 검증.
2. **Gooey Parser 파싱 및 로그 스트리밍 동작**:
   - `sys.stdout` 출력이 Gooey 터미널창에 정상 동기화되어 진행 바가 연동되는지 체크.

### Manual Verification
- `run_optimization_framework.py` 실행 시 Gooey 탭이 정상적으로 나타나는지 확인.
- 임의 변수 범위와 샘플링 방법을 선택하고 "Start"를 눌러 해석이 돌아가는 로그를 관찰.
- 완료 후 DOE Monitor UI가 자동으로 팝업되어 다수 Case를 선택하고 Overlay 플롯이 올바르게 중첩 표시되는지 확인.
- 필터 조건 입력 시 최적 설계안(Best Case) 테이블이 바르게 업데이트되는지 검증.
