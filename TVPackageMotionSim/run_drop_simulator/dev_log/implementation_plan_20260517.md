# Model Configuration & Setup UI 개선 구현 계획서 (Model Config & Setup UI Improvement) - v2

본 계획서는 `box_motion.py`를 직접 임포트하거나 호출하지 않고 오직 **참조(Reference)**용으로만 삼는다는 사용자의 새로운 요구사항에 맞추어 업데이트되었습니다. 모든 핵심 수학적 연산 및 ISTA 자세/시퀀스 매핑 코드는 [whts_ista_helper.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_ista_helper.py)에 완전하고 독립적으로 이식 및 구현됩니다.

---

## 🛠️ 주요 요구사항 및 개선 계획

1. **삼성 TV 참조 모델 외부 CSV화 및 "Ref. Model" 버튼 연동**
   - 기존 `box_motion.py` 내부에 하드코딩되었던 `REFERENCE_MODELS` 데이터를 외부 UTF-8 CSV 파일 `tv_ref_model_info.csv`로 분리 관리(완료).
   - `box_motion.py`는 그대로 두고, `whts_control_panel.py`가 구동 시 이 CSV 파일을 동적으로 로드하게 수정하며, 예외 상황에 대비한 견고한 폴백(Fallback) 하드코딩 데이터를 지원하여 무오류 구동을 보장합니다.

2. **독립 모듈형 [whts_ista_helper.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_ista_helper.py) 구현**
   - `box_motion.py`를 직접 임포트하지 않기 위해, 기하학적 면(Face) 정의 및 ISTA 6A 테스트 시퀀스 생성, 낙하 자세 매핑 로직을 `whts_ista_helper.py`라는 새로운 PySide6 친화적 모듈로 분리 이식합니다.
   - 이 모듈에는 `IstaFaceMapper` 및 `ISTA6ASimulator` 클래스와 그에 수반되는 수치 연산 코드가 포함되며, 깔끔한 OOP 및 Docstring 구조를 준수하여 작성됩니다.

3. **GENERAL 모드에서의 동적 Dropdown 자세 설정 UI 제공**
   - 사용자가 **ISTA Mode**로 `GENERAL`을 선택할 경우:
     - 낙하 형태(`Type`) 선택 드롭다운(Face / Edge / Corner)을 제공합니다.
     - **Corner** 선택 시: 3개의 드롭다운 박스를 활성화하여 `Front`/`Back`, `Top`/`Bottom`, `Left`/`Right`의 조합을 유도합니다.
     - **Face** 선택 시: 1개의 드롭다운 박스를 활성화하여 6개 면 중 하나를 선택하도록 유도합니다.
     - **Edge** 선택 시: 2개의 드롭다운 박스를 활성화하여 수직인 두 면의 조합을 유도합니다.
   - 드롭다운 값을 변경할 때마다 `drop_direction`이 `Corner front-bottom-left` 또는 `Edge front-bottom` 형태로 자동 업데이트되고, 3D Schematic 뷰와 강체 파라미터가 실시간 연계 업데이트됩니다.

4. **PARCEL 또는 LTL 모드에서의 ISTA 6A 테스트 시퀀스 헬퍼(Help Tool) 구현**
   - **ISTA Mode**로 `PARCEL` 또는 `LTL`을 선택할 경우:
     - `Drop Direction` 대신 🔍 **Select Sequence** 버튼과 읽기 전용 텍스트창을 제공합니다.
     - 이 버튼을 누르면 고도로 세련된 모달리스/모달 다이얼로그 `IstaSetupHelperDialog`가 뜹니다.
     - **ISTA Setup Helper Dialog 구성:**
       - **Reference Model 선택:** 💾 `Ref. Model` 버튼을 배치하여, 외부 CSV 목록을 PySide6 `QTableWidget`에 정렬(Sorting) 가능하게 출력하고 선택 시 자동으로 크기(box_w, box_h, box_d)와 무게(pkg_m)를 헬퍼 입력칸에 반영합니다.
       - **수치 직접 수정:** 사용자가 치수 및 중량을 헬퍼 창에서 직접 기입할 수도 있습니다.
       - **Face Numbering 설명 텍스트 라벨:** Parcel과 LTL 모드에 따라 상이한 Face Numbering 정보(1~6)를 시각적으로 보여주는 설명 라벨 영역을 가독성 있게 렌더링합니다.
       - **Test Sequence Table:** `whts_ista_helper.py` 모듈을 임포트 및 활용하여 현재 설정된 치수/무게에 맞는 12~17개의 낙하 규격을 테이블 형태로 실시간 출력(규격 코드 A~H 자동 진단)합니다.
       - **적용 및 동기화:** 테이블에서 시퀀스를 선택하고 `Apply` 버튼을 누르면, 메인 `ModelSetupDialog` 창의 무게, 사이즈, 낙하 높이, 낙하 자세(Azimuth, Lat/Tilt 및 drop_direction)가 즉시 연동 업데이트되어 XML 생성을 바로 수행할 수 있게 합니다.

---

## 🏗️ 상세 변경 대상 파일 및 역할

### 1. [NEW] [whts_ista_helper.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_ista_helper.py)
- `IstaFaceMapper` 및 `ISTA6ASimulator` 클래스를 PySide6 GUI 요건 및 물리 연산 요건에 최적화하여 담는 독립 헬퍼 모듈.
- 치수 및 중량을 바탕으로 한 낙하 각도/자세(Azimuth, Lat/Tilt 등) 및 시퀀스 명세 계산의 백엔드 연산을 지원합니다.

### 2. [MODIFY] [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)
- `whts_ista_helper.py` 모듈을 임포트합니다.
- `tv_ref_model_info.csv`를 파싱하여 PySide6 용 데이터로 로드하는 헬퍼 함수를 추가합니다.
- `ModelSetupDialog` 클래스 개선:
  - `_init_ui` 메서드에 GENERAL 모드용 동적 드롭다운 전용 컨테이너 위젯과 PARCEL/LTL 모드용 **Select Sequence** 버튼 UI를 스택 형태로 삽입합니다.
  - 드롭다운 선택 변경 시 `drop_direction`을 `Corner front-bottom-left` 등과 같이 파싱 가능한 최적화 규격명으로 계산하여 동기화합니다.
- **[NEW]** `SelectTVModelDialog` 클래스 구현:
  - TV 레퍼런스 모델을 한눈에 보고 가로/세로/무게 등으로 정렬하여 선택할 수 있는 PySide6 테이블 기반 팝업.
- **[NEW]** `IstaSetupHelperDialog` 클래스 구현:
  - ISTA 6A 규격 낙하 시퀀스를 진단하고, 선택하여 메인 무게/치수/높이/자세를 일괄 적용 및 업데이트할 수 있는 통합 GUI.

---

## 🧪 검증 계획 (Verification Plan)

### 수동 검증 및 시각적 모니터링
1. `whts_control_panel.py`를 단독 기동하거나 MuJoCo 시뮬레이터 연동을 통해 개선된 `Model Configuration & Setup` 창을 호출합니다.
2. `ISTA Mode`를 `GENERAL`로 전환하고, `Face`, `Edge`, `Corner` 라디오/콤보 선택에 따라 동적 드롭다운 갯수(1~3개)가 올바르게 스위칭되는지 확인합니다.
3. 드롭다운 값을 바꾸었을 때 `drop_direction` 텍스트 필드와 3D Schematic 뷰의 바닥 타겟팅이 실시간으로 바뀌는지 확인합니다.
4. `ISTA Mode`를 `PARCEL` 또는 `LTL`로 변경하고 🔍 **Select Sequence** 버튼을 클릭하여 `IstaSetupHelperDialog`가 미려하게 팝업되는지 확인합니다.
5. 헬퍼 다이얼로그에서 `Ref. Model` 버튼을 클릭해 삼성 TV 모델 선택창을 띄우고, 특정 TV 모델을 더블클릭/선택하였을 때 치수와 무게가 헬퍼에 로드되는지 확인합니다.
6. Parcel/LTL 라디오 변경 시 하단 설명 박스의 Face Numbering 안내문이 갱신되는지 확인합니다.
7. 생성된 시퀀스 리스트에서 10번째 시퀀스(예: Corner 2-3-5, Edge 3-4 등)를 마우스로 더블클릭 또는 선택 후 Apply 하였을 때, 메인 설정창의 무게, 사이즈, 낙하 높이, 회전 자세 정보가 완벽하게 일괄 주입 및 갱신되는지 최종 확인합니다.

---

## ⚖️ 5. Mass, CoG, MoI 자동 컴포넌트 밸런싱 최적화 도구 구현 (Mass, CoG, MoI Component Auto-Balancing Optimizer)

* **상세 연동 구조:**
  - 메인 `ModelSetupDialog` 창의 하단 그룹명 `"Mass & Dynamic Reporting"`을 `"Mass, CoG, MoI"`로 리네임하고, 우측에 파란색 테마의 `⚖️ Balance` 버튼을 추가합니다.
  - `⚖️ Balance` 버튼 클릭 시 `ComponentBalanceDialog`를 모달로 호출합니다.
  - 이 대화 상자는 Target Mass, Target CoG, Target MoI, Balancing Mass Count를 입력받습니다.
  - 무게중심(CoG)과 관성모멘트(MoI) 매칭에 대한 강도를 미세 튜닝하는 Optimization Focus Priority 슬라이더를 배치합니다.
  - SLSQP 최적화 엔진을 구동하여 9차원 자유도(보조 질량 중심 `pos_aux`, 보조 질량 분산 Span `dx, dy, dz`, 분포 계수 `a, b, g`) 하에서 타겟을 추종하는 최적 위치 및 무게 분포를 역산합니다.
  - 질량추들의 설치 한계 경계 구속조건(Bounding Box Clipping)을 반영하여 물리적 충돌 및 외곽 이탈을 원천 차단합니다.
  - 격자 형태의 5열 결과 테이블(`Metric | Initial (Base) | Target (Req) | Final (Balanced) | Status`)을 렌더링하고, 오차 수렴 여부에 따라 초록색 `✅ OK` 또는 노란색 `⚠️ LIMIT` 상태 표시를 제공합니다.
  - 질량추 개수 최적화 마법사 버튼(`Suggest Optimal Count`)을 제공하여, 백그라운드 루프 최적화를 거쳐 최적의 개수를 추천(Increase/Decrease) 및 적용합니다.
  - **레이아웃 위치 최적화:** 사용자 편의성과 시각적 직관성을 향상하기 위해, "Mass, CoG, MoI" 그룹 패널을 상세 설정 트리 및 에디터(`mid_splitter`) 보다 상단이자, 상단 드롭다운 및 스키매틱 뷰(`top_splitter`) 바로 아래(중앙 섹션)로 이동하여 배치합니다.
  - **QSpinBox / QDoubleSpinBox 화살표 스타일 고도화 (Side-by-side SVG Chevrons):** 기존의 기형적인 native 닷(.) 형태의 업다운 화살표 뭉개짐 오류를 제거하기 위해, 서브컨트롤을 `transparent` 배경 하에 `subcontrol-position: center right`를 이용한 마진 분기로 가로(side-by-side) 정렬하고, 벡터형 인라인 SVG 코드로 `^` 및 `v` chevron 화살표 그래픽을 강제 적용합니다.

---

## ⚖️ 6. Mass, CoG & MoI Balancing Optimizer Layout 개편 및 오동작 조치 (Layout Restructuring & Debugging)

### [요구사항 1] 왼쪽 입력 영역 폭 50% 제한 및 결과 테이블 하단 배치
- **수평에서 수직 레이아웃 개편:** `ComponentBalanceDialog` 내에서 기존에 `QHBoxLayout`으로 좌/우 배치되어 있던 입력 패널(`specs_group` + `focus_group`)과 결과 테이블(`results_group`)을 수직 적층 `QVBoxLayout` 구조로 전면 전환합니다.
- **가로 폭 50% 조밀화:** 상단 영역에 `specs_group`과 `focus_group`을 `QHBoxLayout` 하위로 병렬(Side-by-side) 배치하고 각각 1:1 stretch factor를 지정합니다. 이로써 `specs_group`(목표 사양 입력창)의 가로 폭이 전체 다이얼로그 너비의 딱 50% 수준으로 정밀 조율되며, 수직 밀도가 대폭 개선됩니다.
- **하단 테이블 와이드 배치:** 결과 테이블(`results_group`)을 하단에 단독 배치하여, 모든 열("Metric", "Initial", "Target", "Final", "Status")이 잘림 현상 없이 수평으로 충분한 여백을 확보한 고해상도 테이블 디자인을 제공합니다.
- **다이얼로그 크기 조율:** 최적의 수직 및 수평 스케일 유지를 위해 기본 다이얼로그 크기를 `(850, 680)`으로 지능적으로 크기 조절합니다.

### [요구사항 2] Run Balancing Optimization & Apply to Configuration 버튼 오동작 완벽 조치
- **Scipy Bounds ValueError 차단:** SLSQP 최적화 구동 시, 섀시 치수가 가변적이거나 수치가 극단적인 환경에서 초기 탐색 변수 `dx_init`, `dy_init`, `dz_init`가 Scipy Bounds 영역을 초과하여 `ValueError`를 유발하는 현상을 차단합니다. `dx_init`, `dy_init`, `dz_init`를 `np.clip`을 통해 섀시 한계 수치(`limit_x * 0.95`, `limit_y * 0.95`, `limit_z * 0.95`)와 하한선 `0.0011` 사이의 안전 반경 내에 항상 가두어 강제 초기화함으로써 솔버의 수렴 신뢰도를 극대화합니다.
- **연산 오류의 시각적 로깅 (Try-Except Guard):** `on_optimize_clicked` 및 `on_apply_clicked` 내부에서 일어나는 모든 선형 시스템/최적화 계산 흐름을 단단한 `try-except` 블록으로 방어하고, 에러 검출 시 PySide6 `QMessageBox.critical` 팝업에 명확한 traceback 정보와 에러 내용을 노출하여 오동작 원인을 실시간 진단하도록 만듭니다.
- **Apply 시 강제 최적화 루프 선행 동기화:** 사용자가 입력창에서 Target Mass 등을 수정한 직후 `Run Balancing Optimization` 버튼을 누르지 않고 `Apply to Configuration` 버튼을 다이렉트로 눌렀을 때, `on_apply_clicked` 내부에서 최신 입력값을 감지해 최적화 연산(`run_optimization_engine()`)을 백그라운드에서 강제로 선행 수행한 뒤 최신 데이터를 XML 및 시뮬레이터 구성 트리에 바인딩하도록 보완합니다.

### Automated Verification Plan (검증 계획)
1. `whts_control_panel.py` 단독 기동 혹은 시뮬레이션 인터페이스에서 `Model Configuration & Setup` 창의 최소 가로 폭이 750px 및 기본 초기 가로 폭이 800px로 세팅되어 나타나는지 점검합니다.
2. `⚖️ Balance` 버튼을 클릭하여 다이얼로그가 좌측/우측 분할식 정밀 레이아웃(좌측: Specs 및 결과 테이블 상하 적층, 우측: Focus 슬라이더 및 Stretch)으로 완벽하게 나타나는지 검증합니다.
3. Target Mass, CoG, MoI 입력칸과 결과 테이블의 가로 폭이 좌측의 약 65~70% 너비를 조화롭게 점유하고, 우측에는 Focus 슬라이더가 30~35% 폭을 깔끔하게 차지하는지 시각적 유효성을 점검합니다.
4. Target Mass와 Count를 임의로 조정한 뒤, `Run Balancing Optimization` 버튼을 클릭하여 SLSQP 솔버가 오차 수렴 결과를 테이블에 `✅ OK` 상태로 실시간 업데이트하는지 확인합니다.
5. Target 값을 수정한 후 최적화 버튼을 누르지 않고 곧바로 `Apply to Configuration`을 누르고, 백그라운드 최적화가 자동 유도되면서 다이얼로그가 닫히고 부모 트리의 `components_balance` 데이터가 정상 리로드되는지 최종 점검합니다.

---

## ⚖️ 7. ComponentBalanceDialog 좌측 하단 배치 레이아웃 정밀 개편 (Results Table bottom-left Relocation)

### [요구사항] Optimization & Balancing Results 테이블을 왼쪽 아래(Left Column Bottom)로 배치
- **수평 분할 레이아웃 적용 및 정렬:**
  - 다이얼로그의 메인 레이아웃을 다시 좌/우 2열 배치(`QHBoxLayout`)로 구성하되, 사용자의 요구사항에 맞추어 요소들을 배치합니다.
  - **좌측 열 (Left Column - 65% ~ 70% 폭 할당):**
    - `specs_group` (Target Specifications - 입력창)을 상단에 배치합니다.
    - `results_group` (Optimization & Balancing Results - 결과 테이블 및 조작 버튼)을 하단에 배치하여, 입력값 바로 하단에서 최적화 결과를 직관적으로 볼 수 있게 설계합니다 (즉, 기존 우측에 있던 결과 테이블을 **왼쪽 아래**로 완전히 이전합니다).
  - **우측 열 (Right Column - 30% ~ 35% 폭 할당):**
    - `focus_group` (Optimization Focus Priority - 가중치 튜닝 슬라이더)을 우측 상단에 단독 배치하고 아래에 stretch를 지정하여 깔끔하고 조밀한 여백을 구성합니다.
  - **다이얼로그 크기 조율:**
    - 이 2열 스택 구조에 가장 안정적인 화면 밀도를 위해 `self.setMinimumSize(950, 620)`로 최종 조절하여, 결과 테이블이 충분한 가로 길이를 보장받으면서도 전체 창의 가로 폭이 너무 과도하지 않도록 조화롭게 구성합니다.

---

## ⚖️ 8. ModelSetupDialog 메인 모델 설정 UI 정밀 수정 및 중복/레거시 제거 (Surgical Clean-up 완료)
- **중복 뷰어 제거:**
  - `ModelSetupDialog` 내부 `_init_ui`에서 중복 배치되었던 `"Mass, CoG, MoI"` 그룹박스(`bottom_group`) 관련 레이아웃 구문을 완전히 제거하여 화면을 한결 심플하고 실용적으로 정비합니다.
- **🔍 Setup 버튼의 단일 통로 연결:**
  - `TV Size Preset` 옆에 배치된 `self.btn_select_ref_model_direct` 버튼 명칭을 `"🔍 Setup"`으로 재지정하고, 클릭 시 즉시 `IstaSetupHelperDialog`가 호출되는 `self._on_select_sequence`로 슬롯을 연결 변경합니다.
- **레거시 무용수 제거:**
  - 더 이상 호출되지 않는 불필요한 메소드들(`_on_select_ref_model_direct`, `_update_reporting`, `_on_balance_clicked`)을 파일 내에서 영구 소거합니다.
  - `__init__`, `_on_apply_value`, `_on_preset_changed`, `_on_general_dropdowns_changed`, `_on_numeric_ui_changed`, `_on_ista_changed` 내부에서 더 이상 호출 대상이 없는 `self._update_reporting()`구문들을 안전하게 제거하여, 혹시 모를 AttributeError를 철저하게 방지합니다.


---

## 📦 9. ISTA 6-Amazon Test Setup Helper 다이얼로그 레이아웃 & 스핀박스 입력부 정밀 개선 (Layout & SpinBox Improvement)

### [요구사항 1] Custom 선택 시 공간 확장 및 크기 요동 방지 (QStackedWidget 도입)
- **원인 분석:** `IstaSetupHelperDialog`는 LTL/Parcel일 때 세 그룹박스(`diag_group`, `face_desc_box`, `seq_group`)를 보이고 Custom 모드일 때는 이들을 숨깁니다. 이로 인해 아래가 텅 비면서 크기가 극도로 줄어들거나 창 내부 공간 밸런스가 파괴됩니다.
- **개선 방안 (QStackedWidget 스위칭 & 고정 사이즈):**
  - 다이얼로그의 전체 크기를 `self.setFixedSize(540, 720)`으로 완전 고정하여 모드 스위칭 시 창 크기가 요동치지 않도록 방지합니다.
  - 다이얼로그 하단에 `QStackedWidget`을 배치하여 두 개의 모드 전용 뷰를 전환합니다.
    - **Page 0 (LTL / Parcel Mode):** 기존의 `diag_group` (진단 결과), `face_desc_box` (면 번호 매핑 가이드), `seq_group` (테이블)을 담은 위젯을 수직 적층하여 정보 가독성을 극대화합니다.
    - **Page 1 (Custom Mode):** 텅 비는 공간을 메우기 위해 세련된 **[Custom Drop Mode Guide]** 정보 상자를 배치하고, 그 하단에 2D 박스-부품 비례 스키매틱 뷰인 `VisualSchematicWidget`을 임베딩하여, 540x720 고정 크기 내에서 시각적 긴장감을 훌륭하게 채우고 아름다운 디자인을 달성합니다.

### [요구사항 2] 창 폭을 2/3 수준(540px)으로 컴팩트하게 슬림화
- **해상도 축소:** 다이얼로그 가로 폭을 기존 800px에서 약 2/3 수준인 **540px**로 정밀 정렬합니다. 
- **레이아웃 다듬기:** 좁아진 가로 폭에 맞추어 `table_seq` 테이블의 가로 스크롤 및 잘림을 최소화하고, 버튼 패널들의 stretch factor를 유연하게 재분배합니다.

### [요구사항 3 & 4] Width, Height, Depth 실수 값 입력 보장 및 가용 입력부 너비 확장
- **원인 분석:** 테마 스타일시트(`GLOBAL_QSS`)에 선언된 `QDoubleSpinBox`의 `padding-right: 36px`과 `padding: 4px` 때문에, 스핀박스 가로 너비 `90`은 글자가 다 잘려서 한 자만 보이는 심각한 기하학적 문제를 야기했습니다.
- **해결 방안 (너비 확장 및 2열 최적화):**
  - 세 스핀박스(`spin_w`, `spin_h`, `spin_d`)의 `setFixedWidth(90)` 설정을 **`setFixedWidth(125)`**로 확장합니다. 우측 패딩 36px과 좌측 패딩 4px을 제하고도 85px이 남으므로 `1.425`와 같은 소수점 3자리 실수값이 눈이 시원하도록 또렷하게 잘림 없이 표기됩니다.
  - 가로 폭 540px 안에서 125px의 스핀박스를 3개 배치하기에 레이아웃 충돌이 나지 않도록, `QGridLayout`인 `input_layout`의 컬럼 가로 간격을 정밀 조율하여 꽉 차고 세련된 그리드를 형성합니다.

### Automated Verification Plan
1. `whts_control_panel.py` 단독 기동 혹은 메인 GUI를 통해 `IstaSetupHelperDialog`를 로드합니다.
2. 다이얼로그의 전체 크기가 가로 540px, 세로 720px로 정밀 세팅되어 나타나는지 확인합니다.
3. Width, Height, Depth 입력부에서 실수값 `0.700`, `1.425`를 입력했을 때 소수점 및 숫자가 가려지거나 잘리지 않고 통째로 온전하게 출력되는지 확인합니다.
4. LTL/Parcel 라디오 토글 시 시퀀스 테이블과 가이드 정보가 정상 노출되는지 확인합니다.
5. Custom 라디오 토글 시 창 크기가 요동치지 않고 그대로 유지된 채, 하단 StackedWidget이 전환되면서 2D 스키매틱 뷰 및 Custom 안내 박스가 공간을 알차게 점유하는지 시각적 premium 완성도를 최종적으로 검증합니다.




