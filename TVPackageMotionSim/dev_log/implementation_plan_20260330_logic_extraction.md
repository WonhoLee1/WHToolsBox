# [WHTOOLS] Post-Processing UI 로직 분리 (UI/Analysis 분리) 계획

`whts_postprocess_ui.py`의 비대화를 방어하고 코드 가독성을 확보하기 위해, 대규모 데이터 처리 및 수학적 연산 로직을 별도의 분석 모듈(`whts_postprocess_analysis.py`)로 추출합니다. UI 파일은 위젯 배치와 이벤트 바인딩 등 **'View'** 역할에 집중하게 됩니다.

---

## 1. 개요 및 설계 원칙

- **Logic Extraction**: 시각화(Rendering)가 아닌 데이터 가공(Data Processing) 성격의 모든 메서드를 추출합니다.
- **Stateless Analysis**: 가능한 분석 함수들을 `DropSimulator`나 `DropSimResult`를 인자로 받는 순수 함수(Pure Functions) 형태로 정의하여 결합도를 낮춥니다.
- **Consistency**: 사용자 요청에 따라 UI의 모든 기능과 외형은 100% 동일하게 유지합니다.

---

## 2. 모듈 분리 및 역할 정의

### 2.1. `whts_postprocess_analysis.py` (New Module)
- **Role**: UI에서 필요로 하는 복잡한 데이터 가공 및 수치 해석 도구 모음.
- **추출 대상**:
    - `get_contour_grid_data`: 특정 시점의 블록 데이터를 2D 그리드 및 물리 좌표로 변환하는 로직 (`_get_contour_grid_at` 대체).
    - `apply_psr_surface_fit`: PSR 엔진을 이용한 고해상도 그리드 생성 로직 (`_draw_single_contour` 내 보간부).
    - `extract_global_metrics_summary`: 전체 부품의 PBA, RRG 피크 시점 및 값을 추출하는 통계 로직 (`_refresh_global_summary` 내부 루프).
    - `detect_critical_events`: 시뮬레이션 결과에서 주요 물리 이벤트를 매핑하는 로직.

### 2.2. `whts_postprocess_ui.py` (Modified)
- **Role**: 사용자 인터페이스(Tkinter) 구성 및 Plotting(Matplotlib) 호출.
- **Change**: 추출된 함수들을 임포트하여 `self` 참조 대신 외부 함수 호출로 전환.
- **Benefit**: 약 2,000라인의 코드 중 중복되거나 비대했던 연산부 약 400~500라인이 경량화됩니다.

---

## 3. 리팩토링 단계

1. **Phase 1: 신규 모듈 생성 및 함수 정의**: `whts_postprocess_analysis.py`를 생성하고 핵심 함수들을 옮깁니다.
2. **Phase 2: PSR 로직 이전**: 기존 `_draw_single_contour` 내부에 하드코딩된 PSR 회귀 로직을 분석 모듈로 이관합니다.
3. **Phase 3: 요약 테이블 데이터 생성부 이전**: 복잡한 루프와 조건문이 섞인 요약 데이터 생성 로직을 독립시킵니다.
4. **Phase 4: 무결성 검증**: UI 실행 및 애니메이션 작동 시 데이터가 이전과 동일하게 표시되는지 확인합니다.

---

> [!IMPORTANT]
> **사용자 피드백 요청**
> - **함수 명칭**: 분석팀(Analysis) 접두어를 붙인 `whts_postprocess_analysis.py` 명칭이 마음에 드시나요? 
> - **데이터 객체 활용**: 분석 함수가 `self.sim` 전체를 인자로 받는 방식과, 필요한 특정 지표만 리스트로 받는 방식 중 선호하시는 설계가 있으신가요? (전자가 코드가 간결하고 확장에 유리합니다.)

위 계획에 따라 리팩토링을 진행하겠습니다. 승인 혹은 의견 주시면 바로 착수하겠습니다.

---
**WHTOOLS** 드림
