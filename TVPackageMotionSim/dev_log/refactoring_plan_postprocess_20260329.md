# [WHTOOLS] Post-Processing UI 코드 분리 및 구조 최적화 계획

현재 약 2,000라인에 달하는 `whts_postprocess_ui.py`의 복잡도를 낮추고 유지보수성을 높이기 위해, 역할을 기준으로 모듈을 분리하는 리팩토링을 제안합니다.

---

## 1. 개요 및 설계 원칙

- **Separation of Concerns (SoC)**: UI 배치(Layout), 데이터 처리(Analysis), 시각화(Plotting) 로직을 분리합니다.
- **WHTS Prefix Consistency**: 모든 신규 파일에 `whts_` 접두어를 부여하여 프로젝트 일관성을 유지합니다.
- **Minimizing Regression**: 기존 `PostProcessingUI` 클래스의 인터페이스를 유지하여 `whts_engine.py` 등 외부 호출부의 수정을 최소화합니다.

---

## 2. 모듈 분리 구조 (제안)

### 2.1. `whts_postprocess_ui.py` (Main Entry)
- **Role**: 메인 창 구성, 메뉴바, 탭 컨테이너 관리.
- **Contents**: `PostProcessingUI` 클래스 메인 정의 및 탭 전환 로직.

### 2.2. `whts_postprocess_tabs.py` (UI Widgets)
- **Role**: 각 탭(Summary, 2D Contour, Structural, Kinematics)의 내부 위젯 구성.
- **Contents**: `SummaryTab`, `ContourTab` 등 개별 클래스화.

### 2.3. `whts_postprocess_plots.py` (Visualization)
- **Role**: Matplotlib을 이용한 그래프 및 컨투어 렌더링 로직.
- **Contents**: `_draw_single_contour`, `_update_kinematic_plots` 등 그래프 생성 함수들을 캡슐화.

### 2.4. `whts_postprocess_ssr.py` (Advanced Engine)
- **Role**: PSR/SSR 관련 고차원 연산 및 팝업 대화상자 관리.
- **Contents**: `SSRAnalyzerDialog` 클래스 및 PSR 보간 유틸리티.

---

## 3. 리팩토링 단계

1. **Phase 1: 기능별 메서드 그룹화**: 현재 파일 내에서 주석을 이용해 위 구분에 따라 메서드들을 논리적으로 묶습니다.
2. **Phase 2: SSR 로직 분리**: 가장 복잡한 PSR/SSR 연산부와 대화상자 로직을 `whts_postprocess_ssr.py`로 우선 분리합니다.
3. **Phase 3: 시각화 로직 분리**: `Plotter` 성격의 메서드들을 `whts_postprocess_plots.py`로 이동합니다.
4. **Phase 4: 통합 테스트**: 전체 시뮬레이션 종료 후 UI가 정상적으로 로드되고 그래프가 그려지는지 확인합니다.

---

> [!IMPORTANT]
> **사용자 피드백 요청**
> - **분리 강도**: 위와 같이 4개의 파일로 세밀하게 나누는 것이 좋을까요, 아니면 (UI + Plots)와 (Analysis) 정도로 2~3개 파일로 큼직하게 나누는 것을 선호하시나요?
> - **클래스 구조**: 기존처럼 하나의 큰 클래스에서 분할된 모듈의 함수를 호출하는 방식이 관리하기 편하실지, 아니면 각 탭을 독립된 클래스로 완전히 분리하는 객체지향적 구조를 선호하실지 궁금합니다.

위 계획에 대해 의견 주시면 바로 리팩토링의 첫 단계를 시작하겠습니다.

---
**WHTOOLS** 드림
