# [Refactor] plate_by_markers_v2.py 가독성 및 유지보수성 강화

`plate_by_markers_v2.py`는 기계공학적 해석과 고성능 JAX 연산이 집약된 핵심 모듈입니다. 현재의 축약된 코드를 해체하고, 상세한 주석과 구조화된 메서드 설계를 통해 "가독성이 높고 확장이 쉬운" 코드로 고도화합니다.

## User Review Required

> [!IMPORTANT]
> **핵심 로직 유지**: JAX 기반의 다항식 기저 행렬 생성 및 판 이론(Kirchhoff/Mindlin) 방정식의 해 산출 로직은 검증된 상태이므로, 수학적 무결성을 유지하면서 코드의 형태만 개선합니다.

## Proposed Changes

### 1. 코드 가독성 극대화 (Code Readability)
- **축약 표현 해제**: 한 줄에 여러 명령을 넣는 세미콜론(`;`) 방식을 모두 제거하고 표준 PEP 8 스타일로 줄바꿈을 적용합니다.
- **명명 규칙 현대화**:
  - `m_raw` -> `marker_raw_data`
  - `q_loc` -> `local_coordinates`
  - `sol` -> `mechanics_solver`
  - `bs` -> `batch_size`
  - `info` -> `actor_info` (UI context)
- **매직 넘버 제거**: 변형 해석 시 사용되는 상수(예: 5/6, 1.2 등)를 명명된 상수로 정의하여 의미를 명시합니다.

---

### 2. 전문가급 풍부한 주석 (Rich Documentation)
- **WHTOOLS 페르소나 적용**: 각 클래스 상단에 기계공학적 배경지식을 포함한 Docstring을 작성합니다.
- **수학식 시각화**: 기계공학 엔지니어가 코드를 보며 수식을 연상할 수 있도록 주석 내에 Latex 형식의 수식 설명을 추가합니다.
  - 예: `D = (E * h^3) / (12 * (1 - nu^2))` (Bending Stiffness)
- **JAX 특이사항 명시**: `@jit`와 `vmap`이 적용된 함수는 최적화 이유와 입력/출력 데이터의 Shape 변화를 주석으로 상세히 기술합니다.

---

### 3. 구조적 리팩토링 (Structural Cleanup)
- **UI 클래스 분리**: `QtVisualizerV2`의 거대한 `__init__` 메서드를 `_init_3d_canvas`, `_init_2d_controls`, `_setup_animation_toolbar` 등으로 모듈화합니다.
- **가독성 향상을 위한 로직 함수화**: 복잡한 조건 처리나 데이터 가공 로직을 별도의 도우미 메서드(Helper methods)로 추출합니다.

---

## Open Questions

- **[IMPORTANT]** 리팩토링 과정에서 변수명을 대폭 변경(`m_raw` -> `marker_raw_data`)하면 기존에 해당 파일을 직접 보시던 분들께 혼동을 드릴 수 있습니다. 적극적인 수정을 원하시는지, 아니면 중요한 부분만 수정하기를 원하시는지 확인 부탁드립니다.
- **[QUESTION]** 특정 물리 엔진 코드(JAX 영역)에 대해 수식 유도 과정을 담은 별도의 기술 문서(Markdown)를 추가로 작성해 드릴까요?

## Verification Plan

### Automated Tests
- 리팩토링 전/후의 `test_run` 결과(변위 및 응력 데이터)를 수치적으로 비교하여 오차가 0임을 확인합니다.
- JAX 컴파일 시간 및 실행 속도에 저하가 없는지 벤치마크를 수행합니다.

### Manual Verification
- 대시보드의 모든 버튼, 메뉴, 슬라이더의 동작을 테스트하고 시각화 품질(폰트, 레이아웃)을 재점검합니다.
