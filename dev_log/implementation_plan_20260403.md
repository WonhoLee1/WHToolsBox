# Refactoring Plan: WHTOOLS Structural Analyst (v8-Pro)

현재의 고성능 JAX 연산 로직을 유지하면서, 코드의 **가독성(Readability)**과 **재사용성(Reusability)**을 극대화하기 위한 전면 리팩토링을 수행합니다.

## User Review Required

> [!IMPORTANT]
> - **구조적 변경**: 기존의 단순 스크립트 형태에서 `PlateConfig`와 같은 설정 클래스 도입 등 객체지향 구조가 강화됩니다.
> - **문서화 표준**: 모든 함수에 PEP 257 표준에 따른 상세 docstring이 추가됩니다.

## Proposed Changes

### [Plate Analytics Engine]

#### [MODIFY] [plate_by_markers.py](file:///c:/Users/GOODMAN/WHToolsBox/plate_by_markers.py)
1. **설정 클래스 도입 (`PlateConfig`)**: 재질 물성치(E, nu, t) 및 시뮬레이션 파라미터를 체계적으로 관리.
2. **구문 확장**: 세미콜론(`;`)으로 연결된 모든 압축 구문을 단일 라인으로 해제하여 가독성 확보.
3. **상세 문서화**: 각 함수의 인자(Parameters), 반환값(Returns), 물리적 의미를 상세히 기술.
4. **UI 클래스 고도화 (`QtVisualizer`)**: 
   - UI 초기화(`_init_ui`), 레이아웃 설정(`_setup_layout`), 시그널 연결(`_connect_signals`)로 논리 분리.
   - 매직 넘버(픽셀 크기, 색상 코드 등)를 클래스 상수로 처리.

## Open Questions

- 특별히 더 강조하고 싶은 상수가 있나요? (예: 특정 재질 리스트, 기본 뷰포트 각도 등)

## Verification Plan

### Automated Tests
- `python plate_by_markers.py`를 실행하여 1000프레임 해석이 기존과 동일한 성능(1.1초 내외)으로 수행되는지 확인.
- UI 컨트롤(슬라이더, 콤보박스, 재생 버튼)의 정상 동작 확인.

### Manual Verification
- 코드 리뷰를 통해 세미콜론 사용 여부 및 docstring 포함 여부 전수 확인.
- 변수명과 클래스 구조가 WHTOOLS Writing Rules를 준수하는지 확인.
