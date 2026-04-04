# Implementation Plan - Plate Assembly & Advanced Alignment (v2)

본 계획서는 `plate_by_markers.py`를 `v2` 버전으로 업그레이드하여 다중 파트 해석(Assembly), UI 기반 가시성 제어, 그리고 규격 기반 마커 정렬 기능을 구현하기 위한 상세 설계를 담고 있습니다.

## 1. 아키텍처 개편 (Assembly Manager)

기존의 `ShellDeformationAnalyzer`를 부품(Part) 단위로 취급하고, 이를 총괄하는 `PlateAssemblyManager` 클래스를 도입합니다.

- **`PlatePart` (구 `ShellDeformationAnalyzer`)**:
  - 개별 판의 물성(Thickness, Young's Modulus), 규격(Width, Height), 마커 오프셋을 보유.
  - 고유 ID와 이름을 부여하여 UI에서 식별 가능하도록 함.
- **`PlateAssemblyManager`**:
  - 여러 개의 `PlatePart` 인스턴스를 관리.
  - 전역 시간 축(Time steps)을 동기화하여 관리.
  - 모든 파트의 해석을 일괄 실행(`run_all()`)하고 결과를 집계.

## 2. 규격 기반 마커 정렬 알고리즘 (Alignment Optimization)

사용자가 입력한 판의 가로(`W`), 세로(`H`) 및 마커의 모서리/변으로부터의 거리(`Offsets`)를 기준으로 마커를 가장 잘 설명하는 로컬 좌표계를 산출합니다.

- **알고리즘 요약**:
  1. 사용자가 정의한 오프셋 좌표 $P_{target} = (x_{off}, y_{off}, 0)$ 정의.
  2. 현재 마커의 전역 위치 $M_{global}$과의 관계에서 회전($R$)과 평행이동($T$)을 최적화.
  3. $\min \sum \| (M - T)R^T - P_{target} \|^2$ 문제를 Procrustes 분석법으로 해결.
  4. 이를 통해 마커가 판의 어느 위치에 부착되었는지를 역계산하여 해석의 정밀도를 높임.

## 3. UI/UX 고도화 (QtVisualizerV2)

다중 파트 시각화와 동적인 제어를 위해 UI를 확장합니다.

- **3D 뷰포트 (PyVista)**:
  - **다중 액터 관리**: 파트 수만큼 메쉬와 마커 액터를 생성하고 `dict`로 관리.
  - **우클릭 컨텍스트 메뉴**:
    - `Parts` 메뉴 트리 생성.
    - 각 파트별 `Checked` 상태로 `Show/Hide` 토글.
    - `Show All` / `Hide All` 일괄 제어 기능.
- **2D 그래프 (Matplotlib)**:
  - **개별 파트 선택**: F1, F2(컨투어), C1, C2(커브) 각각의 차트 위에 `Part Selector` (ComboBox) 추가.
  - 이를 통해 특정 차트에서는 1번 파트의 Stress를, 다른 차트에서는 2번 파트의 변위를 동시에 관찰 가능.

## 4. 검증 시나리오 (Cube Drop Test)

6개 면(Face)을 가진 박스 모델을 시뮬레이션하여 기능을 검증합니다.

- **데이터 생성**:
  - 6개의 `PlatePart` 생성 (Top, Bottom, Front, Back, Left, Right).
  - 전체 박스의 강체 운동(자유낙하 + 회전 + 지면 충돌/전복) 생성.
  - 각 면에 부착된 마커들의 전역 궤적 산출.
- **테스트 포인트**:
  - 다중 파트가 동시에 렌더링되는지 확인.
  - 전복 상황에서도 로컬 배향이 규격(W, H)에 맞게 유지되는지 확인.

## 5. 단계별 작업 계획 (Tasks)

1. **[Phase 1]** 아키텍처 리팩토링 및 `AssemblyManager` 구현.
2. **[Phase 2]** 규격/오프셋 기반 마커 정렬(Alignment) 로직 추가.
3. **[Phase 3]** `QtVisualizerV2` 다중 파트 렌더링 및 컨텍스트 메뉴 구현.
4. **[Phase 4]** 2D 차트별 파트 선택 UI 및 로직 구현.
5. **[Phase 5]** 6면체(Cube) 시뮬레이션 예제 작성 및 최종 검증.

## Open Questions

- 마커 오프셋 입력 방식에 대해: 각 마커별로 $(x, y)$ 좌표를 직접 줄 것인지, 혹은 특정 패턴(예: 3x3 그리드)을 가이드할 것인지에 대한 의견이 필요합니다. (우선 리스트 형태의 수동 입력을 기본으로 하겠습니다.)
- 6면체 예제에서 각 면의 마커 수는 동일하게 9개(3x3)로 설정해도 괜찮을까요?

---
> [!IMPORTANT]
> 본 계획 승인 시 `plate_by_markers_v2.py` 파일 생성을 시작합니다.
