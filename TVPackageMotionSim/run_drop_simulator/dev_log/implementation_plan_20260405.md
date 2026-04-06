# WHTOOLS Dashboard 3D View 개선 및 오류 수정 계획

## 개요
3D View의 시각적 편의성과 기능성을 높이기 위해 색상 그룹화, 뷰 제어, 지면(Floor) 자동 설정 기능을 추가하고, 기존 코드의 런타임 오류를 수정합니다.

## 제안된 변경 사항

### 1. 시각화 엔진 및 로직 (`plate_by_markers_v2.py`)

#### Body Color 그룹화 로직 수정
- **현상**: "Body Color" 선택 시 파트 인덱스별로 색상을 부여함 (Face Color와 동일하게 동작).
- **수정**: 파트 이름의 앞부분(Prefix, `_` 기준 분리)이 같은 파트들끼리는 동일한 색상을 가지도록 맵핑 테이블을 생성하여 적용합니다.

#### Colormap 설정 오류 수정 (`AttributeError`)
- **수정**: `a['mesh'].mapper.set_scalar_range(...)`를 VTK 표준 메서드인 `SetScalarRange(min, max)` 또는 PyVista 속성인 `scalar_range = (min, max)`로 변경합니다. (Line 314)

#### Floor 크기 자동 계산
- **수정**: `QtVisualizerV2` 초기화 또는 지면 가시성 전환 시, 모든 활성 파트의 바운딩 박스를 계산합니다. 바운딩 박스 대각선 길이($L_{diag}$)를 구하고, 지면의 가시적 크기를 $1.5 \times L_{diag}$로 설정합니다.

#### 컨텍스트 메뉴 확장
- **Perspective View**: `v_int.disable_parallel_projection()` (Perspective)과 `v_int.enable_parallel_projection()` (Orthographic)을 토글하는 체크 메뉴 추가.
- **Isometric View (4방향)**: 기본 Isometric 외에 남동, 남서, 북동, 북서 방향의 아이소메트릭 뷰를 선택할 수 있는 서브메뉴 추가.
- **Floor Direction**: 지면의 법선 방향을 X, Y, Z축으로 설정할 수 있는 메뉴 추가. 지면 위치는 바운딩 박스의 해당 방향 최소값에 위치하도록 조정합니다.

## 사용자 리뷰 필요 사항
- [IMPORTANT] 파트 이름을 구분하는 기준은 현재 `_`를 사용하고 있습니다. (예: `Body1_Part1`, `Body1_Part2`). 만약 다른 구분자가 사용 중이라면 알려주시기 바랍니다.
- Floor의 방향을 바꿀 때, 지면의 위치(Offset)를 바운딩 박스의 어느 지점에 맞출지 결정이 필요합니다. 현재는 최소값(bottom)에 맞추는 것을 기본으로 설계합니다.

## 검증 계획

### 자동 테스트 및 수동 확인
1. **Body Color 그룹화 확인**: 동일한 접두사를 가진 파트들이 같은 색상으로 표시되는지 3D 뷰에서 확인.
2. **뷰 전환 확인**: 컨텍스트 메뉴를 통해 Perspective/Orthographic 전환 및 4방향 Isometric 전환이 정상 작동하는지 확인.
3. **Floor 크기 및 방향 확인**: 지면 크기가 모델 크기에 맞춰 자동으로 커지는지, X/Y/Z 방향 변경 시 지면이 올바르게 재배치되는지 확인.
4. **오류 수정 확인**: Field 값 변경 시 더 이상 `AttributeError`가 발생하지 않고 스칼라 바 및 색상이 정상 업데이트되는지 확인.
