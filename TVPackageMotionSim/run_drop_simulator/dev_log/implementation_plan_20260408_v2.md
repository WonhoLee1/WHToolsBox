# Implementation Plan - Dashboard UI Layout Refinement

[WHTOOLS] 대시보드의 심미성과 공간 효율성을 높이기 위해 레이아웃을 미세 조정합니다. 탭 컨트롤을 풀다운 메뉴처럼 슬림하게 구성하고, 데이터 입력부의 직관성을 개선합니다.

## User Review Required

> [!IMPORTANT]
> 1. **탭 컨트롤 레이아웃**: 탭의 외부 마진을 1px로 설정하고 고정 높이(200px)를 제거하여 내부 컨텐츠에 맞게 높이가 자동 조절되도록 합니다.
> 2. **화면 분할 비율**: 3D 뷰 및 2D 그래프 영역의 분할 비율을 **2:3**으로 조정합니다.
> 3. **Min/Max 체크박스 재배치**: `[X] Min: [값]` / `[X] Max: [값]` 형태로 체크박스를 각 라벨 왼쪽에 배치하여 공간을 절약하고 가독성을 높입니다.
> 4. **Aesthetics 그룹 통합**: 별도의 그룹이었던 `Background` 설정을 `View & Deformation` 그룹으로 통합합니다.

## Proposed Changes

### [UI] [whts_multipostprocessor_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_ui.py)

#### 1. 대시보드 메인 레이아웃 수정 (`_init_ui`)
- `tl.setContentsMargins(1, 1, 1, 1)` 설정 (탭 컨트롤 외부 마진 최소화).
- `self.ct.setFixedHeight(200)` 삭제 (가변 높이 적용).
- `self.split.setStretchFactor(0, 2)`, `self.split.setStretchFactor(1, 3)` 적용 (2:3 비율).

#### 2. 3D 제어부 레이아웃 최적화 (`_init_3d_controls`)
- **View & Deformation (g1)**:
    - 기존 `View`, `Scale`, `Perspective`에 더해 `Background (BG)` 콤보박스를 추가 배치.
- **Field & Range (g2)**:
    - **Row 0**: Field (`cmb_3d`), Range (`cmb_l`)
    - **Row 1**: 새로운 수평 레이아웃을 사용하여 `[ch_min] Min: [sp_min] [ch_max] Max: [sp_max]` 순서로 배치.
    - 체크박스의 텍스트는 제거하여 아이콘/체크 상태만 노출.
- **Aesthetics (g3)**: 삭제.

## Open Questions

- 탭의 높이를 가변으로 할 경우, 내부 위젯의 마진(Contents Margins)이 너무 크면 여전히 공간을 많이 차지할 수 있습니다. 각 탭 내부 레이아웃의 마진도 2~5px 수준으로 축약할까요? (현재는 기본값 사용 예정)

## Verification Plan

### Automated Tests
- `python whts_multipostprocessor.py` 실행 시 탭 영역이 슬림하게 정렬되는지 확인.

### Manual Verification
- **Min/Max 제어**: 체크박스를 클릭했을 때 3D 뷰에 마커 텍스트(Note)가 정상 노출/제거되는지 확인.
- **레이아웃**: 탭 높이가 내부 위젯 크기에 딱 맞게 줄어드는지 확인.
- **Aesthetics**: View 그룹 내에서 배경색 변경이 잘 작동하는지 확인.
