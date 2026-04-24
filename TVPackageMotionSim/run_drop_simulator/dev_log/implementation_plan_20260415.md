# Implementation Plan: 상세 변형률(Strain) 및 주응력/주변형률 성분 추출 기능 추가

`PlateMechanicsSolver` 내의 물리 필드 계산 파이프라인을 고도화하여, 설계자가 구조적 취약점을 정밀하게 파악할 수 있도록 다양한 변형률 및 주성분 데이터 추출 기능을 추가합니다.

## User Review Required

> [!NOTE]
> **주성분(Principal Component) 정의**
> 모든 주변형률 및 주응력은 평판의 표면에서의 주응력 상태(Plane Stress)를 가정하여 텐서 계산을 통해 산출됩니다.

## Proposed Changes

### [Plate Analytics]

#### [MODIFY] [plate_by_markers.py](file:///c:/Users/GOODMAN/WHToolsBox/plate_by_markers.py)

1.  **`PlateMechanicsSolver.compute_mechanics_fields_batch` 로직 확장**:
    -   **방향별 변형률(Directional Strains)**: `Strain XX [mm/mm]`, `Strain YY [mm/mm]`, `Strain XY [mm/mm]` 추가.
    -   **주변형률(Principal Strains)**: 2D 스트레인 텐서의 고유값(Eigenvalues) 계산을 통해 `Strain Max Principal [mm/mm]` 및 `Strain Min Principal [mm/mm]` 추가.
        -   $\epsilon_{avg} = (\epsilon_x + \epsilon_y) / 2$
        -   $R_\epsilon = \sqrt{((\epsilon_x - \epsilon_y) / 2)^2 + (\gamma_{xy} / 2)^2}$
        -   $\epsilon_{max, min} = \epsilon_{avg} \pm R_\epsilon$

2.  **출력 키(Key) 명칭 표준화**:
    -   기존 `Principal Max [MPa]` 등과의 일관성을 위해 `Strain Max Principal [mm/mm]` 등으로 명명합니다.

## Verification Plan

### Automated Tests
- `python plate_by_markers.py` 실행을 통한 런타임 에러 여부 확인.
- JAX `vmap` 연산의 배치 처리 정합성 확인.

### Manual Verification
- GUI 콤보박스 아이템 업데이트 확인.
- 주성분 데이터 시각화 결과가 물리적 직관(굽힘 방향 등)과 일치하는지 확인.
