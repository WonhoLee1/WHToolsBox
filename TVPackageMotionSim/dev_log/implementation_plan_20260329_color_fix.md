# Implementation Plan - MuJoCo Cushion Localization Fix (Refined)

본 계획은 유저의 추가 요청에 따라, 쿠션의 시각적 강조 및 소성 변형 추적 대상을 **8개의 꼭짓점과 Depth 방향의 4개 모서리(Z-axis Edges)**로 국한하도록 수정합니다.

## User Review Required

> [!IMPORTANT]
> - 강조 대상 정의: **(ix == 0 or ix == nx-1) AND (iy == 0 or iy == ny-1)** 인 블럭들입니다.
> - 이는 박스의 가로(X)와 세로(Y)가 끝단인 위치로, Depth(Z) 방향으로 길게 이어진 4개의 모서리 기둥을 의미합니다. (8개 꼭짓점 포함)
> - 이 가이드에 따라 `is_edge_block` 대신 `is_corner_block` (또는 유사 명칭)을 사용하여 시각화를 제한합니다.

## Proposed Changes

### 1. [Builder Package] `run_discrete_builder`

#### [MODIFY] [whtb_models.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_models.py)
- `is_corner_block(self, i, j, k)` 추가: 
    - `(i == 0 or i == nx-1) and (j == 0 or j == ny-1)` 조건 적용.
    - 이 조건은 8개 꼭짓점과 그 사이의 Depth 방향 모서리 블록을 모두 포함합니다.

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- XML 생성(`build_discrete_body`) 시:
    - `is_corner_block`인 경우 `contact_bcushion_edge` 클래스 부여.
    - 그 외의 모서리(상하 모서리 등)는 일반 `contact_bcushion` 또는 별도 분석용 클래스 부여.

---

### 2. [Simulator Package] `run_drop_simulator`

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `_init_plasticity_tracker`:
    - 지오메트리 클래스명이 `contact_bcushion_edge` (또는 `_edge` 접미사)인 경우에만 `geom_state_tracker`에 등록.
    - 등록과 동시에 해당 블록의 색상을 **노란색(`[1.0, 1.0, 0.0, 1.0]`)**으로 초기화.
- `_apply_plasticity_v2`: 등록된 블록에 대해서만 소성 변형 물리 연산 수행.

#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)
- `apply_rank_heatmap`: 초기 색상이 노란색인 블록들이 히트맵 적용 시에도 이질감 없이 표현되도록 로직 점검.

## Open Questions

- 현재 좌표계에서 Depth가 Z축인 것이 확실시되므로 `(ix, iy)` 고정 조건으로 진행합니다. 만약 좌표계가 다시 바뀌었다면(예: Depth가 Y) 조건 수정이 필요합니다.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_v4.py` 실행 시 초기 화면에서 Depth 방향 모서리 4줄만 노란색으로 보이는지 확인.
- 생성된 XML 파일의 `geom` 클래스 할당 여부 확인.

### Manual Verification
- 시뮬레이션 중 해당 모서리 블록들의 변형 여부와 색상 변화 시각적 검토.
