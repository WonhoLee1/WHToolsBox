# 좌표계 및 모델 빌더 로직 원상 복구 계획 (Rollback to Z-Depth)

최근 적용된 MuJoCo 좌표계 표준화(Z=Height, Y=Depth) 작업을 취소하고, 기존의 관습적인 좌표계(Z=Depth, Y=Height)로 모델 빌더와 시뮬레이션 로직을 원상 복구합니다. 이는 기존 모델 데이터와의 호환성을 유지하고, 적층 방향이 Z축이었던 초기 모델링 구조를 회복하기 위함입니다.

## User Review Required

> [!IMPORTANT]
> **좌표계 변경 사항 (Rollback 내역):**
> - **X축**: 가로 (Width) - 유지
> - **Y축**: 높이 (Height) - (복구됨)
> - **Z축**: 두께 (Depth) - (복구됨)
> - **적층 방향**: Y축에서 **Z축**으로 다시 변경됩니다.
> - **중력 방향**: MuJoCo 기본 설정에 따라 Z축(-9.81)을 유지할 경우, 모델이 '누운 상태'로 시뮬레이션될 수 있습니다. (기존 로직 확인 결과 Z축 중력을 그대로 사용했었으므로 이를 따릅니다.)

## Proposed Changes

### [run_discrete_builder]

#### [MODIFY] [__init__.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/__init__.py)
- `get_default_config`의 `box_div`, `cush_div` 등의 인덱스를 `[W, H, D]` 순서로 복구합니다. (현재 `[W, D, H]`)
- `create_model` 함수 내부의 부품 배치 로직을 Z축 적층 방식으로 수정합니다.
    - OpenCell, Chassis 등의 오프셋을 `[0, 0, offset_z]` 형태로 복구합니다.
- `parse_drop_target` 함수에서 각 면(Face)의 벡터 매핑을 Z-Depth 기준으로 롤백합니다.
- `BPaperBox`, `BCushion` 등의 `is_cavity` 로직에서 Height/Depth 체크 축을 교체합니다.

---

### [TVPackageMotionSim]

#### [MODIFY] [run_drop_simulation_v3.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_v3.py)
- `compute_corner_kinematics` 함수에서 `y`를 Height, `z`를 Depth로 처리하도록 보장합니다. (이미 일부 반영되어 있으나 빌더와의 정렬을 재확인)
- 시뮬레이션 루프 내의 좌표 의존적 로직(예: Squeeze Film, Plasticity)이 Z-Depth 기준으로 동작하는지 확인합니다.

## Open Questions

- **빌더 코드 구조 복구**: 현재 414라인으로 간소화된 빌더 코드를 백업본(1471라인) 수준의 상세 기능(공기 저항, 상세 소성 로그 등)을 포함한 상태로 복구하시겠습니까? 아니면 현재의 간소화된 구조에서 좌표계만 변경하시겠습니까?
    - *제안*: "원상 복구"의 의미를 고려하여 백업본의 주요 로직(상세 물리 설정 등)을 다시 살리는 방향으로 진행하겠습니다.

## Verification Plan

### Automated Tests
- `run_discrete_builder`를 통해 XML 생성 후, `BPaperBox`와 `AssySet`이 Z축 방향으로 정상적으로 적층되는지 확인.
- `run_drop_simulation_v3.py`를 실행하여 Front/Rear 낙하 시 충격 지점이 Z축(Depth) 방향의 끝단으로 설정되는지 확인.

### Manual Verification
- 생성된 XML을 MuJoCo Viewer로 확인하여 시큐리티(OpenCell)가 Z축 방향을 향하고 있는지 확인.
