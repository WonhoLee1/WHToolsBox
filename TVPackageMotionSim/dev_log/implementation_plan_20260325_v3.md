# Implementation Plan: 소성 변형 방향성 동적 개선

## Goal Description
현재 쿠션의 소성 변형이 낙하 방향과 관계없이 Z축(두께 방향)으로만 일어나는 문제를 해결합니다. 접촉 시의 법선 벡터를 분석하여 실제 압착이 일어나는 주축(X, Y, 또는 Z)을 자동으로 찾아내고, 해당 축을 기준으로 크기 축소와 위치 이동을 적용합니다.

## Proposed Changes

### [run_drop_simulation_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_drop_simulation_v2.py)

#### [MODIFY] `apply_plastic_deformation` 내부 로직
- `ma = 2`로 고정되어 있던 부분 제거.
- `hit['local_n']` (로컬 좌표계 접촉 법선 벡터)의 절대값이 가장 큰 성분을 `major_axis`로 선택.
- 시뮬레이션 로그에 활성화된 축(Axis 0, 1, 2)을 표시하여 디버깅 용이성 확보.

## Verification Plan

### Automated Tests
- `run_drop_simulation_v2.py`를 실행하여 `[Plasticity] Corner Activated` 로그에서 `Axis: 0` 또는 `Axis: 2` 등이 낙하 방향에 맞게 출력되는지 확인.
- 시뮬레이션 종료 후 `Deforming` 메시지의 수치가 실제 낙하 충격 방향의 블록 변형을 반영하는지 검증.

### Manual Verification
- 시각적으로 노란색으로 표시된 코너 블록이 바닥에 닿은 면을 중심으로 실제 "눌리는" 효과가 나타나는지 확인.
