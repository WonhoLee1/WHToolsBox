# Implementation Plan - LTL Drop Mode Orientation Mapping (2026-04-24)

## 1. 개요
LTL(Less than Truckload) 낙하 시뮬레이션 모드에서 사용자 정의 면(Face) 번호와 물리적 방향 간의 맵핑을 조정합니다. 이는 ISTA 규격 또는 특정 시험 표준에 맞추기 위함입니다.

## 2. 목표 매핑 설정
사용자 요청에 따라 `drop_mode="LTL"`일 때 다음과 같이 정의합니다:
- **Face 1 (Top)**: +Y 방향 `[0, 1, 0]`
- **Face 2 (Back)**: -Z 방향 `[0, 0, -1]` (Rear)
- **Face 3 (Bottom)**: -Y 방향 `[0, -1, 0]`
- **Face 4 (Front)**: +Z 방향 `[0, 0, 1]`
- **Face 5 (Right)**: +X 방향 `[1, 0, 0]` (기존 유지)
- **Face 6 (Left)**: -X 방향 `[-1, 0, 0]` (기존 유지)

## 3. 수정 대상 파일
- `TVPackageMotionSim/run_discrete_builder/whtb_utils.py`: `ltl_map` 사전(dict) 데이터 수정.

## 4. 작업 절차
1. `whtb_utils.py` 파일을 열어 `parse_drop_target` 함수 내의 `ltl_map`을 확인합니다.
2. 기존의 `ltl_map` 구성을 위 목표 매핑에 맞게 수정합니다.
3. 수정 후 `run_drop_simulation_cases_v6.py`를 실행하여 `drop_mode="LTL"` 및 `drop_direction="Corner 2-3-5"` 설정 시 의도한 코너(Back-Bottom-Right)로 낙하 자세가 잡히는지 확인합니다.

## 5. 기대 효과
- 사용자가 직관적으로 인지하는 면 번호(1~6)가 시뮬레이션 타겟 점 계산에 정확히 반영됨.
- LTL 특유의 선적/낙하 방향성을 코드 레벨에서 명확히 제어 가능.
