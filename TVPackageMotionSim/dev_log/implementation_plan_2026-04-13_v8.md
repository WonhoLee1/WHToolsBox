# [Goal] Restoration of Motion Analysis Accuracy (Corner Indexing Fix)

쿠션 코너 좌표가 예상과 다르게 출력되는 문제를 해결하기 위해, 시뮬레이션 엔진의 기준 좌표계(Root Body) 식별 로직을 개선하고 부품 명명 규칙 변화에 따른 인덱싱 정렬 문제를 수정합니다.

## User Review Required

> [!CAUTION]
> **Root Body 식별 실패 가능성**: 현재 `root_id`를 "chassis"라는 고정 문자열로만 찾고 있어, 빌더에 의해 "bchassis" 등으로 이름이 변경된 경우 시뮬레이션의 기준점이 World Origin(0,0,0)으로 고정되는 심각한 논리 오류를 발견했습니다. 이를 유연한 탐색 방식으로 수정합니다.

## Proposed Changes

### 1. Simulation Engine (Root Identification)

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `setup()` 내 `root_id` 할당 로직 수정:
  - "chassis" 문자열이 포함된 바디를 우선 검색 (`bchassis`, `chassis_main` 등 대응).
  - 식별된 Root Body를 기준으로 코너 운동학(`compute_corner_kinematics`)을 계산하도록 보장합니다.

---

### 2. Coordinate Mapping & Marker Extraction

#### [MODIFY] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- `extract_face_markers()` 로직 보완:
  - 부품이 단일 바디(Unified)로 구성된 경우에도 해당 바디의 기하학적 형상(Geom Size)을 바탕으로 8개 모서리 마커를 정확히 생성하도록 로직을 강화합니다.
  - 인덱스 기반 정렬 시 `max_i`, `max_j`, `max_k`가 0인 경우(단일 블록)에 대한 예외 처리를 추가합니다.

---

### 3. Data Integrity & Verification

- `v6` 파이프라인에서 추출된 PKL 데이터 내의 `corner_pos_hist`가 글로벌 좌표계를 정확히 반영하는지 확인하는 로깅을 추가합니다.

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v6.py` 재실행:
  - 터미널 로그에 "Root Body Identified: (name)"이 출력되는지 확인.
  - V2 대시보드에서 쿠션의 코너 포인트들이 정적인(Static) 상태가 아닌, 낙하시의 가속도와 변위를 정상적으로 추종하는지 시각적 확인.

### Manual Verification
- 3D 대시보드의 'Motion Tracking' 탭에서 코너 점들이 박스의 외곽선과 일치하는지 확인.
