# [Goal] VTKHDF Export Crash Fix & v6 Pipeline Mapping Stabilization

v6 파이프라인에서 컴포넌트 이름(예: `bcushion` vs `cushion`) 불일치로 인해 분석 대상(Analyzers)이 0개가 되어 VTKHDF 내보내기 시 `np.concatenate` 오류가 발생하는 문제를 해결합니다.

## User Review Required

> [!IMPORTANT]
> **유연한 명칭 맵핑 (Flexible Mapping)**: 
> "cushion, chassis 등 이름이 포함되어 있다면 그것을 사용"하라는 가이드에 따라, `whts_mapping.py`에서 부분 일치(Sub-string matching)를 허용하도록 수정합니다. 이를 통해 `bcushion` 등 접두사가 붙은 이름으로도 시뮬레이션 데이터에 접근할 수 있게 됩니다.

## Proposed Changes

### 1. Mapping Logic (Fuzzy/Partial Match)

#### [MODIFY] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- `extract_face_markers` 함수 수정:
  - 요청된 `part_name`과 저장된 `result.components`의 키들을 유연하게 매칭합니다.
  - 예: `bcushion` 요청 시 `cushion` 키를 찾거나, `cushion` 키가 `bcushion` 문자열 내부에 포함되어 있는지 확인하여 매칭 성공률을 높입니다.

---

### 2. Result Exporter

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `export_to_vtkhdf` 함수 수정:
  - `manager.analyzers`가 비어있을 경우 조기에 리턴하고 경고 메시지를 출력하여 `np.concatenate` crash를 방지합니다.

---

### 3. Simulation Engine (Stability)

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `_discover_components`에서 맵핑 시 디버깅용 로그 출력을 강화하여 어떤 키로 데이터가 저장되는지 명확히 표시합니다.

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v6.py` 실행:
  - `Analyzers: 4` (Box, Cushion, Chassis, Opencell) 가 정상적으로 인식되는지 확인.
  - `Result.vtkhdf` 파일이 생성되고 ParaView 대시보드가 성공적으로 팝업되는지 확인.

### Manual Verification
- `p` 키와 `backspace` 키를 사용한 리플레이 시에도 컴포넌트 데이터가 유실되지 않는지 확인.
