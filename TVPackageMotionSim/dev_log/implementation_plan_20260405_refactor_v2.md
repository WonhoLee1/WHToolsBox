# [Refactor] Configuration 시스템 재구축 및 XML 생성 안정화 (V2)

`test_run_case_1`의 설정을 완벽히 흡수하면서도, 수치적 불안정성을 원천 차단하는 정밀 설정 시스템을 구축합니다.

## User Review Required

> [!IMPORTANT]
> 이번 수정은 단순 네이밍 변경을 넘어, 시뮬레이션 폭발의 주범인 **데이터 타입 불일치(Float in Bitmask)**를 완벽히 해결합니다. 또한 모든 접촉 물성이 최신화된 상태로 XML에 기록되도록 강제하는 동기화 로직이 추가됩니다.

## Proposed Changes

### [Component] run_discrete_builder

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- **표준 네이밍 적용**: `opencell_`, `opencellcoh_`, `chassis_`
- **동기화 로직 도입**: `sync_phys_config()` 구현 (solref 문자열 조립 및 mat_ 맵 갱신 전담)
- **Case 1 사양 이식**: `chassis damping (0.3)`, `cushion damping (0.8)` 등을 기본값으로 설정

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- 변경된 `config` 키(`opencell_div` 등)에 맞춰 참조 코드 업데이트
- XML 템플릿의 `option` 태그 내 누락된 속성(`noslip_iterations` 등) 보강

### [Component] Testing

#### [MODIFY] [verify_refactor.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/verify_refactor.py)
- XML 생성 여부 및 생성된 파일 내 주요 물리 문자열(`solref`) 존재 여부 물리 검사 추가

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v4.py` (Case 1) 실행 및 안정성 확인
- `verify_refactor.py`를 통한 키 매핑 무결성 점검
