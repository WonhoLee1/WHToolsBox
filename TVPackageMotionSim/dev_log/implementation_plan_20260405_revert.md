# [Revert] Configuration 및 Naming Refactoring 이전 상태로 복구

리팩토링 이후 발생한 수치적 불안정성(Explosion)을 해결하기 위해, 작업 직전의 백업 파일들을 복원하여 안전한 시뮬레이션 환경으로 롤백합니다.

## User Review Required

> [!IMPORTANT]
> 본 작업은 최근 진행한 네이밍 표준화(`opencell_`, `chassis_` 등)를 다시 이전의 축약형(`oc_`, `chas_` 등)으로 되돌리는 작업입니다. 리팩토링 중에 발생한 수치적 감도 차이를 해결하지 못했으므로, 가장 안정적이었던 상태로 물리적 수치를 완벽히 복원하는 것에 초점을 맞춥니다.

## Proposed Changes

### [Backup Restoration]

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- `whtb_config_backup_20260405.py`의 내용으로 전체 복원

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- `whtb_builder_backup_20260405.py`의 내용으로 전체 복원 (네이밍 복구)

#### [MODIFY] [run_drop_simulation_cases_v4.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v4.py)
- `run_drop_simulation_cases_v4_backup_20260405.py`의 내용으로 전체 복원

#### [MODIFY] [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py)
- `run_drop_simulation_cases_v5_backup_20260405.py`의 내용으로 전체 복원

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- `whts_engine_backup_20260405.py`의 내용으로 전체 복원

## Verification Plan

### Manual Verification
- `run_drop_simulation_cases_v4.py`를 실행하여 `test_run_case_1`이 임팩트 구간(`t=0.38s`)을 오류 없이 완료하는지 확인
