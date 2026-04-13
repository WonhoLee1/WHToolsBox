# Implementation Plan - [v6.9d] Unified ParaView Specification & Numerical Integrity

ParaView의 VTKHDF 1.0 시계열 데이터 규격 및 최신 API(6.0.x 등)와의 호환성을 완벽히 확보하고, 해석 수치가 물리적 한계를 벗어나지 않도록 엔진을 보강합니다.

## User Review Required

> [!IMPORTANT]
> - **VTKHDF 규격 강화**: `Steps/PartOffsets` 데이터셋을 추가하여 ParaView 5.10+ 버전의 읽기 오류를 해결합니다.
> - **매크로 API 수정**: `DescriptiveStatistics` 필터의 속성을 `Variables`에서 **`ModelVariables`**로 수정합니다.
> - **곡률 직접 제어**: 응력 폭주의 근원인 곡률($\kappa$)을 JAX 엔진 수준에서 물리적 한계치로 클리핑합니다.

## Proposed Changes

### [ParaView Compatibility]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `export_to_vtkhdf`: `Steps/PartOffsets` 데이터셋(zeros) 추가 및 데이터 타입 정밀화.
- `launch_paraview`: 대시보드 매크로 내 `DescriptiveStatistics.ModelVariables` 적용.

### [Numerical Deep Integrity]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateMechanicsSolver.__init__`: 영률 E가 $1.0e6$ 초과 시 즉시 MPa로 보정하는 하드 필터 주입.
- `PlateMechanicsSolver.evaluate_batch`: 곡률 `kxx, kyy, kxy`에 대해 `jnp.clip` 적용.
- `PlateConfig`: `reg_lambda`를 **`1.0`**으로 상향하여 극한의 안정성 확보.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 `Result.vtkhdf` 파일 내부에 `Steps/PartOffsets`가 존재하는지 확인.
2. 모든 부품의 수치 로그 상에서 응력이 **200 MPa 미만**인지 확인.

### Manual Verification
1. ParaView 6.0.1+에서 대시보드가 에러 없이 자동으로 열리며, 타임 스텝 간 이동이 자유로운지 확인.
