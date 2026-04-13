# Implementation Plan - [v7.1] Physical Curvature Scaling & Final Integrity

JAX 가속 엔진(`evaluate_batch`) 내부에 누락되었던 물리적 좌표계 스케일 인자를 주입하여, 비물리적으로 부풀려진 응력 수치를 정상화합니다.

## User Review Required

> [!IMPORTANT]
> - **곡률 물리적 변환**: 정규화 공간(`[0, 1]`)에서 계산된 곡률을 실제 물리 공간(`mm`)으로 변환하기 위해 `x_rng^2` 및 `y_rng^2`로 나누어주는 로직을 주입합니다.
> - **응력 정상화**: 이 조치로 인해 별도의 클리핑 없이도 모든 응력이 항복 강도 미만의 정상 범위로 돌아옵니다.

## Proposed Changes

### [Engine Core Physics]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `evaluate_batch`: `kxx`, `kyy`, `kxy` 계산 시 각각 `x_rng**2`, `y_rng**2`, `x_rng * y_rng`로 나누어주는 스케일링 로직 추가.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 로그 상의 응력이 **100 MPa 미만** (예상치 1~30 MPa)으로 안착되는지 확인.
2. `Opencell_Left/Right` 등 모든 파트에서 물리적으로 타당한 수치가 나오는지 확인.

### Manual Verification
1. ParaView 시각화에서 응력이 지나치게 붉게(폭주) 표시되지 않고, 변형 부위에만 상식적으로 분포하는지 확인.
