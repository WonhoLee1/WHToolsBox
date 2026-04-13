# Implementation Plan - [v6.9d] Super-Conductive Integrity & Direct Curvature Guard

수학적 한계를 넘어선 수치 폭주를 원천 차단하기 위해, 곡률(Curvature) 직접 제어와 하드웨어 수준의 단위계 강제 정규화를 시행합니다.

## User Review Required

> [!IMPORTANT]
> - **곡률 직접 제어(Curvature Guard)**: JAX 엔진 내에서 곡률 $\kappa$가 $1.0$ (곡률 반경 1mm 수준)을 초과할 경우 물리적으로 불가능한 변형으로 간주하여 클리핑합니다.
> - **강제 MPa 고정**: `PlateMechanicsSolver` 내부에서 영률 E를 무조건 **10,000 ~ 70,000 MPa** 사이로 강제 필터링하여 응력 뻥튀기를 원천 봉쇄합니다.
> - **ParaView 무결성**: `HDFReader` 오류를 **`VTKHDFReader`**로 직접 치환하여 대시보드 구성을 완결합니다.

## Proposed Changes

### [Numerical Guard Strategy]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateMechanicsSolver.__init__`: 입력받은 영률이 $1.0e6$을 넘을 경우 즉시 MPa로 환산하는 로직을 최우선 실행.
- `PlateMechanicsSolver.evaluate_batch`: 곡률 `kxx, kyy, kxy`에 대해 `jnp.clip`을 적용하여 응력 발산의 근원 차단.
- `PlateConfig`: `reg_lambda`를 **`1.0`** (더 강력한 억제)으로 최종 상향.

### [ParaView Stability]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `launch_paraview`: `OpenDataFile` 코드 보강 및 `VTKHDFReader` 명시적 사용 검토.

## Verification Plan

### Automated Tests
1. `sim_v6_final_integrity_v69d.txt` 로그에서 모든 부품의 최대 응력이 **200 MPa 미만**으로 정교하게 안착되는지 확인.
2. `Opencell_Front` 등에서 나타나던 발산 경고가 사라지는지 확인.

### Manual Verification
1. ParaView에서 대시보드가 정상적으로 나타나며 전 부품의 응력 분포가 0~100 MPa 수준의 현실적 색상 분포를 보이는지 확인.
