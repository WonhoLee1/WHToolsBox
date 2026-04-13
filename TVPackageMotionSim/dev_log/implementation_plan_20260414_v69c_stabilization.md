# Implementation Plan - [v6.9c] Polynomial Regularization & Safe Degree Fix

다항식 피팅의 급격한 휘어짐(발산)을 억제하고, 정밀도보다는 물리적 안정성을 우선시하는 설정을 주입합니다.

## User Review Required

> [!IMPORTANT]
> - **피팅 차수 제한**: 마커 16개 이하 파트는 **2차(Quadratic)** 다항식으로 강제 제한합니다. (3차 이상의 변곡점 폭주 방지)
> - **초강력 규제화**: `reg_lambda`를 **`0.1`**로 상향하여, 노이즈에 의한 곡률 폭주를 원천 차단합니다.
> - **E-Normalization 확행**: 로그 출력 지연을 방지하기 위해 `flush=True`를 적용하고, 내부적으로 1,000 MPa 이상의 영률은 무조건 Pa 단위로 간주하여 처리합니다.

## Proposed Changes

### [Polynomial Engine]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `PlateConfig`: `reg_lambda` 기본값을 **`0.1`**로 대폭 상향.
- `analyze`: `max_safe_deg` 결정 로직을 강화하여 16개 이하 마커는 무조건 **2차**로 제한. (라인 389 부근)

### [Integrity & Logging]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `__init__`: 영률 보정 로그에 `flush=True` 주입 및 보정 로직 임계치 강화.

## Verification Plan

### Automated Tests
1. `sim_v6_integrity_v69c_test.txt` 로그에서 `[2x2]` 또는 `[2x1]` 피팅 차수가 적용되는지 확인.
2. `Opencell_Front`의 응력이 **클리핑 상한선(10,000) 미만**으로 내려오는지 확인.

### Manual Verification
1. ParaView에서 면이 찢어지거나 진동하는 현상 없이 부드러운 곡면이 나오는지 확인.
