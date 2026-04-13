# Implementation Plan: Resolving SVD Convergence & Export Robustness

낙하 시뮬레이션 파이프라인 v6의 자율 분석 과정에서 발생하는 SVD(Singular Value Decomposition) 수렴 오류와 이로 인한 통합 데이터 내보내기(Export) 중단 문제를 해결하기 위한 계획입니다.

## User Review Required

> [!IMPORTANT]
> **SVD 수렴 오류 및 피팅 정밀도 저하**
> 현재 `Chassis_Front`와 같이 마커가 박판 상에 배열된 경우, 자율 좌표계 산출 과정에서 수치적 불안정성으로 인해 SVD가 수렴하지 않거나 ("linalgerror_svd_nonconverg"), 실제 변위보다 수십 배 큰 가상의 변위(Fit > Markers)가 산출되는 현상이 보고되었습니다. 이를 방지하기 위해 정규화 및 예외 처리를 도입합니다.

## Proposed Changes

### 1. Issue Tracking & Logging

#### [MODIFY] [issue_tracker.md](file:///c:/Users/GOODMAN/WHToolsBox/issue_tracker.md)
- [P5] 항목 추가: SVD 수렴 오류 및 `KeyError` 이슈 기록.

---

### 2. Multi-PostProcessor Engine Robustness

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- **`remove_rigid_motion` 메소드 개선**:
    - SVD 연산(`np.linalg.svd`) 전 공분산 행렬 `H`에 아주 작은 Epsilon(1e-12)을 더해 수치적 안정성을 확보합니다.
    - `try-except` 블록을 추가하여 SVD 실패 시 에러가 아닌 단위 행렬(`np.eye(3)`)을 반환하도록 하여 분석이 중단되지 않게 합니다.
- **`analyze` 메소드 개선**:
    - "Fit > Markers" 경고가 발생할 경우, 해당 프레임의 결과가 비정상적임을 로그에 남기고 분석 결과가 오염되지 않도록 가드를 추가할 수 있습니다.

---

### 3. Exporter Persistence

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- **`export_to_vtkhdf` 메소드 가드 추가**:
    - 각 부품(Analyzer)을 순회하며 `analyzer.results`에 접근하기 전, 필수 키(`Displacement [mm]`)가 존재하는지 확인합니다.
    - 분석에 실패한 부품은 건너뛰고 나머지 정상 부품들만이라도 내보내기를 완료하도록 수정합니다.

---

### 4. Simulation Configuration Adjustments

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- **자율 분석 모드 보완**:
    - (선택 사항) 특정 부품에서 계속 오류가 발생할 경우, `W`, `H` 정보를 명시적으로 제공할 수 있는 구조를 열어줍니다.

## Open Questions

- `Chassis_Front`에서 유독 오류가 발생하는 이유가 마커의 수 부족 때문인가요, 아니면 배치가 너무 일직선(Colinear)이기 때문인가요? (현재 코드에서 마커 좌표를 덤프하여 확인 가능)

## Verification Plan

### Automated Tests
- `run_command`를 통해 `TVPackageMotionSim/run_drop_simulation_cases_v6.py` 실행.
- 터미널 로그에서 `❌ Critical Error`가 발생하더라도 Export 프로세스가 끝까지 진행되는지 확인.
- `KeyError` 발생 여부 확인.

### Manual Verification
- ParaView에서 "Fit > Markers" 경고가 떴던 부품들이 어떻게 렌더링되는지 시각적 확인.
