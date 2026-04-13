# Implementation Plan - [v6.9] Stress Normalization & ParaView Stability

비공학적인 응력 수치(수천만 MPa)를 정상화하고, ParaView가 이 데이터를 읽을 때 크래시가 발생하지 않도록 익스포트 파이프라인을 보강합니다.

## User Review Required

> [!IMPORTANT]
> - **단위계 변경**: 영률(E)을 Pa 단위에서 MPa($N/mm^2$) 단위로 내부 변환합니다.
> - **응력 상한선**: 시각화 안정성을 위해 10,000 MPa 이상의 값은 클리핑 처리합니다. (실제 구조물은 그 전에 파손되므로 시각화에 지장 없음)

## Proposed Changes

### [Structural Analysis Engine]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `ShellDeformationAnalyzer.__init__`: 입력받은 영률 `E`가 $10^6$ 이상일 경우 Pa로 간주하여 MPa로 자동 변환하는 로직 추가.
- `PlateConfig`: 기본 재질 상수를 공학적으로 타당한 값(예: PP 1,500MPa, EPS 50MPa 등)으로 클래스별 차등 적용 검토.

### [Data Export & ParaView]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- `export_to_vtkhdf`: 저장 직전 데이터에 `np.nan_to_num` 및 `np.clip`을 적용하여 ParaView 렌더링 엔진 보호.
- Path Handling: 모든 경로에 대해 `os.path.normpath` 및 슬래시(/) 변환 재확인.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 로그 상의 응력 수치가 **100 MPa 미만**으로 나오는지 확인.

### Manual Verification
1. 생성된 `vtkhdf` 파일을 ParaView 6.x에서 열어 대시보드가 정상적으로 출력되는지 확인 (크래시 여부 중점).
