# VTKHDF Transient 포맷 정합성 수정 및 내보내기 안정화 계획

현재 모델이 생성하는 `.vtkhdf` 파일이 ParaView 6.0에서 `Steps/ConnectivityIdOffsets` 데이터셋을 찾지 못해 발생한 호환성 문제를 해결하고, 시뮬레이션 결과 내보내기 과정에서의 예기치 않은 종료(Exit Code 1)를 방지하기 위한 안정화 작업을 수행합니다.

## User Review Required

> [!IMPORTANT]
> - **ParaView 버전 호환성**: 본 수정은 VTKHDF 1.0 규격 및 ParaView 5.10~6.0 이상의 최신 API 규격에 맞추어 데이터셋 명칭을 조정합니다.
> - **메모리 점유**: Transient 데이터 세트를 저장할 때 Connectivity와 Types를 타일링(Tiling)하므로 대규모 모델에서 메모리 사용량이 일시적으로 증가할 수 있습니다.

## Proposed Changes

### [WHTOOLS Post-Processor Exporter]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)

1. **VTKHDF Steps 데이터셋 명칭 수정**:
   - `ConnectivityOffsets`를 `ConnectivityIdOffsets`로 변경하여 ParaView 규격을 준수합니다.
2. **PartOffsets 계산 로직 보강**:
   - `PointData` 그룹의 `PartID` 데이터셋이 시계열(Transient)로 저장됨에 따라, 각 스텝별 오프셋을 정확히 가리키도록 `PartOffsets`를 `total_points` 배수로 설정합니다.
3. **데이터 스트리밍 루프 안정화**:
   - `az.results` 내에 필요한 키가 없는 경우에 대비한 방어 로직을 추가합니다.
   - 대량의 데이터 처리를 시각화하기 위해 로그 출력을 개선하고, 예외 발생 시 상세 원인을 출력하도록 합니다.
4. **ParaView 대시보드 스크립트(Dashboard Script) 개선**:
   - ParaView 6.0 API에서 변경된 `ModelVariables` 대응 로직을 확실하게 보정합니다.

## Open Questions

- **ParaView 설치 경로**: 현재 `C:\Program Files\ParaView 6.0.1\bin\paraview.exe`를 우선 탐색하도록 되어 있습니다. 혹시 다른 경로를 사용 중이시라면 말씀해 주세요.

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v6.py`를 실행하여 새로운 `.vtkhdf` 파일이 생성되는지 확인합니다.
- `h5py`를 사용하여 생성된 파일의 `Steps` 하위에 `ConnectivityIdOffsets`가 존재하는지 스크립트로 검증합니다.

### Manual Verification
- 생성된 파일을 ParaView 6.x 버전에서 직접 열어 `Information` 탭에 시계열 데이터(Transient data)가 정상적으로 인식되는지 확인합니다.
- WarpByVector 필터가 정상 작동하는지 확인합니다.
