# Walkthrough - VTKHDF Compatibility & Exporter Stability Fix

ParaView 6.0에서 발생하던 VTKHDF 시계열 데이터(Transient Data) 로드 오류를 해결하고, 내보내기 과정에서의 안정성을 대폭 강화했습니다.

## 🛠️ 주요 수정 사항

### 1. ParaView 6.0 규격 준수 (VTKHDF Compliance)
- **명칭 표준화**: `Steps` 그룹 내의 `ConnectivityOffsets` 데이터셋 명칭을 최신 규격인 `ConnectivityIdOffsets`로 변경했습니다. 이 수정으로 ParaView가 시계열 위상 정보를 정상적으로 인식합니다.
- **PartOffsets 정밀화**: `PartID` 데이터셋의 시간축 맵핑을 위해 `PartOffsets`를 프레임별 `total_points` 누적으로 정확하게 설정했습니다.

### 2. 데이터 스트리밍 루프 안정화 (Robustness)
- **KeyError/IndexError 방어**: 분석이 실패하거나 프레임 수가 일치하지 않는 파트가 있어도 전체 내보내기 프로세스가 중단되지 않도록 `try-except` 블록과 `results.get()` 로직을 적용했습니다.
- **상세 에러 리포팅**: 프레임별 처리 중 오류 발생 시 `traceback`을 출력하여 문제 진단이 용이하도록 개선했습니다.

### 3. ParaView 대시보드 스크립트 최적화
- **API 대응**: ParaView 버전에 따라 `DescriptiveStatistics` 필터의 속성명이 `Variables` 또는 `ModelVariables`로 바뀌는 것에 대응하는 동적 체크 로직을 강화했습니다.

## 🧪 검증 결과

### 1. 통합 파이프라인 테스트 (`v6.0`)
- `run_drop_simulation_cases_v6.py` 실행 결과, 모든 파트의 분석 후 VTKHDF 및 GLB 내보내기가 오류 없이 완료됨을 확인했습니다.
- **로그 요약**:
  ```text
  [WHTOOLS] Exporting COMPLIANT VTKHDF (Transient) to: .../Result.vtkhdf
    > Streaming Transient Data...
  📦 [WHTOOLS] EXPORT COMPLETE
  🚀 [WHTOOLS] Launching ParaView Dashboard: Result.vtkhdf
  ```

### 2. ParaView 로드 확인
- 생성된 `.vtkhdf` 파일이 ParaView에서 더 이상 `ConnectivityIdOffsets` 관련 오류를 발생시키지 않으며, 시계열 애니메이션이 정상 작동함을 확인했습니다.

## 📌 참고 사항
> [!IMPORTANT]
> - 대규모 모델 해석 시 VTKHDF 파일 용량이 커질 수 있으므로, 결과 폴더의 디스크 공간을 사전에 확인하시기 바랍니다.
> - `v6` 파이프라인은 이제 최소한의 마커 정보만으로도 내보내기까지 완벽하게 자율적으로 수행합니다.
