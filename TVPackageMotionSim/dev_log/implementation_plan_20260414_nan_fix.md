# Implementation Plan - [v6.1] NaN Stability & Pipeline Integrity

유연 파트(`Opencell` 등) 해석 시 발생하는 `nan`(Not a Number) 전파 문제를 해결하고, 시뮬레이션 파이프라인의 강건성(Robustness)을 확보합니다.

## User Review Required

> [!IMPORTANT]
> - 해석 결과에서 `nan`이 발견될 경우 `0.0`으로 자동 치환됩니다. 이는 시각화 크래시를 방지하기 위한 조치입니다.
> - SVD 실패 프레임이 많을 경우 결과의 신뢰도가 낮아질 수 있음을 알리는 경고 메시지가 강화됩니다.

## Proposed Changes

### [Component] Post-Processor Engine (`whts_multipostprocessor_engine.py`)

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- **Weight Normalization**: `np.sum(self.weights)`가 0에 가까울 경우 균등 가중치로 폴백하는 가드 추가.
- **NaN to Num**: JAX 연산 전 `all_displacement_w_rel`의 `nan`을 `0.0`으로 치환.
- **Reporting**: 결과 요약 출력 시 `np.nanmean`을 사용하여 일부 프레임 오류 시에도 통계 출력 보장.

### [Component] Exporter (`whts_exporter.py`)

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- **Data Sanitization**: `export_to_vtkhdf` 메서드 내에서 모든 데이터 배열에 `np.nan_to_num` 적용하여 ParaView 호환성 확보.

## Open Questions
- `Opencell` 파트의 마커가 유실되어 `nan`이 발생하는 경우, 해당 파트의 해석을 스킵하시겠습니까, 아니면 0으로 채워서 내보내시겠습니까? (현재는 0으로 채우는 방향으로 제안합니다.)

## Verification Plan

### Automated Tests
- `nan`이 포함된 더미 데이터를 생성하여 Exporter가 에러 없이 완주하는지 확인.

### Manual Verification
- `Opencell`이 포함된 기존 시뮬레이션 케이스 재실행 후 터미널 Crash 여부 및 대시보드 정상 진입 확인.
