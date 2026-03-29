# PBA 및 구조 지표 고도화 (v4.5) Implementation Plan

## 개요
PBA(Principal Bending Axis)는 부품의 거동을 대표하는 고유한 회전축입니다. 기존의 2D(XY 평면) 제한적 공분산 분석을 3D로 확장하여, 부품의 물리적 배향과 관계없이 가장 지배적인 변형 축을 정밀하게 탐색합니다.

## Proposed Changes

### [Structural Analysis Engine]
#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)
- **3D PBA 연산**: `rot_vec`의 3개 성분(X, Y, Z)을 모두 사용하여 $3 \times 3$ 공분산 행렬을 구축하고 고유값 분해(EVD)를 수행합니다.
- **주축 물리량 추출**:
  - 최대 고유값의 제곱근을 PBA Magnitude로 정의.
  - 해당 고유벡터를 PBA Vector(3D)로 저장.
  - 방위각(Azimuth) 및 고도각(Elevation) 산출.
- **Bending Stress 정밀화**: PBA 방향 성분과 Twist(법선 방향) 성분을 물리적으로 엄밀히 분리하여 스트레스 계산에 반영.

### [Post-Processing UI]
#### [MODIFY] [postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/postprocess_ui.py)
- **데이터 리포트 업데이트**: 3D PBA 정보를 요약 테이블 및 상세 인포창에 반영.
- **시계열 그래프**: PBA의 3차원 방향 변화(방위각/고도각)를 모니터링할 수 있는 옵션 검토.

## Verification Plan
### Automated Tests
- 특정 축(예: [1, 1, 0] 방향)으로 강제 변형된 더미 데이터를 생성하여 EVD 결과가 해당 축을 정확히 찾아내는지 검증하는 스크립트 실행.
### Manual Verification
- 시뮬레이션 후 Post-UI 요약 테이블에서 PBA Peak 시점의 각도(Angle)와 벡터(Vector) 값이 물리적 상식(낙하 방향 및 충격 지점)과 부합하는지 확인.
