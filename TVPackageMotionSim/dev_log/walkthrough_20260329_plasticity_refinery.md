# Walkthrough - Advanced Plasticity and Real-Time Visualization

이번 업데이트를 통해 소성 변형(Plasticity) 알고리즘을 물리적으로 정교화하고, 시뮬레이션 과정에서 변형 상태를 시각적으로 즉각 확인할 수 있는 기능을 구현했습니다.

## 주요 변경 사항

### 1. 방향성 소성 변형 (Directional Plasticity)
기존에는 부품의 가장 긴 축만 줄어들었으나, 이제는 **충돌 법선 벡터(Contact Normal)**를 분석합니다.
- **로컬 축 탐지**: 접촉이 발생한 시점의 법선 벡터를 지오메트리의 로컬 좌표계로 변환하여, 실제로 힘을 받는 축(X, Y, Z 중 하나)을 찾아냅니다.
- **정밀 수축**: 탐지된 축의 `geom_size`만 선별적으로 감소시켜, 낙하 방향에 따른 실제 찌그러짐 현상을 물리적으로 모사합니다.

### 2. 실시간 색상 전이 (Yellow -> Blue)
시뮬레이션 루프가 돌아가는 동안 변형률을 계산하여 색상을 즉시 업데이트합니다.
- **색상 보간**: 초기 상태(노란색, `[1, 1, 0]`)에서 변형이 진행될수록 파란색(`[0, 0, 1]`)으로 서서히 변합니다.
- **시각적 피드백**: 시뮬레이션이 끝난 후가 아니라, **낙하 충격이 발생하는 실시간 과정**에서 색상이 변하는 것을 Viewer를 통해 확인할 수 있습니다.

## 작업 내용 요약
- [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py): `_apply_plasticity_v2` 메서드 전면 개편.
- `geom_xmat` 접근 버그 수정 및 최적화.
- 소성 변형 감지 임계값 및 수축 비율 동기화.

## 실행 및 확인 방법
1. `python run_drop_simulation_cases_v4.py`를 실행합니다.
2. MuJoCo Viewer에서 낙하 시 모서리 블록들이 압축 방향으로 얇아지며 **노란색에서 파란색**으로 변하는지 확인합니다.
