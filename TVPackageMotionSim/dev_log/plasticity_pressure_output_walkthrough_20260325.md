# Walkthrough - Plasticity Pressure Output Enhancement 및 버그 수정
Date: 2026-03-25

## 1. 개요
소성 변형 정보 출력 시 압력(Pressure)을 포함하는 기능을 구현하던 중 발생한 변수 참조 에러를 해결하고, 압력 추적 및 출력 로직을 더욱 견고하게 보완하였습니다.

## 2. 주요 개선 및 수정 사항

### 2.1. 버그 수정 (UnboundLocalError 해결)
- `run_simulation` 함수 내에서 `yield_strain` 변수가 정의되기 이전에 로그 출력문에서 이를 참조하여 발생하던 에러를 수정하였습니다.
- 소성 변형 관련 파라미터(`enable_plasticity`, `yield_strain` 등)의 초기화 위치를 함수 상단부로 이동시켜 모든 로그 출력 시 안전하게 참조할 수 있도록 하였습니다.

### 2.2. 소성 변형 알고리즘 v2 (Strain 기반) 보완
- **실시간 압력 추적**: Strain 기반으로 작동할 때도 모든 완충재 블록의 접촉력을 감시하여 kPa 단위의 압력을 실시간으로 계산합니다.
- **최대 압력 기록 및 업데이트**: 단순히 활성화(Activated) 시점뿐만 아니라, 변형이 점진적으로 증가하는 전 과정 동안의 **최대 압력**을 기록하고 이를 로그에 반영합니다.
- **가독성 개선**: 로그 출력 시 압력이 측정되지 않은 경우(내부 Strain만 발생한 경우)를 명학히 구분하여 출력합니다.

### 2.3. 요약 리포트 가시성 개선
- `Calculated K & C` 리포트 출력 시 `enable_plasticity` 설정이 활성화된 경우에만 임계값(`Yield Thresholds`)을 표시하도록 하여 정보의 혼선을 방지하였습니다.

## 3. 결과 확인
- 이제 터미널 로그에서 다음과 같이 소성 변형 진행 상황과 압력을 함께 확인할 수 있습니다.
```text
[Plasticity] Strain Activated: g_bcushion_0_0_0 (Strain: 0.02, Axis: 2, Pressure: 12.8kPa)
[Plasticity] g_bcushion_0_0_0 Deforming(v2): -1.5mm (Strain: 0.01, Pressure: 15.2kPa)
```

## 4. 마치며
이번 수정을 통해 시뮬레이션의 안정성을 확보하였으며, 더욱 정밀한 물리적 변형 데이터(압력-변형 관계)를 실시간으로 모니터링할 수 있는 기반을 마련하였습니다.
