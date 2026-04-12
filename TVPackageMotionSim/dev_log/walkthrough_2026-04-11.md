# Walkthrough: Inertia Override & Auto-Interpolation

부품별 관성 모멘트(Inertia)를 수동으로 설정하고, 누락된 값(`None`)은 형상 기반으로 자동 계산하여 보간하는 기능을 성공적으로 구현하였습니다.

## 주요 구현 성과

### 1. 지능형 관계(Inertia) 보간 엔진 (`whtb_base.py`)
- **Full Tensor 계산**: 기존 3개 성분(Ixx, Iyy, Izz)에서 6개 성분(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)으로 계산 범위를 확장했습니다.
- **병합 로직**: 사용자가 입력한 리스트 내에 `None`이 포함된 경우, 해당 위치만 엔진이 계산한 실측 관성값으로 채워 넣는 하이브리드 방식을 적용했습니다.

### 2. 설정 시스템 고도화 (`whtb_config.py`)
- `components` 딕셔너리에 `inertia` 필드를 추가했습니다.
- **우선순위 제어**: `Dict` 설정과 `Flat` 설정(Legacy) 간의 동기화 시, `None`이 아닌 값이 있는 쪽을 우선시하도록 로직을 강화하여 설정 유실을 방지했습니다.

### 3. MuJoCo XML 정밀 제어
- 입력된 값의 개수에 따라 `<inertial>` 태그의 속성을 동적으로 선택합니다.
  - 3개 입력 시: `inertia="Ixx Iyy Izz"`
  - 6개 또는 `None` 포함 시: `fullinertia="Ixx Iyy Izz Ixy Ixz Iyz"`

## 검증 결과

`scratch/test_inertia_config.py`를 통해 다음 시나리오를 검증 완료했습니다:

| 테스트 시나리오 | 설정값 | 결과 (XML) | 비고 |
| :--- | :--- | :--- | :--- |
| **자동 계산** | `inertia: None` | (태그 없음) | MuJoCo가 Geoms로부터 자동 계산 (기존 방식 유지) |
| **대각 성분 오버라이드** | `[2, 2, 2]` | `inertia="2 2 2"` | 3개 성분 전용 속성 사용 확인 |
| **혼합/부분 오버라이드** | `[1, 1, 1, None]` | `fullinertia="1 1 1 0 0 0"` | `None` 부위가 엔진 계산값(0)으로 자동 보간됨 |

> [!TIP]
> 이제 복잡한 형상의 부품이라도 특정 관성 모멘트만 측정값으로 고정하고, 나머지는 시뮬레이션 모델의 격자 분포를 따르도록 유연하게 설정할 수 있습니다.

## 백업 및 관리
- **Task List**: [task_2026-04-11.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/task_2026-04-11.md)
- **Implementation Plan**: [implementation_plan_2026-04-11.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_2026-04-11.md) (백업 완료)
