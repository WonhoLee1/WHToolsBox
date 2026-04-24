# [PLAN] 마차 계수 정규화 함수(get_friction_standard) 복구 - 2026-04-13

안녕하세요, **WHTOOLS**입니다. 

시뮬레이션 구동의 발목을 잡고 있는 `ImportError`를 해결하기 위해, `whtb_config.py`에서 유실된 `get_friction_standard` 함수를 재구현하고 시스템을 정상화하겠습니다.

## 🛠️ 제안된 변경 사항

### 1. [Component] Configuration Library (`whtb_config.py`)
- **[NEW] `get_friction_standard` 함수 추가**:
    - **목적**: 입력된 마찰 계수(단일 값 또는 리스트)를 MuJoCo 표준 5차원 배열로 변환
    - **입력**: `mu` (Union[float, list, tuple]), `dim` (int, default=5)
    - **로직**:
        1. 입력값이 스칼라인지 시퀀스인지 확인 (이슈 #001 대응)
        2. Tangential 마찰 계수가 1개만 제공된 경우 복사하여 2개로 확장
        3. Torsional(0.005) 및 Rolling(0.0001) 기본값을 적용하여 지정된 차원까지 패딩

```python
def get_friction_standard(mu, dim=5):
    # 입력값 타입 가드 및 리스트화
    if isinstance(mu, (list, tuple)):
        result = list(mu)
    else:
        result = [float(mu)]
    
    # 부족한 차원 보완 (Tangential 2개, Torsional 1개, Rolling 2개 기준)
    defaults = [0.0, 0.0, 0.005, 0.0001, 0.0001]
    
    if len(result) == 1:
        result.append(result[0]) # Tangential 1 -> 2 확장
        
    while len(result) < dim:
        result.append(defaults[min(len(result), len(defaults)-1)])
        
    return result[:dim]
```

## ✅ 검증 계획

### 1. 단위 테스트
- `whtb_config.py`를 직접 실행하거나 간단한 스크립트를 통해 다음 케이스 검증:
    - `get_friction_standard(0.3)` -> `[0.3, 0.3, 0.005, 0.0001, 0.0001]`
    - `get_friction_standard([0.3, 0.4])` -> `[0.3, 0.4, 0.005, 0.0001, 0.0001]`

### 2. 통합 테스트
- `run_drop_simulation_cases_v6.py`를 다시 실행하여 `ImportError`가 사라지고 시뮬레이션이 시작되는지 확인

## 📋 오픈 질문

> [!QUESTION]
> Torsional 및 Rolling 마찰 계수의 기본값(0.005, 0.0001)은 기존 시스템에서 사용하던 표준값입니다. 현재 특정 테스트 케이스에서 다른 기본값이 필요한 상황인가요?
