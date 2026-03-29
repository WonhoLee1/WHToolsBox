# [Refactoring] run_discrete_builder 모듈화 완료

기존에 약 1,500줄에 달하던 `run_discrete_builder/__init__.py` 파일을 기능별로 분리하고 `whtb_` 접두사를 적용하여 구조를 개선했습니다.

## 주요 변경 사항

### 1. 모듈화 (Modularization)

단일 파일 구조를 다음과 같이 5개의 독립된 모듈로 분리했습니다.

- **[whtb_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_utils.py)**: 수학 연산 및 좌표 파싱 로직 (`get_local_pose`, `calculate_solref`, `parse_drop_target`)
- **[whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)**: 기본 파라미터 및 재질 설정 (`get_default_config`)
- **[whtb_base.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_base.py)**: 이산형 모델링을 위한 핵심 기반 클래스 (`DiscreteBlock`, `BaseDiscreteBody`)
- **[whtb_models.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_models.py)**: 개별 부품(Cushion, Chassis 등) 클래스 정의
- **[whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)**: 전체 조립 및 MuJoCo XML 생성 오케스트레이션 (`create_model`, `get_single_body_instance`)

### 2. 하위 호환성 유지 (Backward Compatibility)

기존 시뮬레이션 코드에서 이 패키지를 사용하는 방식은 그대로 유지됩니다. `__init__.py`에서 새로운 모듈들을 임포트하여 API를 통합 노출하도록 설계했습니다.

```python
# 기존 방식 그대로 사용 가능
from TVPackageMotionSim.run_discrete_builder import create_model
```

---

## 검증 결과

- **모듈 실행 테스트**: `python -m TVPackageMotionSim.run_discrete_builder.whtb_builder` 명령을 통해 정상적인 XML 생성 및 관성 보고서(Inertia Report) 출력을 확인했습니다.
- **순환 참조 확인**: 각 모듈 간의 임포트 구조가 깔끔하게 정리되어 오류 없이 작동함을 확인했습니다.

---

> [!TIP]
> 이제 각 부품의 물리 로직을 수정하거나 설정을 추가할 때, 전체 코드를 뒤질 필요 없이 해당 `whtb_*` 파일만 수정하면 됩니다. 유지보수 공수가 크게 줄어들 것으로 예상됩니다.
