# [ run_discrete_builder 모듈 리팩토링 계획 ]

현재 `run_discrete_builder/__init__.py` 파일에 하나로 묶여 있는 1,500줄이 넘는 코드를, 객체지향 설계 원칙에 따라 여러 파일로 모듈화하여 분리하는 계획입니다. 이 작업은 코드의 유지보수성과 가독성을 획기적으로 향상시킬 것입니다.

## User Review Required

> [!warning] 외부 인터페이스 호환성 보호
> 기존 시뮬레이터(예: `run_drop_simulation_v3.py` 등)에서 `run_discrete_builder`를 호출할 때 오류가 발생하지 않도록, `__init__.py` 파일이 새롭게 분리된 파일들의 기능을 모아서 다시 밖으로 내보내는(Export) 역할을 동일하게 수행합니다. 하지만 구조 분리가 이뤄지기 때문에 작업 전 이 계획서를 상세히 검토 부탁드립니다.

## Proposed Changes

### run_discrete_builder 폴더 내부 분리

단일 구조의 `__init__.py`에서 핵심 로직을 5개의 파일로 쪼갭니다. 이를 통해 각 파일이 단일 책임을 지도록(Single Responsibility) 합니다.

#### [NEW] `config.py` (설정 및 파라미터 담당)
- `calculate_solref(K, C)`
- `get_default_config(user_config=None)` 

#### [NEW] `utils.py` (수학 및 좌표 보조 함수)
- `get_local_pose(...)`
- `parse_drop_target(...)`

#### [NEW] `core.py` (기반 시스템/뼈대 모델 뼈대 담당)
- `DiscreteBlock` 클래스: 이산형 블록의 기본 구조체
- `BaseDiscreteBody` 클래스: 메인 바디가 되는 최상위 계층 기저 클래스

#### [NEW] `models.py` (개별 파생 부품 클래스 담당)
- `core.py`의 `BaseDiscreteBody`를 상속받는 구체적인 파생 부품 모음.
- `BPaperBox`, `BCushion`, `BOpenCellCohesive`, `BOpenCell`, `BChassis`, `BAuxBoxMass`, `BUnitBlock` 구현

#### [NEW] `builder.py` (실제 파일 조립 및 출력 담당)
- `get_single_body_instance(...)`
- `create_model(...)`: 설정과 모델을 모아서 MuJoCo XML 파일을 생성하는 핵심 로직.

#### [MODIFY] `__init__.py`
방대한 로직 코드는 모두 지우고, 깔끔하게 외부 노출용 진입점(Facade 파사드 패턴)으로 구성합니다.
```python
from .config import get_default_config, calculate_solref
from .utils import parse_drop_target, get_local_pose
from .core import DiscreteBlock, BaseDiscreteBody
from .models import BPaperBox, BCushion, BOpenCellCohesive, BOpenCell, BChassis, BAuxBoxMass, BUnitBlock
from .builder import create_model, get_single_body_instance

__all__ = [
    "get_default_config", "calculate_solref",
    "parse_drop_target", "get_local_pose",
    "DiscreteBlock", "BaseDiscreteBody",
    "BPaperBox", "BCushion", "BOpenCellCohesive", "BOpenCell", "BChassis", "BAuxBoxMass", "BUnitBlock",
    "create_model", "get_single_body_instance"
]
```

## Open Questions

> [!question] 확인 사항
> 1. 분리될 신규 파일명(`config.py`, `core.py`, `models.py`, `builder.py`, `utils.py`)이 마음에 드시는지, 혹은 다른 직관적인 이름을 선호하시는지 확인 부탁드립니다.
> 2. 이번 파일 분할 작업(리팩토링)을 시작하게 되면 일시적으로 이 파일들이 작동하지 않을 수 있으나 가급적 한 번에 모든 작업을 끝내고 상호 순환 참조(Circular import)가 없도록 조율하겠습니다. 승인해 주시면 실행하겠습니다.

## Verification Plan

### Automated Tests
- 리팩토링 직후 제공되는 `test_shapes_check.xml` (테스트 파일 생성 로직)이 수정 전과 동일하게 에러 없이 생성되는지 확인.
- 분리된 파이썬 모듈 내에 Circular Import(순환 참조) 에러가 떨어지지 않는지 Python Run 체크 실시.

### Manual Verification
- 구조(폴더 및 파일)가 정상적으로 생겼는지 직접 로컬 파일에서 확인 및 뷰잉.
