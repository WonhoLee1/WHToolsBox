# [PLAN] XML 클래스 네이밍 현대화 (contact_ 접두사 제거) - 2026-04-13

안녕하세요, **WHTOOLS**입니다. 

사용자님께서 제공해주신 최신 시뮬레이션 모델 XML(`rds-20260412_021836`) 분석 결과, 현재 시스템은 더 이상 `contact_` 접두사를 사용하지 않고 부품명을 클래스 이름으로 직접 사용하는 방향으로 진화했음을 확인했습니다. 

이에 따라 `whtb_base.py`의 구식 네이밍 로직을 수정하여 시스템 전체의 일관성을 확보하겠습니다.

## 🛠️ 제안된 변경 사항

### 1. [Component] Base Discrete Body (`whtb_base.py`)
- **[MODIFY] `get_worldbody_xml_strings` 메서드 수정**:
    - **변경 전**: `geom_class = f"contact_{self.__class__.__name__.lower()}"`
    - **변경 후**: `geom_class = self.__class__.__name__.lower()`
    - **대상**: `use_internal_weld`가 `True`인 경우와 `False`인 경우 모두 적용
    - **효과**: 생성되는 XML 지오메트리 태그가 `<geom class="bcushion" .../>`과 같이 깔끔해지며, `whtb_builder.py`에서 정의한 `default class`와 완벽히 일치하게 됩니다.

## ✅ 검증 계획

### 1. 시뮬레이션 재구동
- `run_drop_simulation_cases_v6.py`를 실행하여 더 이상 `ValueError: unknown default class name` 오류가 발생하지 않는지 확인합니다.

### 2. 생성된 XML 검증
- 생성된 `temp_drop_sim.xml` (또는 유사 파일)을 열어 `geom`의 `class` 속성이 `contact_` 없이 올바르게 출력되는지 육안으로 최종 점검합니다.
