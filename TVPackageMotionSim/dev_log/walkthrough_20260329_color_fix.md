# Walkthrough - Cushion Localization & Color Fix

쿠션의 시각적 로컬라이징 문제를 해결하고, 유저님의 요청에 따라 **8개의 꼭짓점 및 Depth(Z) 방향 모서리**에만 특화된 시각적 피드백 시스템을 구축했습니다.

## 변경 사항 및 주요 피처

### 1. [Builder] 정교해진 모서리 판정 로직
- `whtb_models.py`에 `is_corner_block` 메서드를 추가하여 (ix, iy)가 끝단인 블록(Z방향 기둥)을 정확히 식별합니다.
- `whtb_base.py`에서 XML 생성 시, 이 로직을 사용하여 해당 블록들의 지오메트리 이름에 `_edge` 접미사를 추가하고 `contact_bcushion_edge` 클래스를 부여하도록 개선했습니다.

### 2. [Engine] 국소적 소성 추적 및 시각적 강조
- `whts_engine.py`의 `_init_plasticity_tracker`에서 이제 모든 쿠션이 아닌, 이름에 `_edge`가 포함된(즉, Z방향 모서리인) 지오메트리만 소성 변형 추적 대상으로 등록합니다.
- 시뮬레이션 시작 시, 이 타겟 블록들의 색상을 **노란색(`[1.0, 1.0, 0.0, 1.0]`)**으로 자동 변경하여 v3와 동일한 시각적 가이드라인을 제공합니다.

### 3. [Reporting] 히트맵 정합성
- `whts_reporting.py`의 히트맵 로직이 이제 각 바디(블록)의 변형 정도에 따라 색상을 입히되, 추적 대상에서 제외된 일반 쿠션 블록들은 소성 변형 연산이 수행되지 않으므로 초기 회색 상태를 유지하거나 변형도 0의 색상을 가지게 됩니다.

## 시각적 검증 예시

시뮬레이션을 새로 실행하면 아래와 같은 변화를 보실 수 있습니다:
- **시작 시점**: 쿠션의 4개 세로 모서리 기둥(Depth-wise)만 선명한 노란색으로 보이고, 나머지 면과 내부는 반투명 회색으로 표시됩니다.
- **충격 시점**: 바닥이나 다른 물체에 부딪힐 때, 오직 노란색 모서리 블록들만 법선력에 따라 소성 수축(Size reduction)이 발생하며 시각적으로 변형이 인지됩니다.

## 파일별 수정 내역 요약

| 파일명 | 주요 수정 내용 |
| :--- | :--- |
| [whtb_models.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_models.py) | `is_corner_block`(Depth-wise) 로직 추가 |
| [whtb_base.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_base.py) | XML 생성 시 `_edge` 접미사 및 클래스 조건부 할당 |
| [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py) | 소성 추적기 필터링 및 초기 노란색 강조 구현 |

이제 `python run_drop_simulation_v4.py`를 실행하여 개선된 시각화 결과를 확인해 보시기 바랍니다.
