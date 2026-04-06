# [WHTOOLS] Config Refactor & Parameter Standardization Walkthrough

`get_default_config()`를 `test_run_case_1` 기반으로 최적화하고, 프로젝트 전반의 파라미터 네이밍을 표준화하여 유지보수성과 가독성을 개선했습니다.

## 주요 변경 사항

### 1. `get_default_config()` 구조 혁신 및 기본값 상향
- **골든 스탠다드 적용**: `test_run_case_1`에서 사용되던 정밀 물리/기하 파라미터를 시스템 기본값으로 상향했습니다.
- **카테고리별 구조화**: 설정값을 Geometry, Physics, Mass, Air, PostProcess 등 논리적 그룹으로 분류하여 가독성을 높였습니다.
- **기본값 통합**: 솔버 및 엔진 내부에서 `.get()`으로 처리되던 숨겨진 기본값들을 모두 `get_default_config`로 명시적으로 통합했습니다.

### 2. 파라미터 네이밍 표준화 (Prefix/Suffix 확장)
사용자의 요청에 따라 모호한 축약 코드를 명확한 엔지니어링 용어로 확장했습니다.
- `oc_` / `_oc` → `opencell_` / `_opencell`
- `occ_` / `_occ` → `opencellcoh_` / `_opencellcoh`
- `chas_d` → `chassis_d`

### 3. 전체 프로젝트 동기화
변경된 네이밍 표준을 다음 파일들에 일괄 적용했습니다.
- `run_discrete_builder/whtb_config.py` (핵심 엔진)
- `run_discrete_builder/whtb_builder.py` (모델 빌더)
- `run_drop_simulation_cases_v4.py` (시나리오 V4)
- `run_drop_simulation_cases_v5.py` (시나리오 V5)

### 4. 하위 호환성 및 안전 장치
- **Legacy Mapping**: 기존 테스트 코드에서 여전히 `oc_`, `occ_` 등을 사용할 경우를 대비하여, `get_default_config` 내부에서 이를 최신 네이밍으로 자동 전환하는 로직을 추가했습니다.
- **백업 생성**: 작업 전 주요 파일들을 `_backup_20260405.py` 형태로 백업하여 안전성을 확보했습니다.

## 검증 결과

### 1. 설정값 검증 스크립트 실행 (`verify_refactor.py`)
새로운 네이밍 표준과 하위 호환성 매핑이 올바르게 동작하는지 확인하는 검증 스크립트를 작성하여 테스트를 완료했습니다.

> [!check]
> **검증 요약**:
> - [x] `opencell_div` 기본값: [5, 5, 1] (PASS)
> - [x] `mass_opencellcoh` 기본값: 0.1 (PASS)
> - [x] `ssr_resolution` 통합 확인: 40 (PASS)
> - [x] 하위 호환성 매핑 (`oc_` -> `opencell_`): 작동 확인 (PASS)

---
**WHTOOLS**는 이번 리팩토링을 통해 더욱 견고하고 표준화된 시뮬레이션 환경을 구축했습니다. 이제 모든 시나리오에서 일관된 물리 파라미터를 기반으로 정밀한 해석이 가능합니다.
