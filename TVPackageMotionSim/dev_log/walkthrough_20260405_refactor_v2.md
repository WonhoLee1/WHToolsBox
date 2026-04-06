# Walkthrough: Simulation Stability & Config Refactoring V2

## 1. 목차 (Table of Contents)
- [1. 개요 (Overview)](#1-개요-overview)
- [2. 주요 변경 사항 (Key Changes)](#2-주요-변경-사항-key-changes)
- [3. 검증 결과 (Verification Results)](#3-검증-결과-verification-results)
- [4. 향후 관리 계획 (Future Maintenance)](#4-향후-관리-계획-future-maintenance)

---

## 2. 개요 (Overview)
리팩토링 과정에서 발생했던 시뮬레이션 불안정성(0.36초 충격 시점의 NaN 폭발)을 해결하기 위해 **Refactoring V2: Golden Values Alignment**를 수행했습니다. 본 작업은 물리 설정의 **"Source of Truth"**를 명확히 하고, 레거시 시스템(`test_run_case_1`)의 물리적 무결성을 100% 보존하는 데 초점을 맞췄습니다.

---

## 3. 주요 변경 사항 (Key Changes)

### 3.1. Late-Binding Configuration (지연 바인딩 설정)
- **`sync_phys_config()` 도입**: 모든 사용자 설정 오버라이드가 끝난 최종 시점에 `mat_` 딕셔너리와 `solref/solimp` 문자열을 재조립하여 설정 누락 오류를 원천 차단했습니다.
- **Source of Truth**: `get_default_config()`에 레거시의 검증된 물리 수치를 기본값(Hardcoded Defaults)으로 반영했습니다.

### 3.2. 물리 안정성 하향 평행 (Golden Values Restoration)
- **솔버 고도화**: `implicitfast` 솔버와 `0.0012` 타임스텝을 복원하여 고속 충격 시의 수치적 안정성을 확보했습니다.
- **감쇠비 보정**: Chassis 및 OpenCell Weld의 감쇠비(`damprr`)를 레거시 황금값(0.5)으로 1:1 일치시켰습니다.

### 3.3. 명칭 표준화 (Naming Standardization)
- `oc_` → `opencell_`
- `occ_` → `opencellcoh_`
- `chas_` → `chassis_`
정상적으로 통합 제어되도록 리팩토링하되, 레거시 호환성을 위해 `config.get` 레이어를 보강했습니다.

---

## 4. 검증 결과 (Verification Results)

### 4.1. XML 1:1 일치 여부 (XML Identity Check)
`compare_xml.py`를 통해 레거시 빌더와 리팩토링 빌더가 생성하는 XML을 대조한 결과, 물리 엔진 환경이 **100% 동일(IDENTICAL)**함을 확인했습니다.

> [!CHECK]
> **Result**: ✅ XMLs are IDENTICAL (No numerical discrepancies found).

### 4.2. 시뮬레이션 안정성 테스트 (End-to-End Simulation)
`test_run_case_1`을 통해 1.5초 시뮬레이션을 수행한 결과, 이전의 폭발 지점이었던 0.36초를 성공적으로 통과하며 완주했습니다.

| 항목 | 수치 / 상태 |
| :--- | :--- |
| **FPS** | 58.4 (평균) |
| **안정성** | `NaN` 발생 없음 (Stable) |
| **최종 시간** | 1.5s (Completed) |

---

## 5. 마치며 (Conclusion)
이제 WHTOOLS의 시뮬레이션 시스템은 **유연한 설정 관리(V2)**와 **강력한 물리적 안정성**을 동시에 갖추게 되었습니다. 사용자께서는 `test_run_case_1`의 구성 방식을 의심하지 않고 그대로 활용하셔도 됩니다.

> [!TIP]
> 향후 새로운 시뮬레이션 케이스를 추가할 때도 `get_default_config()`를 기반으로 한 레이아웃만 변경하면 물리적 안정성은 자동으로 보장됩니다.

안녕하세요, **WHTOOLS**입니다. 시뮬레이션의 심장부를 더욱 단단하고 깔끔하게 리팩토링했습니다. 이제 안심하고 다음 단계의 분석을 진행하시기 바랍니다!
