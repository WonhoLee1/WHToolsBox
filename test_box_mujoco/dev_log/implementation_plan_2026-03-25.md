# Implementation Plan - MuJoCo Simulation Stability Fix (2026-03-25)

## 1. 개요 (Overview)
현재 `run_drop_simulation.py`에서 발생하는 두 가지 주요 `NameError` 오류를 해결하여 시뮬레이션의 안정성과 데이터 수취 기능을 복구합니다.
- **오류 1**: `NameError: name 'gid_hits' is not defined` (소성 변형 연산 중 변수명 오기)
- **오류 2**: `NameError: name 'relevant_ids' is not defined` (배치 해석 단계에서 ID 리스트 미정의)

## 2. 세부 단계별 수정 계획 (Detailed Steps)

### 2.1. 시뮬레이션 초기화 영역 보강 (Line 590~610 사이)
- **대상 ID 수량화**: `relevant_ids`와 `relevant_ids_arr`를 생성하여 배치 해석에서 추적할 모든 컴포넌트 블록의 Body ID를 미리 확보합니다.
- **히스토리 리스트 추가**: `raw_analysis_hist`, `metrics_time_history` 등 누락된 데이터 저장용 리스트를 초기화합니다.
- **강성 프록시 설정**: `k_spring_proxy`를 `solref` 및 질량 설정을 활용해 동적으로 산출하거나 합리적인 기본값(`1e6`)으로 정의합니다.

### 2.2. 소성 변형(Plasticity) 함수 수정 (Line 639~)
- **변수명 통일**: `gid_hits`와 `geom_hits`를 `geom_hits`로 일원화합니다.
- **들여쓰기 및 로직 정리**: 
    - `data.ncon` 루프를 통한 하중/침투량 집계 단계와 이를 바탕으로 변형을 적용하는 단계를 명확히 분리합니다.
    - `target_gid`와 같은 모호한 변수명을 `gid` 또는 `target_geom`으로 통정합니다.
- **상태 추적기 연동**: `geom_state_tracker`와의 데이터 연동성을 강화하여 매 스텝의 변형이 누락 없이 기록되도록 합니다.

### 2.3. 시뮬레이션 제어 루틴 보강 (Line 871~)
- **리셋 로직 확대**: `ctrl.reset_request` 시 `raw_analysis_hist`와 `metrics` 내의 하위 리스트들도 모두 `clear()` 되도록 코드를 보강하여, 시뮬레이션 재시작 시 이전 데이터와 섞이지 않도록 합니다.

### 2.4. 배치 해석 지표 구조화 (Line 1000~)
- `metrics` 딕셔너리가 모든 컴포넌트와 행(Row)에 대해 사전에 올바른 구조(`bending`, `twist`, `energy` 등)를 갖추도록 초기화 루틴을 안전하게 구성합니다.

## 3. 기대 효과 (Expected Outcomes)
- 시뮬레이션 안정성 확보 및 런타임 오류 방지.
- 낙하 후 정확한 구조적 변형 리포트 및 그래프 생성 가능.
- 사용자의 후속 과제인 '시험 결과 매칭을 위한 파라마터 연동'을 위한 견고한 데이터 베이스 구축.
