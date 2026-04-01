# GitHub 원본 기반 엔진 안정화 및 성능 최적화 계획

사용자의 요청에 따라 GitHub의 `D260329` 브랜치 코드로 원복 완료하였습니다. 이제 이 안정적인 베이스 위에서, 문제가 되었던 "충격 구간 지연"과 "애니메이션 점프"를 해결하기 위한 **최소한의, 그러나 핵심적인** 수정만을 수행합니다.

## Proposed Changes

### 1. 실시간 동기화 '세이프티 가드' 도입
- **문제**: 연산 지연 시 시뮬레이션이 실제 시간을 따라잡으려다 보니 수백 스텝을 한꺼번에 실행하여 화면이 멈추거나 점프함.
- **해결**: `_main_loop`에서 한 프레임(Viewer Sync)당 실행할 수 있는 물리 스텝의 최대치(예: 32스텝)를 설정합니다. 지연이 이보다 클 경우, 무리하게 따라잡지 않고 점진적으로 따라잡거나, 지연이 0.2초를 넘으면 기준 시간(`start_real_time`)을 현재로 리셋하여 "점프"를 원천 차단합니다.

### 2. 소성 변형 연산(Plasticity) 병목 제거
- **문제**: 충격 시 수천 개의 접촉(Contact)이 발생하는데, 매 접촉마다 `mj_contactForce`를 호출하는 것은 파이썬 환경에서 매우 느림.
- **해결**: 접촉한 물체가 `geom_state_tracker`에 등록된 쿠션 계열인지 먼저 검사한 후, 대상일 때만 힘 연산을 수행합니다. (연산량 90% 이상 절감 기대)

### 3. 계측 시스템 정밀화 (v4.2 Parity)
- 모든 시간 계측을 `time.perf_counter()`로 일원화하여 마이크로초 단위의 시뮬레이션-실제 시간 매핑 정확도를 확보합니다.

---

## 단계별 작업 목록 (Task List)

1. [ ] **[whts_engine.py]** `_main_loop` 시간 동기화 로직 전면 개편 (Step Budgeting & Safety Reset)
2. [ ] **[whts_engine.py]** `_apply_plasticity_v2` 내 접촉 필터링 로직 추가
3. [ ] **[whts_engine.py]** `time.time()`을 `time.perf_counter()`로 교체 및 초기화 시점 정교화
4. [ ] **[whts_engine.py]** `compute_structural_step_metrics` 호출 빈도 최적화 (reporting_interval 준수)

## Verification Plan

### Manual Verification
- `run_drop_simulation_cases_v4.py` 실행 시 충격 구간(0.3s)에서 터미널 리포팅이 멈추지 않고 지속되는지 확인.
- Viewer 화면의 애니메이션이 끊기지 않고 부드럽게 이어지는지 확인.
- 지연 발생 시 터미널에 `[WHTOOLS] Timing Reset (Lag > 0.2s)` 메시지가 출력되는지 확인.
