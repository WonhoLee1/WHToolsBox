# Task List: Mass Balancing 고도화 및 Config 통합 (2026-03-26)

- [ ] **[Phase 1] `DropSimulator` 클래스 내 Balancing 로직 통합**
    - [ ] `config` 파라미터에 `enable_target_balancing`, `target_mass`, `target_cog`, `target_moi`, `num_balancing_masses` 추가 대응
    - [ ] `setup()` 메서드에서 `enable_target_balancing` 확인 및 자동 수행 로직 추가
- [ ] **[Phase 2] `calculate_required_aux_masses` 메서드 고도화**
    - [ ] 1, 2, 3, 4, 8개 질량체 지원 로직 구현
    - [ ] 박스 바운딩 영역(`box_w`, `box_h`, `box_d`) 내 위치 제한(Clipping) 로직 추가
    - [ ] `target_moi`가 없을 경우 CoG 매칭 위주로 배치하고 MoI 변화량 계산
- [ ] **[Phase 3] 결과 리포팅 및 비교 기능 강화**
    - [ ] `apply_balancing` 시 Baseline vs Target vs Final 상태 비교 테이블 출력
    - [ ] `summary_report.txt`에 보정 결과 상세 기록
- [ ] **[Phase 4] 검증 및 예제 업데이트**
    - [ ] `run_drop_simulation_cases.py`에서 새로운 balancing 옵션 적용 테스트
    - [ ] 최종 Walkthrough 작성
