# TODO List - VTKHDF Export Compliance & Stability

- [ ] `whts_exporter.py` 수정
    - [ ] `ConnectivityOffsets` -> `ConnectivityIdOffsets` 로 명칭 변경 (ParaView 6.0 대응)
    - [ ] `PartOffsets` 오프셋 계산 로직 수정 (Point 기반 오프셋 적용)
    - [ ] 데이터 스트리밍 루프 내 결과 존재 여부 체크 및 에러 방어 로직 추가
    - [ ] 루프 내 예외 발생 시 상세 Traceback 출력 추가
- [ ] ParaView 대시보드 스크립트 템플릿 수정
    - [ ] `ModelVariables` 대응 로직 강화
- [ ] 검증
    - [ ] `run_drop_simulation_cases_v6.py` 재실행 및 Exit Code 확인
    - [ ] ParaView에서 필드 가시화 테스트
