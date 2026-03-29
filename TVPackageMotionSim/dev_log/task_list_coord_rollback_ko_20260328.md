# MuJoCo 좌표계 및 모델 빌더 원상 복구 작업 현황 (2026-03-29)

- [ ] `run_discrete_builder/__init__.py` 백업본(1,471라인)으로 전체 교체
- [ ] 복구된 빌더의 좌표계(Z=Depth, Y=Height) 설정 최종 확인
- [ ] `run_drop_simulation_v3.py` 내의 운동학 및 좌표 의존 로직 정렬 확인
- [ ] 복구된 모델 빌더의 정상 작동 여부 검증 (XML 생성 테스트)
