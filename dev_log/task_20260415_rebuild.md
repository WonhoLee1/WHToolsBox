# TODO List - Exporter Zero-Base Rebuild

- [x] `whts_exporter.py` 전면 재작성 (Rewrite)
    - [x] `export_to_vtkhdf` 구조 재설계: 정적 토폴로지 (Static Topology)
    - [x] 기저 행렬 전치(`Basis.T`)를 적용한 좌표 변환 수식 교체
    - [x] `Points` 및 `PointData` 시계열 데이터 저장 및 프레임 패딩 추가
    - [x] Root 그룹 메타데이터(`NumberOfPoints` 등)를 시계열 배열로 전환
- [x] `export_to_glb` 수정
    - [x] GLB 수식에 `.T` 적용 및 정렬 안정화
- [/] 검증
    - [ ] `run_drop_simulation_cases_v6.py` 통합 테스트
    - [ ] ParaView 슬라이더 및 가시화 필드 확인
