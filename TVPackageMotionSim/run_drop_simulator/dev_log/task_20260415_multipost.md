# Task List: 멀티 포스트프로세서 엔진 및 UI 고도화

- [x] `whts_multipostprocessor_engine.py` 리팩토링
    - [x] `PlateConfig` 로직 개선 및 자동 설정 추출 기능 추가
    - [x] `RigidBodyKinematicsManager` 클래스 이식 (Kabsch-vmap 정렬)
    - [x] `KirchhoffPlateOptimizer` & `PlateMechanicsSolver` 클래스 이식
    - [x] `ShellDeformationAnalyzer` 구조 전면 재설계 및 이 클래스들 통합
    - [x] `PlateAssemblyManager` 연동 확인
- [x] `whts_multipostprocessor_ui.py` 리팩토링
    - [x] `QtVisualizerV2.update_frame()` 고도화 (글로벌 좌표계 정밀 정렬)
    - [x] 데이터 키 정합성(Stress, Strain 등) 및 단위계(mm/MPa) 검증
- [x] 검증 및 백업
    - [x] 가상/실제 데이터를 활용한 어셈블리 정렬 안정성 테스트
    - [x] `./dev_log/` 폴더에 최종 결과물 백업 저장
- [x] 최종 Walkthrough 작성
