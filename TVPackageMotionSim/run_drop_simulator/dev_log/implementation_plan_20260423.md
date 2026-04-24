# Implementation Plan - WHTS Engine Interactive Refinement (2026-04-23)

본 계획은 `whts_engine.py`와 `whts_control_panel.py`를 고도화하여 사용자 경험을 프리미엄급으로 끌어올리고, MuJoCo의 인터랙티브 기능을 극대화하는 것을 목표로 합니다.

## 1. 주요 목표 (Core Objectives)

- **상호작용성 극대화**: 뷰어 내 실시간 상태 표시 및 정밀 제어 단축키 추가.
- **히스토리 관리 유연화**: 무한 시뮬레이션 중에도 선택적으로 데이터를 기록할 수 있는 'Rec' 기능 구현.
- **안정성 강화**: 스냅샷 점프 시 발생할 수 있는 데이터 불일치 방지 및 리소스 정리 로직 보완.
- **UI 일관성**: PySide6 컨트롤 패널과 MuJoCo 뷰어 간의 상태 동기화 및 브랜드 아이덴티티 적용.

## 2. 완료된 구현 작업 (Completed Tasks)

### Task 1: 카메라 XML 추출 기능 강화
- [x] `_export_camera_xml` 메서드 정밀도 향상 (소수점 4자리).
- [x] MuJoCo 뷰어 내 'C' 키 단축키 연동 및 실시간 로그 출력.
- [x] 컨트롤 패널 내 'Capture Camera XML' 버튼 배치 및 피드백 강화.

### Task 2: 'Reset to Start' 및 제어 기능 확장
- [x] `R` 키를 눌렀을 때 초기 스냅샷으로 즉시 복구하는 `_reset_simulation()` 구현.
- [x] 'Slow Motion' 토글 단축키 (`S` 키) 및 컨트롤 패널 버튼 연동 (0.2x 속도).
- [x] 히스토리 기록 토글 (`L` 키) 및 무한 시뮬레이션 모드 지원.

### Task 3: 브랜드 로고 및 UI 고도화
- [x] `TVPackageMotionSim/sidebar_logo.png` 파일을 100px 크기로 컨트롤 패널 상단에 배치.
- [x] 컨트롤 패널 버튼 상태(Color/Style)와 엔진 상태 간의 실시간 동기화.
- [x] 고해상도 렌더링을 위한 Smooth Transformation 적용.

### Task 4: 데이터 일관성 및 로깅 최적화
- [x] `_truncate_histories` 메서드 도입으로 스냅샷 이동 시 데이터 무결성 확보.
- [x] `rich` 라이브러리를 활용한 고해상도 터미널 리포트 UI (REC/SLOW 표시).
- [x] 모든 신규 메서드에 상세 Docstring 추가 (user_global 준수).

## 3. 향후 과제 (Backlog)
- #005: VTKHDF 멀티스레드 익스포트 구현 (대용량 대응).
- #006: 파트별 가우시안 시그마(Sigma) 자동 최적화 알고리즘.
- 실시간 응력 등고선(Stress Contour)의 뷰어 내 근사 시각화 연구.

---
*Last Updated: 2026-04-23 by Antigravity*
