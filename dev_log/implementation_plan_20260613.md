# WHTOOLS TV Drop Motion Simulator v6.0 설명서 PPTX 생성 계획

본 계획서는 **WHTOOLS TV Drop Motion Simulator v6.0**의 프로젝트 상세설명 및 사용 설명서를 고품질 라이트 테마(화이트/블루 계열) PPTX 파일로 자동 생성하는 파이썬 스크립트를 새로 작성하고 실행하는 계획을 담고 있습니다.

## User Review Required

> [!IMPORTANT]
> - **디자인 테마 변경**: 기존 딥 네이비(다크) 테마에서 깔끔한 **화이트/블루 계열의 라이트 테마**로 전면 전환합니다.
> - **슬라이드 추가 및 구성 변경**: 프로젝트 개요뿐만 아니라 실제 GUI 조작법, 모델 파라미터 튜닝, OpenRadioss 연동 등 구체적인 **사용 설명서 슬라이드**를 다수 추가하여 총 11장으로 구성합니다.
> - **스크립트 파일명**: 새로운 스크립트는 `make_manual_pptx.py`로 생성하며, 결과 PPTX는 `WHTools_DropSimulator_Manual.pptx`로 저장됩니다.

## Proposed Changes

### [PPTX Generator Script]

#### [NEW] [make_manual_pptx.py](file:///c:/Users/GOODMAN/WHToolsBox/make_manual_pptx.py)
* **라이트 테마 색상 팔레트 정의**:
  - `C_BG`: 밝은 소프트 블루/그레이 (`0xf4, 0xf7, 0xfc`)
  - `C_PRIMARY`: 딥 로열 블루 (`0x1a, 0x36, 0x5d`) - 제목용
  - `C_SECONDARY`: 스카이 블루 (`0x2b, 0x6c, 0xb0`) - 강조선 및 부제목용
  - `C_ACCENT`: 옅은 블루그레이 (`0xeb, 0xf8, 0xff`) - 카드/박스 배경용
  - `C_TEXT`: 다크 차콜 그레이 (`0x2d, 0x37, 0x48`) - 본문 텍스트용
  - `C_WHITE`: 흰색 (`0xff, 0xff, 0xff`) - 카드 내 배경 또는 강조용
  - `C_BORDER`: 연한 그레이 (`0xe2, 0xe8, 0xf0`) - 테두리용
* **슬라이드 상세 구성 (총 11장)**:
  - **Slide 1: 타이틀** (WHTools TV Drop Motion Simulator v6.0 설명서)
  - **Slide 2: 프로젝트 개요 및 배경** (낙하 시뮬레이션 자동화 도입 배경 및 통합 플랫폼의 목적)
  - **Slide 3: 시스템 아키텍처 및 데이터 흐름** (UI, Physics, FEM, Post-processing 레이어와 흐름)
  - **Slide 4: 핵심 기능 1 - Simulation Control Center** (실시간 GUI 제어 패널 조작법 및 기능)
  - **Slide 5: 핵심 기능 2 - JAX 기반 자율 구조 해석** (마커 궤적 기반의 Kirchhoff 박판 솔버 동작 방식)
  - **Slide 6: 핵심 기능 3 - ISTA-6 Amazon 배치 해석** (22가지 시나리오 병렬 실행 및 결과 자동 수집)
  - **Slide 7: 사용법 1 - 환경 설정 및 프로그램 실행** (ini 파일 설정 및 파이썬 진입점 실행 가이드)
  - **Slide 8: 사용법 2 - 모델 설정 및 물리 파라미터 튜닝** (Model Setup Dialog 사용법, 관성 보정 및 용접 프리셋)
  - **Slide 9: 사용법 3 - OpenRadioss FEM 해석 및 후처리** (스냅샷 자동 내보내기, Ground Penetration 보정, ParaView 연계)
  - **Slide 10: 주요 성과 및 차별성** (상용 FEA 소프트웨어 대비 효율성 및 비용/시간 단축 데이터)
  - **Slide 11: 마무리 및 Q&A** (개발팀 연락처 및 기술 스택 종합 정보)
* **구현 방식**:
  - `python-pptx` 라이브러리를 사용하여 텍스트 프레임, 사각형 도형, 선, 글머리 기호 박스 등을 프로그래밍 방식으로 조절하여 생성합니다.
  - 가로 세로 비율은 16:9 와이드스크린으로 설정합니다.

## Verification Plan

### Automated Tests
- 생성된 파이썬 스크립트를 실행하여 오류 없이 PPTX 파일이 생성되는지 확인합니다:
  ```powershell
  python c:\Users\GOODMAN\WHToolsBox\make_manual_pptx.py
  ```

### Manual Verification
- 생성 완료된 `WHTools_DropSimulator_Manual.pptx` 파일을 열어 레이아웃이 깨지지 않고 텍스트 배치가 적절한지 검토합니다.
