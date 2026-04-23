# Walkthrough: WHTS Engine Refactoring (2026-04-23)

## 1. 주요 변경 사항 (Key Changes)

### 1.1. 구조적 개선 (Structural Improvements)
- **모듈화**: `DropSimulator` 클래스를 논리적 메서드로 분해하여 가독성과 유지보수성을 극대화했습니다.
- **로깅 시스템 도입**: `logging` 모듈과 `rich`를 통합하여 가독성 높은 전문적인 터미널 인터페이스를 구축했습니다.
- **Pathlib 적용**: Windows 환경에서의 경로 처리를 위해 `pathlib.Path`를 전면 도입했습니다.
- **타입 힌팅**: 모든 핵심 메서드에 명확한 타입 어노테이션을 추가하여 정적 분석 및 협업 효율을 높였습니다.

### 1.2. 물리 로직 정밀화 (Physics Refinement)
- **공기 역학 (Aerodynamics)**: Quadratic Drag, Viscous Drag, Squeeze Film Effect를 각각 명확한 메서드로 분리하고 물리적 주석을 추가했습니다.
- **소성 변형 (Plasticity)**: 접촉 압력 기반의 영구 변형 로직을 최적화하고, 변형률에 따른 실시간 시각적 피드백(Color Mapping) 시스템을 강화했습니다.
- **상태 관리**: 시뮬레이션 상태 변수와 히스토리 데이터를 체계적으로 관리하도록 초기화 로직을 정비했습니다.

### 1.3. 코드 품질 및 문서화 (Code Quality & Documentation)
- **Google 스타일 Docstring**: 모든 클래스와 메서드에 한국어 상세 설명을 추가했습니다.
- **예외 처리**: 주요 작업 단계에 `try-except` 블록을 적용하여 오동작 시에도 안전한 종료 및 로그 기록이 가능하게 했습니다.

## 2. 코드 구조 상세 (Code Structure Details)

### 🎬 메인 루프 (Main Loop)
- `simulate()`: 전체 시뮬레이션 흐름 제어.
- `_main_loop()`: 타임스텝 진행, 물리 계산, 데이터 수집, 진행 상황 보고를 오케스트레이션합니다.

### 🧠 물리 엔진 제어 (Physics Control)
- `_physics_control_callback()`: MuJoCo의 `mjcb_control` 연동.
- `_apply_aerodynamics()`: 복합 공기 역학 모델 적용.
- `_apply_plasticity_v2()`: 동적 접촉 면적을 고려한 소성 변형 연산.

### 📊 데이터 및 리포팅 (Data & Reporting)
- `_collect_history()`: 기구학 및 물리 지표 기록.
- `_report_progress()`: `rich`를 활용한 실시간 진행 상황 및 핵심 물리 지표(SE, PRS, PE, DF) 출력.

## 3. 향후 계획 (Future Plans)
- **JAX 연동 강화**: 민감도 해석을 위한 자동 미분 파이프라인과의 인터페이스 최적화.
- **대규모 배치 시뮬레이션**: 여러 시나리오를 병렬로 실행하고 통합 리포트를 생성하는 기능 확장.
