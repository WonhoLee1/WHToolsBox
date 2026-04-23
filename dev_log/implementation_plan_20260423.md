# Implementation Plan: WHTS Engine Refactoring and Optimization (2026-04-23)

## 1. 개요 (Overview)
`whts_engine.py`의 코드 구조를 최상급으로 정리하고, 상세한 주석 및 무결점 동작을 보장하기 위한 리팩토링 작업을 수행합니다. `Agent_Python`, `Agent_FEM`, `Agent_JAX`의 전문성을 결합하여 고성능 및 고신뢰성 물리 엔진 아키텍처를 구축합니다.

## 2. 목표 (Goals)
- **최상급 코드 구조**: OOP 원칙 준수, 모듈화 강화, 명확한 책임 분리.
- **상세한 주석**: 모든 클래스, 메서드, 핵심 로직에 대한 한국어 Docstring 및 인라인 주석 추가.
- **오동작 무결점**: 로버스트한 예외 처리, 정밀한 물리 연산, 안정적인 시뮬레이션 루프 확보.
- **사용자 규칙 준수**: `koreanize-matplotlib` (필요 시), 상세한 매개변수 설명, 시스템적 변수 관리.

## 3. 리팩토링 전략 (Refactoring Strategy)

### 3.1. 아키텍처 개선 (Architecture)
- **DropSimulator 클래스 분해**:
    - `StateDataManager`: 시뮬레이션 상태 및 히스토리 관리 전담.
    - `PhysicsEngine`: 공기 저항, 소성 변형 등 물리 계산 콜백 관리.
    - `Orchestrator`: 전체 시뮬레이션 제어 및 UI/리포팅 연동.
- **경로 관리**: `pathlib.Path`를 사용하여 Windows 환경에서의 경로 호환성 극대화.
- **로깅 시스템**: `rich.logging` 및 `rich.console`을 활용한 가독성 높은 로그 출력.

### 3.2. 물리 로직 정밀화 (Physics)
- **Aerodynamics (공기 역학)**: `Agent_Prof`의 검증을 거친 수학적 수식 기반으로 정리.
- **Plasticity (소성 변형)**: `Agent_FEM`의 이론적 배경을 바탕으로 접촉 압력 및 변형량 계산 로직 최적화.
- **Auto-Balancing**: 질량 및 관성 모멘트 밸런싱 로직의 엄밀성 강화.

### 3.3. 안정성 및 성능 (Stability & Performance)
- **Exception Handling**: 모델 로드, 시뮬레이션 단계, 결과 저장 등 각 단계별 `try-except-finally` 적용.
- **Type Hinting**: `typing` 모듈을 사용한 완벽한 타입 어노테이션.
- **Memory Optimization**: 히스토리 데이터 축적 시 메모리 사용량 모니터링 및 필요 시 최적화.

## 4. 상세 작업 단계 (Step-by-step Tasks)
1. **백업 생성**: `whts_engine.py`의 현재 버전을 별도 파일로 백업.
2. **Implementation Plan & Task List 작성**: 현 문서 및 `task_20260423.md` 생성.
3. **핵심 모듈 리팩토링**: `DropSimulator` 클래스 재설계 및 구현.
4. **물리 콜백 최적화**: 공기 역학 및 소성 변형 로직 정교화.
5. **UI 및 후처리 연동 강화**: `whts_gui`, `whts_postprocess_ui`와의 인터페이스 정리.
6. **검증 및 테스트**: 기본 낙하 시나리오를 통한 정상 동작 확인.

## 5. 기대 결과 (Expected Outcomes)
- 유지보수가 용이한 고도로 구조화된 코드베이스.
- 물리적 엄밀성이 확보된 정밀 시뮬레이션 결과.
- 사용자에게 전문적이고 신뢰감을 주는 터미널 인터페이스 및 로그.
