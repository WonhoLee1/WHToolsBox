# WHTOOLS Issue Tracker & Progress Log

본 문서는 개발 과정에서 발견된 주요 시스템적 오류, 튜닝 요청 사항, 회귀 버그를 기록하고 추적하여 안정적인 프레임워크 구동을 지원하기 위해 작성되었습니다.

## Active Issues

### 1. `whtb_physics.py` broadcasting ValueError
- **발견 날짜**: 2026-06-03
- **상태**: Open (수정 예정)
- **설명**: `target_inertia`가 `[Ixx, Iyy, Izz]`의 3성분으로 전달될 때, `analyze_and_balance_components` 함수 내에서 `(3,)`과 `(6,)` 크기 불일치로 인해 브로드캐스팅 오류 발생.
- **조치 계획**: 3성분 검출 시 `[Ixx, Iyy, Izz, 0.0, 0.0, 0.0]`으로 자동 패딩하는 방어 로직 적용 예정.

### 2. Optimization and DOE Framework Implementation
- **발견 날짜**: 2026-06-03
- **상태**: Planning (승인 대기)
- **설명**: 신규 최적화/DOE 엔진 및 PySide6 기반 UI 프레임워크 구축.
- **조치 계획**: implementation_plan.md 수립 완료. 사용자 승인 후 구현 착수.

---

## Resolved Issues

*(해결된 이슈 내역이 여기에 기록됩니다)*
