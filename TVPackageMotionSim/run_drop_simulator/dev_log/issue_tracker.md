# [WHTOOLS] Issue Tracker

본 문서는 **WHTOOLS** 프로젝트에서 발생하는 반복적인 설정 오류, 수치 불안정 이슈 및 개선 요구사항을 추적하고 관리하기 위한 문서입니다. 새로운 세션에서도 이 문서를 참고하여 동일한 이슈의 재발을 방지합니다.

## Active Issues

| ID | Issue Description | Status | Priority | Date |
|:---:|:---|:---:|:---:|:---:|
| #005 | **Multi-Thread Export**: VTKHDF 데이터 스트리밍 시 대용량 데이터 대응을 위한 멀티스레딩 최적화 | Pending | 🔵 Low | 2026-04-13 |
| #006 | **Adaptive Sigma**: 파트의 종횡비(Aspect Ratio)에 따라 가우시안 가중치 `sigma`를 동적으로 최적화하는 로직 도입 | Pending | 🔵 Low | 2026-04-13 |

## Resolved Issues

| ID | Issue Description | Resolution | Date |
|:---:|:---|:---|:---:|
| #001 | **Friction Config TypeError** | `get_friction_standard`에서 `mu`가 시퀀스(리스트)인 경우를 처리하도록 보완 | 2026-04-13 |
| #002 | **SVD Non-convergence** | `sigma` 최소값(Floor) 적용 및 SVD 전 NaN 체크 로직 추가 | 2026-04-13 |
| #003 | **Transient Export KeyError** | `analyzer.results` 유효성 체크 및 키 확인 로직 추가하여 실패 파트 스킵 지원 | 2026-04-13 |
| #004 | **UI Init Failure** | 해석 성공한 첫 번째 파트를 탐색하여 필드 구성을 수행하도록 초기화 로직 수정 | 2026-04-13 |
| #007 | **XML Default Class Naming Mismatch** | `whtb_builder.py`에서 제거되었던 `contact_` 접두사를 복구하여 기존 테스트(`v5.py`)와 호환성 유지 | 2026-04-13 |
| #008 | **ModuleNotFoundError in Pickle Load** | 시뮬레이션 결과 로드 시 `run_drop_simulator`를 포함한 상위 경로를 시스템 경로에 동적 주입하여 해결 | 2026-04-14 |
| #009 | **Numerical Instability at mm-Scale** | mm 단위 대형 좌표에서 고차 다항식 연산 시 발생하는 NaN 폭발을 방지하기 위해 Optimizer 내 좌표 정규화 로직 적용 | 2026-04-14 |
| #010 | **Cohesive Weld Config Disconnect** | `whtb_builder.py`의 하드코딩된 컴포넌트 매핑을 제거하여 `opencellcoh` 물성을 Config에서 독립적으로 제어 가능하도록 수정 | 2026-04-14 |

## Improvement Backlog

- [ ] **Multi-Thread Export**: VTKHDF 데이터 스트리밍 시 대용량 데이터 대응을 위한 멀티스레딩 최적화
- [ ] **Adaptive Sigma**: 파트의 종횡비(Aspect Ratio)에 따라 가우시안 가중치 `sigma`를 동적으로 최적화하는 로직 도입
