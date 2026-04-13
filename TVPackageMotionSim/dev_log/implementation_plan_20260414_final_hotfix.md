# Implementation Plan - [v6.4] Final Pipeline Refinement & Hotfix

ParaView Unicode 오류 및 종료 코드를 완벽하게 해결하고, 손상된 Exporter 코드를 정교하게 재구축합니다.

## User Review Required

> [!IMPORTANT]
> - **코드 전수 재작성**: `whts_exporter.py`의 손상된 구간을 포함하여 클래스 전체를 깨끗한 상태로 재작성합니다.
> - **경로 처리 일원화**: 모든 외부 프로세스 호출 시 윈도우 스타일 경로(`\`)를 리눅스 스타일(`/`)로 통일하여 인코딩 오류를 원천 차단합니다.

## Proposed Changes

### [Exporter & Termination]

#### [MODIFY] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)
- 클래스 전체 정화: `GLB` 내보내기 복구 및 중복 메서드 삭제.
- `launch_paraview` 내 경로 치환 로직 위치 교정 (파일 기록 이전 단계로 이동).

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- 시뮬레이션 종료 시 `os._exit(0)`을 사용하여 터미널 리다이렉션 환경에서도 깨끗한 리턴 보장.

## Verification Plan

### Manual Verification
- `python run_drop_simulation_cases_v6.py` 실행 후:
    1. **ParaView 자동 실행 확인**: 더 이상 `UnicodeEscape` 에러가 뜨지 않고 대시보드 창이 뜨는지 확인.
    2. **종료 상태 확인**: `Exit code: 0`으로 정상 종료되는지 확인.
