# WHTOOLS Codebase Synchronization Implementation Plan (2026-04-06)

안녕하세요, **WHTOOLS**입니다. GitHub 저장소와 로컬 환경의 동기화를 위한 실행 계획입니다.

## 1. 개요 및 분석
- **GitHub 상태**: 최신 업데이트는 4월 4일이며, RMSE 정밀도 리포팅 기능이 포함되었습니다.
- **로컬 상태**: 4월 6일 버전(`whts_mapping_D260406.py`)이 로컬에 존재하며, 이는 GitHub 버전보다 진보된 페이스 인덱싱 로직을 포함하고 있습니다.
- **주요 과제**: 파일명을 `whts_mapping.py`로 표준화하여 엔진에서 정상 호출되도록 하고, GitHub의 4월 4일 업데이트 내역이 로컬에 유실 없이 반영되어 있는지 검증합니다.

## 2. 작업 절차

### 2.1. 로컬 백업 및 파일 정리
- [x] `plate_by_markers_v2.py` → `plate_by_markers_v2.bak.py` (백업 생성됨)
- [ ] `whts_mapping_D260406.py` → `whts_mapping_backup_20260406.py` (백업 생성)

### 2.2. 파일 표준화 및 동기화
- [x] 로컬의 `whts_mapping_D260406.py`를 `whts_mapping.py`로 복사/변경하여 표준 모듈명 확보.
- [x] GitHub 최신 파일을 `whts_mapping_aaa.py`로 다운로드 (비교용).
- [ ] `plate_by_markers_v2.py`의 RMSE 리포팅 로직이 GitHub의 4월 4일 버전과 동일한지 최종 확인 및 미반영 시 패치.

### 2.3. 무결성 검증
- [ ] `whts_postprocess_engine_v2.py`에서 `from .whts_mapping import TV_COMPONENTS`가 에러 없이 작동하는지 확인.
- [ ] 시뮬레이션 엔진(`whts_engine.py`)의 런타임에서 `TV_COMPONENTS` 참조 오류 없는지 검토.

## 3. 기대 효과
- **일관성 확보**: `whts_mapping` 모듈명 표준화를 통해 분석 툴의 임포트 에러 해결.
- **결과 정밀도 향상**: JAX SSR 엔진의 RMSE 피드백 로직을 안정적으로 유지.
- **버전 관리 효율화**: 파편화된 날짜별 파일명을 정리하여 GitHub 푸시 준비 완료.

---
> [!IMPORTANT]
> GitHub의 마지막 업데이트가 4월 4일이므로, 어제(4월 5일) 12시 이후의 신규 코드는 없는 상태입니다. 따라서 로컬의 4월 6일 코드를 보존하는 방향으로 동기화를 진행하겠습니다.
