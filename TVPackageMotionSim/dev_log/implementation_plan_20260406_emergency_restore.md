# [WHTOOLS] 최신 버전(1189라인) UI 환경 긴급 복구 계획

사용자님의 지적대로 `bak.py`는 과거 버전이었으며, 세션 시작 시점의 최신 상태(1189라인)를 손실 없이 복구하기 위한 계획입니다.

## 1. 개요 (Overview)
- **Problem**: `plate_by_markers_v2.py`를 `bak.py`로 복구했으나, `bak.py`는 매우 낮은 버전(412라인)으로 확인됨. 세션 시작 시 파일은 1189라인의 고도화된 상태였음.
- **Solution**: 제가 세션 시작 시(`Turn 1`) 조회했던 1189라인의 전체 텍스트를 이용하여 `plate_by_markers_v2.py`를 원상복구합니다.

## 2. 복구 작업 (Recovery Tasks)

### Task 2.1: 세션 시작 시점의 텍스트 추출
- `Turn 1`에서 `view_file`을 통해 확보된 원본 코드(1-1189라인)를 다시 구성합니다.
- 이 버전은 `ShellDeformationAnalyzer`, `PlateAssemblyManager`, `QtVisualizerV2` (manager 인자 필수 버전) 등을 모두 포함하고 있습니다.

### Task 2.2: 파일 덮어쓰기 (Overwriting)
- 구성된 1189라인의 코드를 `run_drop_simulator/plate_by_markers_v2.py`에 다시 써서 세션 시작 전과 동일한 상태를 만듭니다.

### Task 2.3: `whts_mapping.py` 수정 사항 유지
- `KeyError: 'dj'`를 해결한 `whts_mapping.py`의 수정 사항은 `run_post_only_v5.py` 실행에 필수적이므로 유지합니다.

## 3. 검증 계획 (Verification)
1. `run_post_only_v5.py`를 실행하여 18개 부품 분석이 정상적으로 이뤄지는지 확인 (사용자 터미널 결과와 대조).
2. UI가 기존에 사용하시던 최신 대시보드 형태(Tab 없는 단일 레이아웃 등)로 복구되었는지 사용자 확인 요청.

---
> [!CAUTION]
> `bak.py`가 최신일 것이라고 오판하여 원본을 덮어쓴 점 깊이 반성합니다. 즉시 세션 시작 시점의 "진짜 최신" 코드를 복구하겠습니다.
