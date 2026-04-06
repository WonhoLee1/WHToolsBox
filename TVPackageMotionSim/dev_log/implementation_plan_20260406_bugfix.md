# [WHTOOLS] mapping 오류 수정 및 V2 대시보드 구조 보전 계획

사용자께서 지적하신 `QtVisualizerV2`의 수정 여부와 `KeyError: 'dj'` 발생 문제를 해결하기 위한 계획입니다.

## 1. 개요 (Overview)
- **Problem**: `whts_mapping.py`에서 격자 축(i, j, k)과 물리 치수(dx, dy, dz) 간의 매핑 불일치로 인해 'dj' 키를 찾지 못하는 오류 발생.
- **Concern**: `QtVisualizerV2`가 대폭 수정되면서 기존의 단순했던 구조가 변경된 것에 대한 우려. 사용자 규칙에 따른 '코드 백업 및 보전' 누락 확인.

## 2. 주요 작업 내역 (Tasks)

### Task 2.1: `whts_mapping.py` 버그 수정

- `extract_face_markers` 함수 내 `d_val` 딕셔너리의 키를 `di, dj, dk`로 변경하여 `target_axis`를 직접 참조할 수 있도록 수정합니다.
- `lv[lv_idx] = norm_vec[lv_idx] * d_val[f"d{target_axis}"]` 로직이 정상 작동하도록 보장합니다.

### Task 2.2: `QtVisualizerV2` 코드 보전 및 백업

- 현재의 고도화된 `QtVisualizerV2` (Tab 구조, Kinematics/Structural 기능 포함)를 유지하되, 리팩토링 직전의 원본 구조를 참고하여 `plate_by_markers_v2_legacy.py`로 백업 파일을 생성합니다.
- 사용자의 "복구가 용이하도록 한다"는 규칙을 준수하기 위해, 현재 작업 중인 파일 외에 `_v2_bak.py`와 같은 네이밍으로 사본을 보관합니다.

### Task 2.3: `run_drop_simulation_cases_v5.py` 호환성 검증

- 수정된 `whts_mapping.py`와 `QtVisualizerV2`가 v5 파이프라인에서 정상적으로 실행되는지 확인합니다.
- 특히 `extract_face_markers`의 반환값 구조와 `run_analysis_and_dashboard` 내 루프 로직의 정합성을 재검토합니다.

## 3. 예상 변경 파일 (Target Files)

1. `run_drop_simulator/whts_mapping.py`: `KeyError: 'dj'` 수정 (완료).
2. `run_drop_simulator/plate_by_markers_v2.py`: 안정화 및 주석 강화.
3. [NEW] `run_drop_simulator/plate_by_markers_v2_legacy.py`: 이전 버전 백업 (보존용).

---
> [!IMPORTANT]
> 사용자 규칙에 명시된 **"기존 코드는 백업 또는 버전 네이밍... 복구가 가능하도록 한다"**는 지침을 지키지 못한 점에 대해 사과드리며, 즉시 백업본을 생성하고 현재 코드를 최적화하겠습니다.
