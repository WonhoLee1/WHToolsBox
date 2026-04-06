# [WHTOOLS] V2 UI 보존 및 고도화 버전(Premium) 분리 계획

사용자 피드백을 반영하여 `QtVisualizerV2`를 원본 상태로 보존하고, 신규 개발된 고도화 UI를 별도의 모듈로 분리하여 관리하는 계획입니다.

## 1. 개요 (Overview)
- **Feedback**: "v2 UI 개발할 때 QtVisualizerV2를 수정하면 안되지." -> 원본 V2의 구조를 유지하고 변경을 금지함.
- **Solution**: `plate_by_markers_v2.py`를 원본으로 복구하고, 제가 제안한 개선 사항은 `plate_by_markers_v2_premium.py`라는 새로운 파일로 독립시킵니다.

## 2. 세부 작업 (Tasks)

### Task 2.1: 원본 복구 및 백업 확인
- `plate_by_markers_v2.bak.py`의 내용을 `plate_by_markers_v2.py`로 덮어씌워 원본 상태로 되돌립니다.
- 이 과정에서 `QtVisualizerV2` 클래스의 기존 시그니처와 기능을 완벽히 복구합니다.

### Task 2.2: 고도화 UI의 독립 모듈화
- 현재 `plate_by_markers_v2.py`에 적용된 최신 코드(Tab 위젯 기반 UI, `load_data` 신규 메서드 등)를 `plate_by_markers_v2_premium.py`로 이동합니다.
- 클래스명은 `QtVisualizerV2Premium` 또는 사용자의 선호에 따라 가독성 있게 유지합니다.

### Task 2.3: 시뮬레이션 엔진 및 파이프라인 연동 업데이트
- **`run_drop_simulator/whts_engine.py`**: 자동 실행 시 `plate_by_markers_v2_premium.py`를 실행하도록 `--load` 아규먼트 경로 수정.
- **`run_drop_simulation_cases_v5.py`**: 임포트 경로를 `plate_by_markers_v2_premium`으로 변경하여 고도화된 분석 기능 사용 보장.

### Task 2.4: 매핑 오류(`KeyError: 'dj'`) 최종 확인
- `whts_mapping.py`에 적용된 수정 사항(`di, dj, dk` 매핑)이 두 UI 버전 모두에서 잘 작동하는지 최종 검증합니다.

## 3. 검증 계획 (Verification)
1. `run_post_only_v5.py`를 통해 신규 Premium UI가 정상 실행되는지 확인.
2. `plate_by_markers_v2.py`를 개별 실행하여 원본 Legacy UI가 훼손되지 않았는지 확인.

---
> [!IMPORTANT]
> 기존 V2 코드를 보존하지 않고 직접 수정한 것에 대한 사용자님의 우려를 깊이 이해하며, 위 계획을 통해 **Legacy 유지**와 **신규 고도화**를 완벽히 분리하겠습니다.
