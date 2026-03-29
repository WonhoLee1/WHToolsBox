# [WHTOOLS] 구조 해석 엔진 모듈화 및 시뮬레이션 최적화 Walkthrough (v4.6.2)

본 문서에서는 `whts_postprocess_engine.py` 구축과 데이터 저장 주기 가변화에 따른 변경 사항 및 사용 가이드를 상세히 설명합니다.

---

## 1. 주요 변경 사항 (Key Changes)

### 1.1. SoCs (Separation of Concerns) 기반 리팩토링
- **원인**: 2,000라인이 넘는 `whts_postprocess_ui.py`의 복잡도를 낮추기 위해 연산 로직을 분리.
- **결과**: `whts_postprocess_engine.py` 신규 생성.
  - `get_contour_grid_data`: 2D/3D 블록 데이터의 정밀 투영 및 그리드화.
  - `apply_psr_interpolation`: 고해상도 곡면 재생성(SSR) 엔진 탑재.
  - `extract_global_summary_data`: PBA, RRG 등 전역 지표 및 상태 판정 로직 통합.
- **이점**: UI와 무관하게 연산 로직만 별도로 단위 테스트(Unit Test)가 가능해졌습니다.

### 1.2. 가변 데이터 저장 주기 (`reporting_interval`) 도입
- **원인**: 매 스텝(0.001s) 데이터 저장 시 발생하는 과도한 메모리 점유 및 UI 싱크 불일치 해결.
- **결과**: `config` 내에 `reporting_interval` 파라미터 추가.
  - 기본값: `0.005s` (시뮬레이션 0.001s 기준 5스텝마다 저장).
  - 유연성: 사용자 필요에 따라 `0.01s` 등으로 자유롭게 조절 가능.
- **UI 싱크**: 슬라이더의 1스텝이 곧 `reporting_interval` 1단위와 1:1 대응되도록 정밀 튜닝.

---

## 2. 사용 가이드 (Usage Guide)

### 2.1. 시뮬레이션 설정 (Pre-Simulation)
1. 시뮬레이션 시작 시 실행되는 `Config Control UI`에서 `Physics` 그룹을 확인합니다.
2. `reporting_interval` 항목에 원하는 데이터 저장 간격을 입력합니다. (예: `0.01`)
3. `Apply & Reload` 버튼을 클릭하여 시뮬레이션을 시작합니다.

### 2.2. 포스트 프로세싱 (Post-Processing)
1. 시뮬레이션 완료 후 나타나는 `Post-Processing UI`에서 시간 슬라이더를 이동합니다.
2. 슬라이더의 한 칸 이동이 설정한 `reporting_interval` 단위와 정확히 일치하여 갱신됩니다.
3. **[탭 3] 컨투어**에서 `Sharp SSR(PSR)` 모드를 활성화하면, 듬성듬성한 데이터 사이를 정교하게 채워주는 고해상도 응력 분포를 확인할 수 있습니다.

---

## 3. 기술적 안전 장치 (Safety Features)

- **Peak Projection**: 3D 부품(쿠션 등)의 두께 방향 데이터를 2D로 투영할 때, 가장 높은 전단/굽힘 응력값을 선택하여 위험 부위를 누락하지 않습니다.
- **Persistence Check**: `nominal_local_pos`가 결과 파일에 자동 포함되도록 보장하여, 나중에 데이터를 불러와도 컨투어 재생이 가능합니다.
- **Exception handling**: PSR 연산 중 발생할 수 있는 특이 행렬 오류 등에 대해 안정적인 폴백(Fallback) 로직을 갖추고 있습니다.

---

> [!NOTE]
> **권장 사항**
> 분석 결과 파일(`.pkl`)의 용량이 너무 크다면 `reporting_interval`을 `0.01`~`0.02` 수준으로 높여 보시기 바랍니다. 충분히 부드러운 애니메이션과 가벼운 데이터 관리를 동시에 달성할 수 있습니다.

---
**WHTOOLS** 드림
