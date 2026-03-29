# Global Structural Metrics (GTI, GBI) Implementation Plan

교수님의 직관적인 비틀림(Twist) 정의와 평판 전체의 건전성 평가를 위해, 개별 블록 단위를 넘어선 **전체 컴포넌트 레벨의 지표(Global Metrics)**를 도입합니다.

## Proposed Changes

### 1. Global Torsional Index (GTI)
- **개념**: 교수님께서 정의하신 "양 끝단 모서리의 상하 회전 방향 차이"를 정량화합니다.
- **계산**: 
    - 판의 장축(Major Axis)을 기준으로 양쪽 끝단(End A, End B)의 위치를 식별합니다.
    - `GTI = rotation_avg(End A) - rotation_avg(End B)` (축방향 회전 기준)
    - 이 값이 클수록 판 전체가 꼬이는(Warpage) 비틀림 변형이 심함을 의미합니다.

### 2. Global Bending Index (GBI)
- **개념**: 판의 전체적인 굽힘(Camber) 정도를 평가합니다.
- **계산**:
    - 판의 중심선(Centerline) 상의 블록들의 수직 변위(Z-displacement) 분포를 분석합니다.
    - 선형 회귀(Linear Regression) 대비 최대 편차(Maximum Deviation)를 측정하여 GBI로 정의합니다.

### 3. UI 및 데이터 반영 (postprocess_ui.py & run_drop_simulation_v3.py)
- **[MODIFY] [run_drop_simulation_v3.py](file:///C:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_v3.py)**
    - `_finalize_simulation()` 내에 GTI, GBI 계산 로직 추가.
    - `self.result.global_metrics` 에 컴포넌트별 요약값 저장.
- **[MODIFY] [postprocess_ui.py](file:///C:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/postprocess_ui.py)**
    - [탭 2] 구조 해석 영역 하단에 '전체 평판 지표 요약' 테이블 추가.
    - GTI, GBI를 시계열 그래프 선택 항목에 추가.

## Verification Plan

### Automated Tests
- 낙하 충격 후 판의 양 끝단 데이터를 추출하여 수기 계산 값과 GTI 지표가 일치하는지 검증.
- 인위적으로 뒤틀린(v4_fem 등에서 생성된) 모델에 대해 GBI가 정상적으로 검출되는지 확인.

### Manual Verification
- UI 하단 요약 테이블에서 GTI 값이 교수님이 정의하신 "양 끝단 반대 회전" 상황에서 최대가 되는지 확인.
