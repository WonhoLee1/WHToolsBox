# WHTOOLS Kirchhoff Plate Ultimate Analyst (v7)

본 계획서는 WHTOOLS 구조 해석 파이프라인의 최종 단계인 **고유연 평판 변형 해석 시스템**의 고도화 및 UI 통합을 목표로 합니다.

## Proposed Changes

### 1. 물리 엔진 (Mechanics Engine)
- **[MODIFY] [plate_by_markers.py](file:///c:/Users/GOODMAN/WHToolsBox/plate_by_markers.py)**
    - JAX 기반 Kirchhoff Plate Optimizer 구현 (4차 다항식 기반 곡률 피팅).
    - Von-Mises 응력, 주곡률 기반 PBA(Principal Bending Axis), Strain Energy 산출 로직 통합.

### 2. 기하학 정렬 (Kinematics & Alignment)
- **PCA 기반 좌표계 생성**: 3x3 마커의 분포를 분석하여 평판의 로컬 좌표축(X, Y, Normal)을 자동 정의.
- **Geometric Centering**: 마커 분포의 기하학적 중심을 산출하여 2000x1400mm 평판의 중심과 정렬.

### 3. 전문가급 대시보드 (Qt UI/UX)
- **재생 컨트롤**: `|<<`, `<`, `▶/||`, `>`, `>>|` 버튼 및 자동 재생(QTimer) 기능.
- **필드 선택 시스템**: Matplotlib 상단에 `QComboBox`를 배치하여 Displacement, Stress, Strain, PBA 필드를 실시간 교차 분석.
- **시각화 최적화**:
    - **PyVista**: 투명 파란색 바닥면(2500x2500mm), XYZ 좌표축, 9pt 세로 Legend, In-place Mesh 업데이트(Flicker-free).
    - **Matplotlib**: 20:14 실제 비례(Aspect Ratio) 강제 적용.

### 4. 사용자 피드백 (Terminal)
- **진행 바(Progress Bar)**: 해석 단계에서 터미널에 실시간 진행 상황 표시 (`[Progress] |█████---|`).

## Verification Plan

### Automated Tests
- `python plate_by_markers.py` 실행을 통해 80프레임 코너 낙하 시나리오 해석 및 UI 정상 작동 확인.

### Manual Verification
- 재생 버튼을 통한 애니메이션 부드러움 확인.
- 콤보박스 변경 시 Matplotlib 차트 즉시 업데이트 확인.
- 바닥면과 평판의 물리적 위치 관계(z > 0) 확인.
