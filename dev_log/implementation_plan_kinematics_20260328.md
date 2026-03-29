# Implementation Plan: Kinematics Analysis Parity (v2 Style)

기구학 분석 결과에서 '변위(Displacement)' 데이터가 상대 좌표로 계산되어 현실과 괴감이 발생하는 문제를 해결하고, v2 버전과 같은 절대 좌표계 기반의 '위치(Position)' 정보로 원복합니다.

## 1. 개요 (Overview)
- **문제점**: 현재 `PostProcessingUI`는 기구학 변위 가시화 시 시작점($t=0$)을 0으로 만드는 차분 연산을 수행하여, 낙하 높이와 같은 절대 좌표 정보가 손실됨.
- **해결책**: 가시화 로직에서 차분 연산을 제거하고, UI 레이블을 'Position (World)' 등으로 변경하여 절대 좌표임을 명시함. Y축 단위를 추가하여 물리량의 현실감을 높임.

## 2. 주요 작업 항목 (Task Items)

### 2.1. [UI] 기구학 데이터 종류 명칭 변경
- `self._kin_dtype_vars` 내의 `"displacement"` 키의 표시 이름을 사용자가 이해하기 쉬운 **"Position (Coord)"** 또는 **"위치(좌표)"** 컨셉으로 매핑.
- UI 버튼 및 레이블 업데이트.

### 2.2. [Logic] 절대 좌표계 원복
- `postprocess_ui.py`의 `_get_kinematic_series` 함수에서 `dtype == "displacement"`일 때 적용되던 `arr = [v - arr[0] for v in arr]` 로직 제거.
- `sim.z_hist`, `sim.pos_hist` 등 절대 좌표 데이터가 그대로 그래프에 반영되도록 수정.

### 2.3. [Chart] 단위 및 축 강화
- `_on_plot_kinematics` 함수에서 `ax.set_ylabel`을 추가하여 다음 단위 명시:
    - **Position**: `Position (m)`
    - **Velocity**: `Velocity (m/s)`
    - **Acceleration**: `Acceleration (m/s^2) / G`
- 그래프 제목 및 윈도우 타이틀 보강.

### 2.4. [Stability] 범례 호환성 체크
- Matplotlib 버전 차이에 따라 `ax.legend(alpha=...)` 에러가 발생하던 부분을 `framealpha`로 전역 수정 확인.

## 3. 검증 계획
1. **정적 검증**: 시뮬레이션 시작 시 Z 좌표가 `drop_height` (예: 0.5m) 근처에서 시작하는지 확인.
2. **동적 검증**: 낙하 중 Z 좌표가 감소하여 0(지면)에 도달하는지 확인.
3. **v2 비교**: 기존 v2 리포트 그래프와 유사한 형태(절대값)를 보이는지 대조.

## 4. 백업 정보
- 파일: `postprocess_ui.py`
- 날짜: 2026-03-28
- 작성자: Antigravity (WHTOOLS AI Assistant)
