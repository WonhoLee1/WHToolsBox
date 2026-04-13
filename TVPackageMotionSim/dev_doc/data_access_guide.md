# [WHTOOLS] Simulation Data Access Guide

본 문서는 `DropSimulator`와 `PlateAssemblyManager`에서 생성된 데이터를 분석 및 커스텀 로직에 활용하기 위한 코드 접근 방법을 설명합니다.

## 1. 시뮬레이션 Raw 데이터 (`DropSimResult`)

`DropSimulator.simulate()`가 완료되면 `self.result`에 물리 엔진의 모든 이력이 담깁니다.

```python
# run_drop_simulation_cases_v6.py 예시
sim = DropSimulator(cfg)
sim.simulate()
result = sim.result # DropSimResult 객체

# (1) 시간축 데이터
times = result.time_history  # List or np.ndarray

# (2) 기구학 데이터 (Meters 단위)
positions = result.pos_hist        # (Frames, Bodies, 3)
velocities = result.vel_hist       # (Frames, 3) - Root Body 기준
accelerations = result.acc_hist    # (Frames, 3) - Root Body 기준

# (3) 코너 가속도 (Calib. 에이전트용 추천 데이터)
corner_accs = result.corner_acc_hist # (Frames, 8, 3)
```

## 2. 구조 해석 분석 데이터 (`ShellDeformationAnalyzer`)

`PlateAssemblyManager.run_all()` 실행 후, 각 파트별 해석 결과에 접근할 수 있습니다.

```python
# manager 생성 및 해석 실행 후
for analyzer in manager.analyzers:
    name = analyzer.name  # 예: "Opencell_Front"
    res = analyzer.results # Dictionary 형태의 가속된 JAX 결과 데이터
    
    # (1) 2D 면(Surface) 그리드 정보
    # res['grid_x'], res['grid_y']는 분석 대상 면의 도면 좌표를 나타냄
    
    # (2) 시간에 따른 변형량 및 응력 (mm, MPa 단위)
    disp = res['Displacement [mm]']        # (Frames, Grid_Y, Grid_X)
    bending = res['Principal Bending [deg]'] # (Frames, Grid_Y, Grid_X)
    stress = res['Bending Stress [MPa]']   # (Frames, Grid_Y, Grid_X)

    # (3) 분석 신뢰도 및 수치 지표
    f_rmse = np.mean(res['rmse'])   # 피팅 오차 (Fitting RMSE)
    r_rmse = np.mean(res['r_rmse']) # 강체 변환 오차 (Rigid RMS error)
```

## 3. 데이터 활용 팁

> [!tip] **SVD/PCA 기반 로컬 좌표계**
> `analyzer.ref_basis`를 통해 시뮬레이션의 글로벌 좌표계를 각 파트 표면의 로컬 좌표계(U, V, Normal)로 변환하는 회전 행렬을 얻을 수 있습니다.

> [!warning] **단위 주의**
> 시뮬레이션 Raw 데이터(`result`)는 기본적으로 **Meter[m]** 단위이며, 분석 매니저(`manager`)에 의해 처리된 결과는 가독성을 위해 **Millimeter[mm]**로 스케일링되어 있습니다.
