# Implementation Plan - [v7.3] High-Fidelity Sampling & Autonomous Normalization

단일 강체 부품에서도 면 전체의 변형을 해석할 수 있도록 '가상 격자 샘플링'을 도입하고, v6 파이프라인에서 생략된 치수 정보(W, H)를 마커로부터 자율 복구합니다.

## User Review Required

> [!IMPORTANT]
> - **코너점 위주 추출 탈피**: 기존의 `[-1, 1]` 코너 추출 대신, 부품의 면적을 조밀하게 샘플링하는 `np.linspace` 기반 격자 마커 추출 로직을 `whts_mapping.py`에 주입합니다.
> - **자율 치수 복구**: `ShellDeformationAnalyzer(W=0, H=0)`으로 생성되어도, 초기 마커 분포로부터 부품의 실제 $W, H$를 수학적으로 유추하여 해석 정밀도를 유지합니다.

## Proposed Changes

### [Mapping Engine Enhancement]

#### [MODIFY] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- `extract_face_markers`: `for s1 in [-1, 1]` 루프를 부품의 `div` 설정에 비례하는 격자 샘플링(`np.linspace`)으로 교체.
- 이를 통해 단일 바디인 `Chassis`, `Opencell`에서도 16개 이상의 마커 확보.

### [Analyzer Intelligence]

#### [MODIFY] [whts_multipostprocessor_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_multipostprocessor_engine.py)
- `ShellDeformationAnalyzer.fit_reference_plane`: $W$ 또는 $H$가 0인 경우, `o_data`의 최대/최소 편차로부터 즉석에서 치수를 산출하여 정규화 로직에 반영.

## Verification Plan

### Automated Tests
1. `python run_drop_simulation_cases_v6.py` 실행 후 `Chassis_Front` 및 `Opencell_Front` 로그에 **Markers: 16** 이상이 찍히는지 확인.
2. `Max Stress`가 0.00이 아닌 **유의미한 공학적 수치**로 리포트되는지 확인.

### Manual Verification
1. ParaView 시각화에서 `Chassis` 표면에 바둑판 모양의 마커들이 가득 차 있는지 확인.
