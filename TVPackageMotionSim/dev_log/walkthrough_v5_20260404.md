# [Walkthrough] MuJoCo Digital Twin & Plate Assembly Integration (v5)

MuJoCo 시뮬레이션 결과를 `ShellDeformationAnalyzer`와 직접 연동하여, 시뮬레이션만으로 정밀 구조 해석 및 3D 변형 가시화가 가능한 **V5 디지털 트윈 파이프라인**을 완성하였습니다.

## 주요 변경 사항

### 1. 데이터 인터페이스 구축 (`whts_data.py`, `whts_engine.py`)
- MuJoCo의 수천 개 블록 중 특정 면에 해당하는 블록을 찾기 위해 **`body_index_map`** 스키마를 도입하였습니다.
- 시뮬레이션 중 이산화 블록의 그리드 인덱스(`i, j, k`)와 MuJoCo `body_id` 간의 매핑 정보를 자동으로 저장합니다.

### 2. 자동 마커 추출 엔진 (`whts_mapping.py`) [NEW]
- 시뮬레이션 결과 파일에서 Cushion, Chassis, OpenCell 등 각 파트의 **6개 외곽면(Front, Rear, Left, Right, Top, Bottom)**을 자동으로 식별합니다.
- 식별된 면상의 블록 궤적을 `ShellDeformationAnalyzer`가 인식할 수 있는 마커 데이터 형식으로 변환합니다.

### 3. 통합 실행 파이프라인 (`run_drop_simulation_cases_v5.py`) [NEW]
- **One-Stop Execution**: 시뮬레이션이 종료되면 별도의 조작 없이도 다음과 같은 과정이 자동으로 수행됩니다.
    - 데이터 매핑 및 마커 추출
    - 파트별 6개 면에 대한 `ShellDeformationAnalyzer` 자동 생성
    - `PlateAssemblyManager`를 통해 SSR(Structural Surface Reconstruction) 해석
    - `QtVisualizerV2` 통합 대시보드 실행

## 사용 방법

터미널에서 다음 명령을 실행하여 통합 파이프라인을 구동할 수 있습니다.

```powershell
python run_drop_simulation_cases_v5.py
```

---
**WHTOOLS** 올림.
