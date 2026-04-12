# [WHTOOLS] ParaView 차세대 분석 대시보드 자동화 완료

본 작업을 통해 시뮬레이션 종료 후 분석 결과를 확인하기까지의 수동 조작을 "제로(0)"로 만드는 **Full-Autonomous Post-Processing Pipeline**을 성공적으로 구축하였습니다.

## 주요 성과 (Key Achievements)

### 1. VTKHDF 1.0 Transient Unified Mesh 엔진 (Source of Truth)
사용자님의 VTKHDF 노하우를 완벽하게 이식하여, 분산되어 있던 부품 데이터를 단 하나의 바이너리 파일(`Result.vtkhdf`)로 통합하였습니다.
- **바이너리 성능**: 289MB 규모의 모든 타임스텝 데이터를 단일 파일에 안전하게 기록.
- **Win32 Lock Bypass**: ParaView가 파일을 점유하고 있어도 해석 결과를 저장할 수 있도록 자동 접미사(`_1`, `_2`) 부여 로직 적용.
- **데이터 통합**: 18개 부품을 하나의 `UnstructuredGrid`로 병합하고 `PartID`를 부여하여 효율적으로 관리.

### 2. "Zero-Click" ParaView 대시보드 자동화
해석 완료 후 ParaView가 켜질 때, 사용자 정의 스크립트(`whts_auto_dashboard.py`)를 통해 전문적인 분석 화면이 즉시 구성됩니다.
- **3D Render View**: `displacement_vec`를 이용한 Warp 애니메이션 자동 적용.
- **2D XY Chart View**: 시뮬레이션 전체 기간 동안의 **최대 응력 히스토리(Max Von-Mises)** 그래프 자동 생성.
- **다크 모드**: 전문가용 Elegant Dark 배경 설정.

### 3. 영구 매크로 등록 (One-Click Restore)
사용자의 `AppData/Roaming` 폴더를 탐색하여 ParaView 상단 메뉴에 `WHTOOLS_Dashboard` 버튼을 자동으로 등록하였습니다. 파일을 직접 열었을 때도 버튼 하나로 대시보드 레이아웃을 복구할 수 있습니다.

## 작업 상세

### 수정된 파일
- [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py): VTKHDF 엔진 및 자동화 로직 핵심 구현.
- [v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py): 파이프라인 최종 통합.

### 검증 결과
- **[SUCCESS]** `Result.vtkhdf` 생성 확인 (Size: 289MB).
- **[SUCCESS]** ParaView 자동 런처 작동 확인 (`--script` 인자 포함).
- **[SUCCESS]** `AppData/ParaView/Macros/WHTOOLS_Dashboard.py` 등록 확인.

---

> [!TIP]
> 이제 해석 파이프라인은 단순히 수치를 계산하는 도구를 넘어, **완성된 분석 보고서 화면**까지 전달하는 진정한 의미의 자율 주행 엔지니어링 시스템으로 진화하였습니다.
