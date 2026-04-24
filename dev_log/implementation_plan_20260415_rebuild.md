# VTKHDF 내보내기 엔진 제로베이스 재구축 계획

기존의 부분적인 수정을 중단하고, ParaView 6.0 및 VTKHDF 1.0 규격의 핵심 원리에 따라 내보내기 엔진을 완전히 새로 작성하여 안정성과 정확성을 확보합니다.

## 핵심 설계 원칙

1. **정적 토폴로지(Static Topology)**: 
    - 부품 간의 연결 상태(Connectivity, Offsets, Types)는 프레임마다 복제하지 않고 루트 그룹에 단 한 번만 기록합니다. 
    - 이를 통해 ParaView의 인덱싱 오류를 원천 차단하고 메모리 효율을 높입니다.
2. **동적 지오메트리(Temporal Geometry)**: 
    - 변형된 좌표(`Points`)와 결과 필드(`PointData`)만 시계열로 저장합니다. 
    - `Steps` 그룹은 `PointOffsets`만 참조하여 불필요한 인덱스 간섭을 제거합니다.
3. **엄밀한 좌표 변환**: 
    - 글로벌 좌표 산출 시 기저 행렬의 전치(`Basis.T`)를 적용하여 수학적 무결성을 증명합니다.
4. **결측치 패딩(Frame Padding)**: 
    - 해석이 조기 중단된 파트가 있더라도 마지막 유효 프레임을 패딩하여 전체 타임라인의 연속성을 유지합니다.

## 세부 변경 사항

### [WHTOOLS Post-Processor Exporter]

#### [NEW BUILD] [whts_exporter.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_exporter.py)

- **`export_to_vtkhdf` 재작성**:
    - [1단계] 전체 부품의 정적 메쉬 정보(Topology) 통합 및 루트 데이터셋 작성.
    - [2단계] `Steps` 그룹 생성 및 시계열 메타데이터(`Values`, `PointOffsets`) 정의.
    - [3단계] `Points`, `displacement_vec`, `Von-Mises [MPa]`, `PartID` 데이터셋 chunked 생성.
    - [4단계] 프레임별 좌표 변환(`@ rb.T` 적용) 및 필드 데이터 스트리밍.
- **`export_to_glb` 수정**:
    - 전치 행렬 수식을 적용하여 GLB 결과물의 위치 정렬 오류 수정.

## 검증 계획

### Automated Tests
- `python run_drop_simulation_cases_v6.py` 실행 시 로그에 `h5py` 관련 오류가 없는지 확인.
- `PartID` 필드가 모든 스텝에서 정상적으로 유지되는지 확인.

### Manual Verification
- ParaView 6.0에서 슬라이더를 끝까지 움직여도 크래시가 발생하지 않는지 확인.
- 모든 부품의 글로벌 위치가 MuJoCo의 지오메트리 배치와 일치하는지 확인.
