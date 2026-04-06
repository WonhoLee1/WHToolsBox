# [WHTOOLS] V2 Dashboard Refactoring & Legacy Feature Migration Plan

본 문서는 `use_postprocess_ui` (Legacy Tkinter)의 모든 기능을 PySide6 기반의 `plate_by_markers_v2.py` (V2 Dashboard)로 이식하고, 현재 발생하고 있는 `TypeError` 및 데이터 연동 문제를 해결하기 위한 상세 실행 계획입니다.

## 1. 목표 (Objectives)
- **Halt Prevention**: 시뮬레이션 종료 후 UI가 멈추지 않고 즉시 관리 센터 또는 대시보드 실행.
- **Robust Bootstrapping**: `DropSimResult` 데이터를 `PlateAssemblyManager` 구조로 즉시 변환하여 시각화 창 실행.
- **Feature Parity**: 구형 UI의 3대 핵심 기능(기구학, 구조 해석, 2D 컨투어)을 V2의 고정밀 3D 뷰와 통합.
- **UX Alignment**: 사용자가 `use_postprocess_v2=True` 설정 시 시뮬레이션 종료 후 자동으로 결과 분석 창이 열리도록 개선.

## 2. 주요 작업 내역 (Tasks)

### Task 2.1: `plate_by_markers_v2.py` 오류 수정 및 부트스트래핑 구현
- **Fix `TypeError`**: `QtVisualizerV2.__init__` 호출 시 `manager` 인자가 누락되는 문제 해결.
- **Data Conversion**: 시뮬레이션 결과 파일(.pkl, `DropSimResult`)을 읽어 `PlateAssemblyManager`를 자동 생성하는 로직 추가.
- **Entry Point Refinement**: `if __name__ == "__main__":` 블록에서 `argparse`를 통해 전달된 경로의 데이터를 파싱하고 UI를 구동.

### Task 2.2: `whts_postprocess_ui_v2.py` (Control Center) 연동 강화
- **Auto-Launch Logic**: `--load` 인자를 받았을 때, 관리 센터 창을 띄우는 대신 (또는 동시에) 3D 대시보드를 즉시 실행하도록 `__main__` 수정.
- **Result Path Propagation**: 시뮬레이션 엔진에서 전달한 경로가 대시보드로 유실 없이 전달되도록 보장.

### Task 2.3: `use_postprocess_ui` 기능 이식 (Feature Migration)
- **Kinematics Tab**:
    - 8개 코너 및 CoM/Center의 변위, 속도, 가속도 데이터를 Matplotlib 그리드(4x1 또는 2x2)로 시각화하는 기능 추가.
    - 좌표계(Global/Local) 전환 기능 이식.
- **Structural Tab**:
    - PBA(Principal Bending Axis), RRG, Von-Mises stress 등의 시간 이력 그래프 추가.
    - 임계 시점(Critical Timestamps) 수직 점선 표시 기능 통합.
- **Sync & Animation**:
    - 3D View와 2D Plot 간의 시간 동기화(Time Scrubber) 로직 고도화.
    - 애니메이션 속도 조절 및 Play/Pause 제어 기능 강화.

### Task 2.4: 안정화 및 검증
- **Encoding Safety**: 파일 경로 및 한글 출력 시 인코딩 문제 방지 로직 적용.
- **MuJoCo Compatibility**: 시뮬레이션의 이산 블록(Discrete Blocks) 데이터가 SSR 엔진에서 누락 없이 처리되는지 확인.

## 3. 예상 변경 파일 (Target Files)
1. `run_drop_simulator/plate_by_markers_v2.py`: 메인 대시보드 로직 및 데이터 변환.
2. `run_drop_simulator/whts_postprocess_ui_v2.py`: 관리 센터 자동 실행 로직.
3. `run_drop_simulator/whts_engine.py`: 시뮬레이션 종료 시 호출 방식 최적화 (필요시).

## 4. 일정 및 체크리스트
- [ ] `plate_by_markers_v2.py` 수정 완료 및 실행 테스트
- [ ] 관리 센터 자동 분석 트리거 기능 확인
- [ ] 기구학/구조해석 탭 기능 정상 작동 여부 검증 (Case 2 실행)

---
> [!IMPORTANT]
> 기존 Tkinter UI는 `DropSimulator` 인스턴스를 직접 참조했으나, V2는 독립 프로세스로 실행되므로 **Pickle 데이터 기반의 상태 복원(Re-hydration)**이 이 작업의 핵심입니다.
