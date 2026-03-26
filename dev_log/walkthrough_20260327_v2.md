# Walkthrough: MuJoCo 시뮬레이션 제어 시스템 고도화

본 가이드는 `run_drop_simulation_v3.py`를 기반으로 한 시뮬레이션 제어 시스템의 기능 확장 및 UI 고도화 작업 결과를 요약합니다.

## 1. 주요 업데이트 내역

### 1.1. Config 제어 UI 현대화
- **항목 그룹화**: 환경(Environment), 물리(Physics), 구조(Structure), 쿠션(Cushion) 카테고리로 설정을 분류하여 복잡한 파라미터를 체계적으로 관리할 수 있습니다.
- **고급 제어 패널**:
    - `Step Size` 슬라이더 (1~100) 추가.
    - `Step Forward / Backward` 버튼 추가 (슬라이더 수치만큼 이동).
    - `Play / Stop` 버튼 추가 (무조코 뷰어의 Space바와 상호 동기화).
- **사용 편의성**:
    - 가로/세로 스크롤바 및 마우스 휠 스크롤 지원.
    - 설정 항목별 상세 설명 보강.

### 1.2. 시뮬레이션 엔진 고도화
- **Step Backward 지원**: 최근 500개 스텝의 상태를 버퍼링하여 시뮬레이션 시간을 뒤로 되돌리는 기능을 구현했습니다. (키보드 좌측 화살표 연동)
- **시각화 강화**: 무조코 뷰어 상단에 실시간 시뮬레이션 시간이 상시 표시됩니다.
- **카메라 뷰 정보 추출**: `Print View Info` 버튼 클릭 시 현재 카메라의 `LookAt`, `Distance`, `Azimuth` 등의 좌표를 터미널에 출력하여 코드에 즉시 활용할 수 있습니다.

### 1.3. 종료 시퀀스 최적화
- 시뮬레이션 시간이 종료되어도 즉시 창이 닫히지 않고 일시정지 상태를 유지합니다.
- 포스트 프로세싱(결과 저장 및 플로팅) 완료 후, 터미널에서 사용자의 명시적인 종료 의사(`Y/n`)를 확인하는 안전장치를 추가했습니다.

## 2. 주요 코드 스니펫

### 상태 버퍼링 및 스텝 백 (DropSimulator)
```python
def _step_once(self):
    # 상태 저장 (Step Backward용)
    state = {
        'qpos': self.data.qpos.copy(),
        'qvel': self.data.qvel.copy(),
        'time': self.data.time
    }
    self.state_buffer.append(state)
    ...
```

### UI 그룹화 및 스크롤 (ConfigEditor)
```python
# 그룹별 항목 생성 예시
for group_name, keys in self.groups.items():
    g_frame = ttk.LabelFrame(self.scroll_frame, text=f" [{group_name}] ")
    g_frame.pack(fill="x", padx=5, pady=5)
    for k in keys:
        if k in self.sim.config:
            self.add_config_row(g_frame, k, self.sim.config[k])
```

## 3. 검증 결과
- **로깅 시스템**: 헤더 정합성 및 실시간 물성 지표(P, S, D) 출력 정상 확인.
- **UI 반응성**: Tkinter 이벤트 루프와 무조코 렌더링 간의 간섭 없는 비동기적 상호작용 확인.
- **물리 정합성**: 스텝 이동 시 히스토리 데이터와 물리 상태가 동기화되어 정확한 분석 데이터 생성 완료.
