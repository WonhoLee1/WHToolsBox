# Walkthrough - WHTS Engine Interactive Refinement (2026-04-23)

본 문서는 낙하 시뮬레이션 엔진의 상호작용성 강화 작업을 위한 주요 변경 사항과 사용 방법을 설명합니다.

## 1. 개요 (Overview)

기존의 정적인 시뮬레이션 환경을 **'인터랙티브 분석 환경'**으로 전환하였습니다. 사용자는 시뮬레이션 도중 실시간으로 과거 시점으로 되돌아가거나(Rewind), 특정 시점부터 다시 데이터를 기록(Record)하고, 정밀 분석을 위해 슬로우 모션(Slow-Motion)을 적용할 수 있습니다.

## 2. 주요 변경 사항 (Key Changes)

### 2.1 WHTS Engine (`whts_engine.py`)
- **스냅샷 기반 리셋**: `R` 키를 통해 초기 상태(`snapshots[0]`)로 즉시 복구하는 기능을 추가했습니다.
- **슬로우 모션 (Slow-Motion)**: `S` 키 토글을 통해 시뮬레이션 속도를 0.2배로 늦출 수 있습니다 (접촉 직전 정밀 분석용).
- **인터랙티브 레코딩**: 타겟 시간(`sim_duration`) 이후에도 `L` 키를 눌러 데이터를 계속 누적할 수 있습니다.
- **리포트 UI 고도화**: 터미널 리포트에 `REC` 상태와 `SLOW` 모드 표시등을 추가하여 현재 상태를 직관적으로 파악할 수 있게 했습니다.

### 2.2 Control Panel (`whts_control_panel.py`)
- **기능 버튼 추가**: `Reset`, `Slow Motion`, `Record History` 버튼을 추가하여 GUI에서도 엔진의 새로운 기능을 제어할 수 있습니다.
- **상태 동기화**: 엔진의 내부 상태(Pause, Slow, Rec)가 컨트롤 패널의 버튼 스타일과 텍스트에 실시간으로 반영됩니다.

## 3. 사용 방법 (How to Use)

### 단축키 가이드 (MuJoCo Viewer 활성 시)
- `Space`: 일시 정지 / 재개
- `Backspace`: 1단계 되감기 (Snapshot 기반)
- `R`: 전체 리셋 (초기 시점으로 복구)
- `S`: 슬로우 모션 토글 (0.2x <-> 1.0x)
- `L`: 데이터 기록 토글 (무한 모드에서 유용)
- `K`: 설정 편집기(Config Editor) 열기
- `ESC`: 시뮬레이션 종료 및 후처리 단계 진입

### 컨트롤 패널 활용
1. `Playback Controls`에서 `Reset` 버튼을 눌러 언제든 처음부터 다시 낙하를 시작할 수 있습니다.
2. `Interactive Effects` 그룹에서 `Slow Motion`을 켜고 `Pause`와 `Step` 기능을 조합하여 임팩트 순간을 정밀 분석하십시오.
3. `Record History`를 활성화한 상태에서 시뮬레이션을 진행하면, 원래 설정된 시간보다 더 긴 범위의 데이터를 후처리 UI에서 확인할 수 있습니다.

## 4. 기술적 세부 사항 (Technical Details)

- **Snapshot Management**: MuJoCo의 `mj_getState` 및 `mj_setState`를 활용하여 물리적 엄밀성을 유지하며 상태를 복구합니다.
- **History Truncation**: 과거 시점으로 Jump 시, 인과관계가 어긋나는 것을 방지하기 위해 해당 시점 이후의 히스토리 데이터를 자동으로 제거합니다.
- **Async Synchronization**: PySide6 UI와 MuJoCo 시뮬레이션 루프 간의 통신은 플래그 변수를 통해 비동기적으로 안전하게 처리됩니다.

---
*Created by Antigravity AI on 2026-04-23.*
