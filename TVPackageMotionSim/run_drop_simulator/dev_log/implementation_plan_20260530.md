# [Goal Description]

BatchRdsWorker를 통해 N개의 시나리오를 동시 병렬 실행할 때, `n+1 ~ n+n` 번째 작업들에서 에러가 발생하며 진행되지 않는 문제를 해결합니다.

문제의 원인은 `concurrent.futures.ThreadPoolExecutor`를 사용하여 멀티 스레드로 시뮬레이션을 병렬 실행할 때, MuJoCo의 물리 제어 콜백(`mujoco.set_mjcb_control`)이 C-level에서 공유되는 전역(Global) 싱글톤 함수 포인터이기 때문입니다. 
- 스레드 1(Task 1)이 끝나고 `mujoco.set_mjcb_control(None)`을 호출하여 콜백을 초기화해버리면, 동시에 아직 실행 중이던 스레드 2(Task 2)의 물리 제어 로직도 유실됩니다.
- 또는 이후 재사용된 스레드가 새로운 콜백을 덮어쓰면서 서로 다른 시뮬레이션의 인스턴스를 참조하게 되어 메모리 참조 에러 및 물리 엔진 충돌(`Python exception raised`)을 발생시킵니다.

## User Review Required

> [!IMPORTANT]
> 이 문제를 해결하기 위해, `mujoco.set_mjcb_control`에 직접 개별 인스턴스 메서드를 꽂는 대신, `whts_engine.py` 모듈 레벨에 **스레드 ID를 식별키로 하는 글로벌 디스패처(Dispatcher) 레지스트리**를 구현합니다. 
> 이 방식은 Python 런타임의 병렬 구조를 해치지 않으면서(Process 기반 전환 불필요) MuJoCo의 C 레벨 콜백 제약을 우회하여 완벽한 멀티스레드 안전성(Thread-Safety)을 제공합니다.

## Proposed Changes

### TVPackageMotionSim

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
1. 상단 모듈 레벨에 스레드 세이프티를 보장하는 전역 콜백 디스패처 로직 추가.
```python
import threading
_mujoco_thread_registry = {}

def _global_mujoco_control_callback(model, data):
    tid = threading.get_ident()
    cb = _mujoco_thread_registry.get(tid)
    if cb is not None:
        cb(model, data)
```
2. `DropSimulator.setup()` 로직 변경
- `mujoco.set_mjcb_control(None)` 대신 `_mujoco_thread_registry.pop(threading.get_ident(), None)`로 본인 스레드의 기존 콜백만 삭제하도록 변경.
- 콜백 등록 시 `_mujoco_thread_registry[threading.get_ident()] = self._mjcb_control` 등록.
- `mujoco.set_mjcb_control(_global_mujoco_control_callback)` 고정 할당.
- `except Exception:` 블록에서도 레지스트리에서 본인 스레드를 `pop`하도록 예외 처리.

#### [MODIFY] [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)
1. `BatchRdsWorker.run_one()` 내 시나리오 종료 로직 수정
- 기존의 `mujoco.set_mjcb_control(None)` 코드를 삭제.
- 대신, `whts_engine.py`로부터 `_mujoco_thread_registry`를 임포트하여 `_mujoco_thread_registry.pop(threading.get_ident(), None)`을 호출하여 안전하게 해당 스레드의 콜백만 해제하도록 수정.

## Verification Plan

### Manual Verification
- `whts_control_panel.py` 배치 시뮬레이션을 실행하여 `StructuralDynamicsDialog`를 오픈합니다.
- 복수의 시나리오(예: 8개 이상)를 선택하고 `Parallel workers` 옵션을 2~4로 맞추고 `🚀 Do It`을 클릭합니다.
- `n+1 ~ n+n` 작업들이 에러 없이 모두 끝까지 완료되는지, CSV 결과가 온전하게 산출되는지 로그와 결과 폴더를 검증합니다.
