# 🚀 멀티 스레드 MuJoCo 물리 제어 콜백 안전성 패치 (BatchRdsWorker Parallel Bug Fix)

BatchRdsWorker를 이용한 배치 시뮬레이션에서 $N$개의 병렬 실행 시 $n+1 \dots n+n$ 번째 시나리오들이 에러(`Python exception raised` 등)와 함께 멈추는 현상을 해결했습니다.

## 📌 변경 사항 요약

### 1. `whts_engine.py` - 글로벌 스레드 디스패처 도입
> [!NOTE]
> `mujoco.set_mjcb_control()`은 프로세스 단위의 **전역 C-Level 함수 포인터**를 변경합니다. 따라서 여러 스레드가 동시에 시뮬레이션을 실행하면 덮어쓰기 경쟁(Race Condition)이 발생하고 다른 스레드의 콜백을 오염시킵니다.

- **`_mujoco_thread_registry` 딕셔너리 생성**: 스레드 ID (`threading.get_ident()`)를 키값으로 받아 각 스레드의 개별 인스턴스에 속한 `_physics_control_callback` 람다 함수를 캐싱합니다.
- **`_global_mujoco_control_callback` 구현**: 단일 전역 C-Level 콜백으로 동작하며, 진입 시 현재의 스레드 ID를 조회하여 해당 스레드의 시뮬레이터 인스턴스 제어 로직만 정확히 맵핑하여 호출합니다.
- `DropSimulator.setup()`에서 이제 이 글로벌 래퍼를 등록하도록 변경하고, 예외 발생 시 `set_mjcb_control(None)` 대신 레지스트리에서 스레드를 팝업(`pop`) 시켜 전역 콜백을 초기화해버리는 부작용을 제거했습니다.

### 2. `whts_control_panel.py` - 안전한 스레드 종료 처리
> [!TIP]
> 작업이 끝난 후 전역 싱글톤을 초기화해버리면 다른 진행 중인 병렬 스레드가 물리 연산 제어권을 잃어버리는 현상을 방지합니다.

- `BatchRdsWorker.run_one()` 실행 블록이 끝날 때, 무차별적인 `mujoco.set_mjcb_control(None)` 호출을 삭제하고, `whts_engine`으로부터 전역 레지스트리를 가져와 `_mujoco_thread_registry.pop(threading.get_ident(), None)`로 본인 스레드에 맵핑된 콜백만 안전하게 수거하도록 조치했습니다.

## 🛠 검증 방법
- `Control Panel` 실행 후, `ISTA-6 Amazon Setup Helper`를 통해 시나리오들을 다량 생성합니다.
- `Parallel workers`의 숫자를 2~8개 등으로 설정하고 배치 작업을 실행합니다.
- 모든 $N$ 차수 큐들이 중단이나 물리 효과(Aero 등) 상실 없이 원활하게 병렬로 실행 및 저장되는지 확인합니다.
