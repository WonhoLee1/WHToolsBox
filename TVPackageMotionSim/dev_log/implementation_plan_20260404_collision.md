# [Plan] Paper Box 충돌 매트릭스 최적화 (Collision Mask Optimization)

현재 `BPaperBox` 활성화 시 성능이 급감하는 원인으로 **불필요한 충돌 쌍(Collision Pairs) 생성**이 지목되었습니다. 특히 이산화된 종이 박스 블록들이 내부의 모든 부품(OpenCell, Chassis 등)과 충돌 가능성을 계산하고 있어 연산 부하가 가중되고 있습니다.

## Proposed Changes

### 1. `run_discrete_builder/whtb_builder.py` [MODIFY]

충돌 비트마스크(conType/conAffinity) 로직을 재설계하여 물리적으로 유의미한 접촉만 허용합니다.

**기존 로직 (Wide Scope):**
- **PaperBox (1)**: Cushion(2) + OpenCell(4) + Tape(8) + Chassis(16) 와 충돌
- **Internal (4,8,16)**: PaperBox(1) + Cushion(2) 와 충돌

**변경 로직 (Isolated Scope):**
- **PaperBox (1)**: **Ground(1)** 및 **Cushion(2)** 과만 충돌
- **Cushion (2)**: 모든 부품(1, 4, 8, 16) + Ground(1) 와 충돌
- **Internal (4,8,16)**: **Cushion(2)** 과만 충돌 (박스와의 무의미한 내부 충돌 제거)

#### 세부 비트마스크 계획 (Decomposition):
- `bit_paper (1)`: conAffinity = `bit_cushion | 1` (Cushion + Ground) = **3**
- `bit_cushion (2)`: conAffinity = `all_bits | 1` (All + Ground) = **31**
- `bit_oc (4)` / `bit_occ (8)` / `bit_chassis (16)`: conAffinity = `bit_cushion` = **2**

## User Review Required

> [!IMPORTANT]
> **설계 의도 확인**: 박스 내부의 제품(OpenCell/Chassis)이 박스 내벽과 직접 닿는 극한의 상황(완충재를 뚫고 지나가는 경우)은 시뮬레이션에서 '비물리적 오류'로 간주하고 무시해도 되는지 확인 부탁드립니다. 이 최적화는 완충재가 박스와 제품 사이의 물리적 장벽 역할을 완벽히 수행한다고 가정합니다.

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v5.py`에서 `box_div`를 활성화한 상태로 FPS를 측정합니다.
- 최적화 전/후의 FPS 및 충돌 연산 시간(MuJoCo Profiler 활용 가능 시)을 비교합니다.

### Manual Verification
- MuJoCo 뷰어에서 `BPaperBox` 내부로 제품이 뚫고 나가는 현상이 발생하는지, 혹은 충합(Overlap) 시에 연산 오류가 발생하는지 시각적으로 검토합니다.
