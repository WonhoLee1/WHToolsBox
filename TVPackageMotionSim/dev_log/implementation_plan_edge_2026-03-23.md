# Implementation Plan - 쿠션 엣지 판별 로직 수정 (2026-03-23)

## 1. 개요
`BCushion` 클래스의 `is_edge_block` 메서드가 육면체의 4개 측면(Shell) 전체를 엣지로 오인하고 있는 현상을 수정합니다. 
사용자의 의도에 따라 8개 꼭짓점(Vertices)과 이를 잇는 깊이(Depth, Z) 방향의 4개 엣지만 `contact_bcushion_edge` 클래스가 적용되도록 변경합니다.

## 2. 문제 분석
- **현재 코드 (802-804행)**: `(i == 0 or i == nx - 1) or (j == 0 or j == ny - 1)`
    - 이는 $X$ 방향 끝면 또는 $Y$ 방향 끝면에 속하는 모든 블록을 선택합니다.
    - 결과적으로 육면체의 4개 수직면 전체가 엣지로 분류됩니다.
- **수정 방향**: `(i == 0 or i == nx - 1)` 이면서 `(j == 0 or j == ny - 1)` 인 블록만 선택
    - 이는 네 모서리의 수직 기둥(Z-Edges)에 해당하는 블록들만 선택하게 됩니다.
    - 이 기둥의 상단/하단 끝점이 곧 8개의 꼭짓점이 됩니다.

## 3. 수정 단계
### 3.1. `BCushion.is_edge_block` 수정
- `or` 연산자를 `and` 연산자로 변경합니다.

## 4. 검증 계획

### 4.1. 자동화 테스트 (Automated Tests)
- `/tmp/verify_edge_logic.py` 스크립트를 작성하여 `BCushion.is_edge_block`의 인덱스 선택 로직을 검증합니다.
- **실행 방법**: `python /tmp/verify_edge_logic.py`
- **통과 기준**: `nx=5, ny=4, nz=3` 설정 시 총 12개(4x3)의 블록만 엣지로 판별되어야 합니다.

### 4.2. 수동 검증 및 시각적 확인 (Manual Verification)
- `run_drop_simulation.py`를 실행하여 생성된 XML(`temp_drop_sim.xml`) 파일을 확인합니다.
- `g_bcushion_i_j_k` 지오메트리 중 `class="contact_bcushion_edge"`가 적용된 항목들이 `(0,0), (nx-1,0), (0,ny-1), (nx-1,ny-1)` 인덱스 조합에 대해서만 생성되는지 확인합니다.
- 예: `nx=5, ny=4`일 때, `(0,0,k), (4,0,k), (0,3,k), (4,3,k)`만 엣지 클래스를 가져야 합니다.

---
> [!TIP]
> 이 수정으로 인해 엣지에 특화된 물리 속성(stiffness, damping 등)이 보다 정확한 위치에 적용되어 시뮬레이션의 신뢰도가 향상될 것으로 기대됩니다.
