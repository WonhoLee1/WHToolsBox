# [Implementation Plan] 좌표계 표준화 롤백 및 빌더 복구 (2026-03-29)

사용자 요청에 따라 최근 수행한 좌표계 표준화(X=Width, Y=Depth, Z=Height) 작업을 취소하고, 이전의 공학적 관습(X=Width, Y=Height, Z=Depth)으로 복구합니다. 또한, 이전 편집 과정에서 손실된 `run_discrete_builder/__init__.py` 파일을 완전한 상태로 복구하겠습니다.

## User Review Required

> [!IMPORTANT]
> - **좌표계 체계 복구**: 모든 물리 및 기하학적 연산이 다시 `Z축 = Depth (전후 방향)`, `Y축 = Height (상하 방향)` 기준으로 변경됩니다.
> - **코드 복구**: 현재 손상되어 414줄만 남은 `run_discrete_builder/__init__.py` 파일을 이전의 1500줄 이상의 완전한 기능을 가진 상태로 재구성합니다.
> - **데이터 익스포트 기능**: 개별 창의 'Export' 메뉴 기능은 사용자 편의를 위해 **유지**하는 방향으로 진행하겠습니다. 만약 이 기능도 원치 않으시면 말씀해 주세요.

## Proposed Changes

### [TVPackageMotionSim]

---

#### [MODIFY] [run_discrete_builder/__init__.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/__init__.py)
- **BaseDiscreteBody 및 하위 클래스 복구**: 손실된 클래스 정의와 메서드(geometry build, XML string generation)를 모두 복원합니다.
- **좌표 매핑 엔진 수정**:
    - `is_cavity` (Tape/OCC): X-Y 평면 기준으로 수정.
    - `ltl_map`, `parcel_map` (in `parse_drop_target`): Y=H, Z=D 매핑으로 복구.
    - `Stacking Logic`: 부품 적층 축을 Y에서 Z로 변경.

#### [MODIFY] [postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/postprocess_ui.py)
- `LOCATION_LABELS` 주석 및 좌표 설명을 이전 상태(`Z=Depth`)로 복구합니다.
- 기구학 탭의 축 선택 로직을 확인하여 축 이름(Y/Z)과 물리적 의미(H/D)가 일치하도록 합니다.

#### [MODIFY] [run_drop_simulation_v3.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_v3.py)
- 코너 식별 로직에서 `Z`를 Depth로 인식하도록 필터링 기준을 복구합니다.
- `corner_body_ids` 정렬 순서를 이전 체계에 맞춰 조정합니다.

---

## Open Questions

- **Matplotlib Export Menu**: 이 기능은 현재 `mpl_extension.py`에 구현되어 있으며 좌표계와는 무관합니다. 그대로 두어도 괜찮을까요?

## Verification Plan

### Automated Tests
- `python run_discrete_builder/__init__.py` 실행을 통해 정상적인 XML이 생성되는지 확인.
- 생성된 XML의 `<body name="BPackagingBox">` 내 부품들의 `pos` 값이 Z축 정렬인지 확인.

### Manual Verification
- 시뮬레이션 실행 후 `PostProcessingUI`에서 코너 낙하 시 Target Point가 바닥(Z min/max)이 아닌 제품의 하단(Y min) 혹은 전면(Z min)으로 의도대로 찍히는지 확인.
- 'WHTOOLS: Export' 메뉴가 개별 창에서 여전히 잘 작동하는지 확인.
