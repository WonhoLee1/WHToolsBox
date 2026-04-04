# Structural Analysis Dashboard: V5.5.0 전문화 완료 내역

이 문서는 Qt 기반 Structural Deformation Dashboard (V5.5.0)의 최종 추가/개선 사항을 요약합니다.

## ✨ 1. UI/UX 및 워크플로우 대폭 개선

### 1.1. 컨텍스트 메뉴 "Sticky" & "Grouping" 구현
> [!tip] 그룹 토글 기능
> `Opencell_Right`, `Opencell_Left` 등 수많은 서브 부품들이 이제 메인 이름표인 **`Opencell`** 그룹 아래로 묶여 출력됩니다.

*   컨텍스트 메뉴에서 여러 옵션을 연속으로 켜고 끌 때, 메뉴가 닫히지 않고 계속 유지되도록 (Sticky) 재귀적 메뉴 전시 기술을 적용했습니다. 
*   마커 라벨을 확대/축소할 수 있는 폰트 조절(Font Size +/-) 메뉴가 추가되었습니다.

### 1.2. 재생 컨트롤 패널 개선 & 통계 오버레이
*   **Play/Stop 토글:** 애니메이션 재생 중에는 버튼이 일시정지(`⏸`) 아이콘으로, 멈췄을 때는 부드럽게 재생(`▶`) 아이콘으로 자동 변환됩니다.
*   **Min/Max 오버레이:** 3D View의 좌측 상단에 현재 표시 중인 전체 메쉬를 기준으로 최댓값/최솟값을 계산하고, 이를 가진 파트 이름과 수치를 실시간으로 표시하는 데이터 패널을 추가했습니다.
*   **글로벌 폰트:** 전체 GUI에 `Cascadia Code`가 적용되었습니다. 이 폰트 설정은 코드의 `_init_ui()` 함수 최상단 `WHTS_FONT` 문자열 변수로 지정되어 있으므로 언제든 쉽게 변경할 수 있습니다.

### 1.3. V5.5.0 Isometric 뷰 확장
*   기존 단일 1방향 Isometric 뷰 대신 NE, NW, SE, SW 방위를 바라보는 **4대각 Isometric** 단축 뷰 메뉴가 추가되어, 모델을 입체적으로 돌려보기 쉬워졌습니다.

---

## 🛠 2. 물리 분석 엔진 (JAX-SSR) 확장 및 치명적 버그 수정

### 2.1. 대시보드 실행 오류 (Launch Crash) 해결
> [!important] 버그 리포트 및 조치 사항
> 1. **AttributeError (SetInput):** PyVista 버전 및 `CornerAnnotation` 객체 특성상 `SetInput()` 속성이 누락되어 발생하던 에러를, PyVista의 권장 방식인 `add_text(name='stat_overlay')` 덮어쓰기 방식으로 변경하여 해결했습니다.
> 2. **ValueError (Array Shape Mismatch):** `R` (3x3), `m_raw` (N_markers, 3) 등 행렬 데이터가 필드 콤보박스에 잘못 포함되어 625개의 메쉬 정점과 매핑될 때 발생하던 셰이프 불일치 에러(108 vs 36)를 필터링 로직 고도화로 원천 차단했습니다.

### 2.2. 고전 이론/응력장 추가
*   **형상 곡률:** `Curvature X, Y, XY` 등 형상의 물리적 굽힘률 필드가 추가되었습니다.
*   **전단 응력:** `Shear Stress XY` 와 판재/쉘 이론에 해당하는 `YZ, XZ` 두께 방향 전단장들이 JAX 해석 단계에서 계산되어 콤보 리스트에 추가되었습니다.
*   **막 응력:** 폰 칼만 이론 시, 굽힘만이 아닌 Membrane 응력(`Membrane Stress Y` 추가) 이 정확하게 개별 분할되어 표출됩니다.
*   **Signed Von-Mises:** 응력의 방향성을 알 수 있는 `Signed Von-Mises [MPa]` 필드를 추가하여 압축/인장 영역을 시각화할 수 있습니다.
