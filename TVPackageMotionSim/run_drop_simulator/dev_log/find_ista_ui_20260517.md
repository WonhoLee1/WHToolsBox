# 🛠️ WHToolsBox - ISTA 관련 UI 구현 파일 탐색 로그 (2026-05-17)

## 1. 개요 및 목적
- 사용자 요청에 따라 `WHToolsBox` 하위 코드 중에서 **ISTA (International Safe Transit Association) 낙하 규격 시뮬레이션 설정과 관련된 UI를 제공하는 Python (.py) 파일**을 검색하고 분석합니다.

## 2. 탐색 결과
검색 결과, 핵심적으로 ISTA 모드 설정 UI를 포함하는 파일은 다음과 같습니다:

### 📦 **핵심 UI 파일**
- **경로:** `c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulator\whts_control_panel.py`
- **주요 UI 요소 및 역할:**
  - **`ModelSetupDialog` 클래스:** MuJoCo 모델 구성 및 물리 파라미터 통합 설정 대화상자.
  - **`self.combo_ista` (QComboBox):** "GENERAL", "PARCEL", "LTL" 모드를 지원하는 ISTA 모드 선택 드롭다운 UI.
    ```python
    # ISTA Mode
    setup_layout.addWidget(QtWidgets.QLabel("📦 ISTA Mode:"), 1, 0)
    self.combo_ista = QtWidgets.QComboBox()
    self.combo_ista.addItems(["GENERAL", "PARCEL", "LTL"])
    self.combo_ista.currentTextChanged.connect(self._on_ista_changed)
    setup_layout.addWidget(self.combo_ista, 1, 1)
    ```
  - **`_on_ista_changed(self, text)` 콜백 메서드:** 선택된 ISTA 모드에 따라 낙하 타겟 방향(`drop_direction`)을 자동 설정해 줍니다.
    - `PARCEL` 선택 시 -> `"Corner 2-3-5"`로 변경
    - `LTL` 선택 시 -> `"Face 3 (Bottom)"`으로 변경
    ```python
    def _on_ista_changed(self, text):
        self.config["drop_mode"] = text
        if text == "PARCEL":
            self.edit_direction.setText("Corner 2-3-5")
        elif text == "LTL":
            self.edit_direction.setText("Face 3 (Bottom)")
        # GENERAL은 그대로 유지
    ```

---

### 📐 **배경 로직 및 수학적 계산 파일 (참고)**
직접적인 Qt UI 창은 아니지만, 내부에서 ISTA 면 맵핑(Face Mapping)과 초기 낙하 자세 계산을 전담하여 위 UI에 연동되는 핵심 연산 파일입니다:
- **경로:** `c:\Users\GOODMAN\WHToolsBox\box_motion.py`
- **주요 구성 클래스:**
  - `IstaFaceMapper`: ISTA 규격 번호(1~6)와 박스 기하학 면(T, D, F, B, R, L)을 동적 매핑.
  - `ISTA6ASimulator`: 질량, 크기, 운송 수단 방식 등을 바탕으로 ISTA 6A 시험 타입(A~H) 및 낙하 높이를 판정하고, 시퀀스를 생성하는 시뮬레이터 로직 구현.
  - `show_pyvista()`: PyVista 3D 시각화 연동.

---

## 3. 결론 및 요약
`WHToolsBox` 프로젝트의 **낙하 시뮬레이션 제어 센터 UI**에서 ISTA 규격 낙하 시나리오를 설정하는 인터페이스 파일은 `whts_control_panel.py` 입니다.
해당 파일은 PySide6/PyQt 기반으로 작성되어 있으며, 사용자가 손쉽게 ISTA Drop Mode를 선택하고 그에 맞춰 낙하 파라미터가 실시간으로 조율되도록 구성되어 있습니다.
