# [WHTOOLS] Config Editor UI Overhaul Plan (Backup)

#> [!IMPORTANT]
> - **Continuous Action**: "토크형" 버튼(누르고 있는 동안 연속 동작)은 Tkinter의 `after()`를 활용하여 구현됩니다. 엔진의 `tk_root.update()` 오버헤드에 따라 속도가 조절될 수 있습니다.
> - **Shortcut Guide**: 무조코 기본 단축키(Space: Play/Pause, ESC: Quit 등)와 커스텀 단축키(K: Config, Arrow Keys: Step) 정보를 포함합니다. **배속 조절(0.1x ~ 4.0x)** 기능도 제어 페이지에 추가합니다.
> - **Restart vs Reset**: '재시작'은 설정을 다시 읽어오는 Reload 기능을, '초기화(Rewind)'는 현재 메모리의 t=0 시점으로 돌아가는 기능을 의미하도록 구분합니다.
> - **Conditional Apply**: 시뮬레이션이 진행 중(`step_idx > 0`)일 때 설정을 수정하면 재시작 여부를 묻는 확인 창(Yes/No)을 띄우고, 승인 시 반영 후 시뮬레이션을 다시 시작합니다.
> - **Post-Process Activation**: 시뮬레이션이 목표 시간에 도달하면 사이드바의 `[결과 분석]` 버튼을 활성화(색상 강조)하고, 재시작 시에는 비활성화(Gray-out)합니다.
