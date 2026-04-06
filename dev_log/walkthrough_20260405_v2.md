# Walkthrough - 2026-04-05 UI Optimization (v1.2)

본 문서에서는 `QtVisualizerV2`의 UI 리팩토링 및 2D 플롯 엔진 최적화 작업에 대한 상세 내용을 설명합니다.

## 1. 개요 및 주요 변경 사항

이번 업데이트의 핵심은 **성능 최적화**와 **사용자 인터랙션 강화**입니다. 기존에 레이아웃 변경 시 발생하던 리소스 재생성 부하를 줄이고, 전문가용 분석 도구에 걸맞은 시각적 피드백을 추가했습니다.

### 1.1. 2D 플롯 엔진 최적화 (Phase 1)

- **Persistent FigureCanvas**: 레이아웃이 변경될 때마다 `FigureCanvas`를 다시 생성하지 않고, 기존 캔버스의 `Figure`만 `clear()` 후 서브플롯을 다시 배치하도록 수정했습니다.
- **레이아웃 전환 속도 개선**: 위젯 파괴/생성 과정이 생략되어 3x2 등 복잡한 레이아웃으로의 전환이 비약적으로 빨라졌습니다.

### 1.2. Pop-out 미러링 및 실시간 동기화 (Phase 2)

- **Mirror View**: Pop-out 창을 띄울 때 메인 윈도우의 현재 레이아웃과 데이터 구성을 그대로 복제합니다.
- **애니메이션 동기화**: 메인 윈도우의 타임 슬라이더나 애니메이션 재생 시 Pop-out 창의 그래프들도 실시간으로 함께 업데이트됩니다.

### 1.3. 축 선택 인터랙션 (Phase 3)

- **Visual Highlight**: 활성 슬롯(Selected Plot)의 테두리를 `WHTOOLS Blue (#1A73E8)` 색상과 `3px` 두께로 강조하여 현재 작업 대상을 명확히 알 수 있습니다.
- **배경색 반전**: 선택된 슬롯은 연한 하늘색 배경(`f0f7ff`)으로 처리되어 시인성을 높였습니다.

### 1.4. 브랜드 통합 및 스타일링 (Phase 4)

- **로고 레이아웃 최적화**: 로고 배너를 `3D View Control` 그룹박스 **왼쪽 외부**에 배치하여 시각적인 밸런스를 조정했습니다.
- **초기 환영 메시지**: 프로그램 구동 시 하단 상태 표시줄(Status Bar)에 "Hello!" 메시지를 출력하여 성공적인 로딩을 알립니다.
- **전역 폰트 적용**: `f_font_size`와 `v_font_size` 설정을 Matplotlib의 모든 텍스트 원소(Title, Label, Ticks, Legend)에 완벽히 전파했습니다.

## 2. 주요 코드 변경 설명

### 2.1. `update_frame`의 동기화 로직

애니메이션 프레임 업데이트 시 `self.pop_win`의 가시성 여부를 체크하여 미러링 창을 갱신합니다.

```python
def update_frame(self, f):
    # ... 메인 뷰 업데이트 ...
    if self.pop_win and self.pop_win.isVisible():
        self._update_pop_out_plots(f)
```

### 2.2. 효율적인 레이아웃 관리 (`_init_2d_plots`)

캔버스를 지우고 새로 그리는 대신 `figure.clear()`를 사용하여 리소스를 재사용합니다.

```python
def _init_2d_plots(self):
    self.canv.figure.clear()
    # ... 하위 subplot 생성 ...
    self.canv.draw_idle()
```

## 3. 검증 결과

- **레이아웃 전환**: 1x1에서 3x2까지 즉각적으로 전환됨을 확인했습니다.
- **Pop-out 동기화**: 애니메이션 재생 중 팝업 창의 그래프와 타임라인이 메인 창과 완벽히 일치함을 확인했습니다.
- **상태 표시줄**: 앱 초기 로드 시 "Hello!" 메시지가 정상적으로 표시됩니다.
