# UI Enhancement & English Standardization Plan (v8-Pro+)

사용자의 요청에 따라 누락된 플레이백 컨트롤 버튼을 복원하고, 모든 UI와 소스 코드 문서(Docstrings/Comments)를 **영문**으로 전환하여 글로벌 표준에 부합하는 분석 도구를 완성합니다.

## User Review Required

> [!IMPORTANT]
> - **플레이백 버튼 추가**: `First (<<)`, `Prev (<)`, `Play/Pause (▶)`, `Next (>)`, `Last (>>)` 버튼이 슬라이더 왼편에 배치됩니다.
> - **언어 전환**: 모든 UI 레이블, 콤보박스 항목, 차트 제목, 단위 표시, 그리고 **코드 내 모든 주석과 Docstrings**가 영문으로 변경됩니다.

## Proposed Changes

### [UI/UX Enhancement]

#### [MODIFY] [plate_by_markers.py](file:///c:/Users/GOODMAN/WHToolsBox/plate_by_markers.py)
1. **플레이백 컨트롤 패널 구현**: `QtVisualizer._init_ui` 내부에 버튼 그룹 생성.
   - `btn_first`, `btn_prev`, `btn_play`, `btn_next`, `btn_last` 추가 및 시그널 연결.
2. **영문 표준화 (L10n)**:
   - UI Labels: "좌측 차트" -> "Left Chart", "변위" -> "Displacement", 등.
   - Docstrings & Comments: 모든 한글 설명을 영문으로 번역하여 기술.
3. **레이아웃 미세 조정**: 버튼들이 추가됨에 따라 하단 컨트롤 바의 배치를 최적화합니다.

## Verification Plan

### Automated Tests
- `python plate_by_markers.py`를 실행하여 1000프레임 해석 후 UI가 영문으로 정상 표시되는지 확인.
- 모든 버튼(`<<`, `<`, `▶`, `>`, `>>`)이 프레임을 정확히 이동시키는지 확인.

### Manual Verification
- 슬라이더와 버튼 간의 동기화 상태 확인.
- 영문 오타 및 단위 표기(Standard SI units)의 정확성 검토.
