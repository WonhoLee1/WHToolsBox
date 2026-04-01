# WHTOOLS Simulation Final Report Layout Optimization

## User Review Required
> [!IMPORTANT]
> - 테이블의 열 폭을 기존 22에서 콘텐츠 길이에 맞게 조정(약 24 예상)하여 헤더와 내용이 완벽하게 정렬되도록 합니다.
> - 지표 설명(Legend)은 보고서의 하단(끝선 아래)에 추가됩니다.

## Proposed Changes

### `run_drop_simulator/whts_reporting.py`

#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)
- `finalize_simulation_results` 함수 내에서:
  - `col_width` 변수 값 조정.
  - `_fmt` 헬퍼 함수의 포맷팅 문자열을 `col_width`에 맞춰 수정하여 중앙 정렬 또는 정밀 우측 정렬 구현.
  - 테이블 출력 루프가 끝난 후 다음 내용 추가:
    ```python
    print("-" * total_w)
    print(" [ Metrics Legend ]")
    print(" - Bend  : Principal Bending (Tilt) Angle [deg]")
    print(" - Twist : Torsional (Twist) Angle [deg]")
    print(" - BS    : Max Bending Stress calculated from internal moments [MPa]")
    print(" - RRG   : Rotational Rigidity Gradient (Relative rotation between adjacent blocks) [deg]")
    print("=" * total_w + "\n")
    ```

## Verification Plan

### Automated Tests
- `run_drop_simulation_cases_v4.py`를 실행하여 터미널에 출력되는 최종 리포트의 레이아웃 확인. (사용자의 터미널 덤프와 비교)

### Manual Verification
- 출력된 텍스트가 깨지지 않고 열이 잘 맞는지 육안으로 확인.
- 하단 설명 문구가 정해진 위치에 올바르게 출력되는지 확인.
