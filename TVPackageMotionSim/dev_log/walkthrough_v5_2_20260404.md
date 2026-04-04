# [Walkthrough] V5.2 고도화 평판 이론(Kirchhoff, Mindlin, Von Karman) 비교 분석 대시보드

동일한 시뮬레이션 데이터에 대해 다양한 공학적 평판 이론을 적용하고 결과를 실시간으로 비교할 수 있는 **V5.2 고도화 분석 환경** 구축을 완료하였습니다.

## 주요 성과 및 변경 사항

### 1. 다중 이론 지원 JAX 솔버 (AdvancedPlateOptimizer)
- **Kirchhoff (기본)**: 기존의 박판 굽힘 이론을 유지.
- **Mindlin (전단 고려)**: Chassis 해석을 위해 **횡전단 응력(Transverse Shear Stress)** 계산 로직 통합. `shear_correction` 파라미터 노출.
- **Von Kármán (막 응력 고려)**: Open Cell 대변형 대응을 위해 **막 응력(Membrane Stress)** 항 추가. 변형 구배를 추적하여 '트램펄린 효과' 수치화.

### 2. 실시간 비교 분석 UI (Dynamic Theory Swapping)
- **Theory 선택 콤보박스**: PyVista 대시보드 상단에 이론 선택 메뉴 추가.
- **실시간 재해석**: 이론 변경 시 JAX 솔버가 해당 물리 법칙에 맞춰 전체 프레임을 **즉시 재해석**.
- **가변 필드 업데이트**: 이론별 특화 필드(`Membrane Stress`, `Shear Stress`) 동적 시각화 지원.

---

터미널에서 다음 명령을 실행하여 고도화된 이론 비교 대시보드를 시작하세요.

```powershell
python run_drop_simulation_cases_v5.py
```

---
**WHTOOLS** 올림.
