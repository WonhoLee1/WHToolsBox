# WHTOOLS Dashboard UI Polish: JAX-SSR Accuracy Reporting (v5.9.9)

안녕하세요, **WHTOOLS**입니다.

박판 변형 해석의 신뢰도를 실시간으로 확인하실 수 있도록, **JAX-SSR 피팅 정확도(RMSE) 리포팅 기능**을 추가했습니다.

## 🛠️ 주요 변경 사항

### 1. JAX 기반 RMSE 정밀 연산
- **실시간 오차 분석**: `AdvancedPlateOptimizer` 코어에서 각 프레임별로 실제 마커 위치와 계산된 박판 표면 사이의 **RMSE(Root Mean Square Error)**를 JAX 병렬 연산으로 산출합니다.
- **수치적 무결성**: 단순한 피팅을 넘어, 각 부품이 물리적으로 얼마나 정밀하게 복원되었는지 수치(mm 단위)로 즉각 확인할 수 있습니다.

### 2. 터미널 분석 리포트 강화
- **부품별 개별 출력**: 전체 해석 과정에서 각 부품(Part)의 해석이 완료될 때마다 해당 부품의 **평균 RMSE(Avg RMSE)**를 터미널에 출력합니다.
- **가독성 개선**: 부품명과 오차 범위를 정렬된 텍스트 포맷으로 제공하여, 어떤 부품의 피팅 품질이 낮은지(예: 차수가 부족하거나 정규화가 강한 경우)를 한눈에 파악할 수 있습니다.

## 🧪 실행 확인 (Terminal Output)

```text
⏳ Running Plate Theory Structural Analysis for all parts...
[WHTOOLS] Assembly Analysis Started (18 parts)...
  > [PART] Cushion_Top               analyzed. (Avg RMSE: 4.8289e-03 mm)
  > [PART] OpenCell_Glass            analyzed. (Avg RMSE: 1.2541e-04 mm)
  ...
[WHTOOLS] All 18 Parts Analyzed Successfully.
```

> [!INFO]
> 일반적으로 **1e-03 mm** 이하의 RMSE는 매우 높은 피팅 정밀도를 의미합니다. 만약 특정 파트의 오차가 크다면 마커의 수가 부족하거나 해당 영역의 물리적 변동이 박판 이론의 범위를 벗어났을 가능성이 있습니다.

---
**WHTOOLS** 드림.
