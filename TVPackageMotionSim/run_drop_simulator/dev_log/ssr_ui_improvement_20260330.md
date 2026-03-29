# [2026-03-30] SSR 분석기 물리 속성 자동 예측 및 UI 개선 (v4.4.1)

## 1. 개요

- Precision Stress Field Analyzer (SSR) 대화상자에서 부품별 쉘 두께(t) 및 탄성계수(E)를 자동 예측하여 입력하는 기능 추가.
- 예측 로직:
  - **두께 (t)**: 부품(geom)의 Depth(Z축 크기)로부터 자동 추출.
  - **탄성계수 (E)**: 부품의 Weld Solref(또는 Geom Solref)의 Timeconst로부터 역산.
- UI 개선: 컴포넌트별로 예측된 값을 확인하고 직접 수정할 수 있도록 리스트형 인터페이스 적용.

## 2. 세부 변경 사항

### 2.1. 속성 예측 엔진 (Internal Logic)

- `_predict_properties(comp_name)` 메서드 구현.
- MuJoCo `m.geom_size` (half-size)를 full thickness로 변환.
- `m.eq_solref` 또는 `m.geom_solref`를 사용하여 $k = 1/\tau^2$ 관계로부터 $E = kL/A$ 역산.

### 2.2. SSR 분석기 UI 고도화

- 부품 선택 영역을 단순 체크박스에서 `[체크] 부품명 [두께 입력] [영률 입력]` 행 구조로 변경.
- 부품이 많을 경우를 대비하여 스크롤 가능한 캔버스 영역 적용.
- 전체적인 대화상자 크기 최적화 (가로 확장).

### 2.3. 분석 프로세스 연동

- `_calc_worker` 수행 시 각 부품에 설정된 개별 `t`, `E` 값을 `config`에 반영하여 `compute_ssr_shell_metrics` 호출.

## 3. 적용 파일

- `whts_postprocess_ui.py`: `SSRAnalyzerDialog` 클래스 전면 개편.
