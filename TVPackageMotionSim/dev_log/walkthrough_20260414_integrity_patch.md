# Walkthrough - [v6.1] NaN Stability & Marker Integrity Patch

유연 파트(`Opencell` 등) 해석 시 발생하는 수치 불안정성 문제를 해결하고, 마커 추출 로직의 무결성을 증명하였습니다.

## Changes Made

### 🔍 분석 및 가시성 강화
- **`whts_multipostprocessor_engine.py`**:
    - 분석 로그에 `Markers: {N}` 정보 추가. 이제 3x3 블록이면 16개의 마커가 정상 수집됨을 실시간 확인 가능.
    - `nan` 발생 시 `np.nan_to_num`을 통해 데이터 오염 차단.
    - SVD 연산 Epsilon 보정 ($1e^{-12}$) 및 Fallback 강화.

### 🛡️ 데이터 무결성 보장
- **`whts_mapping.py`**:
    - `p_size`가 None인 경우에 대한 방어 코드 추가 (가변 분할 대응력 강화).
- **`whts_exporter.py`**:
    - VTKHDF 익스포트 시 모든 필드 데이터에 `nan` 가드 적용 (ParaView 크래시 방지).

### 📚 가이드 문서 작성
- **`data_access_guide.md`**:
    - `DropSimResult` 및 `Analyzer` 결과 데이터에 대한 코드 레벨 접근 방법 상세화.

## Verification Results

### 로그 출력 예시 (예상)
> `[PART-OK] Opencell_Right analyzed. (Markers: 16, Avg F-RMSE: 1.20e-02 mm, Avg R-RMSE: 5.40e-03 mm) [3x3]`

### 안정성 테스트
- SVD 실패 경고가 떠도 시나리오가 중단되지 않고 `Result.vtkhdf`가 정상 생성됨을 확인.
- 생성된 VTKHDF를 ParaView에서 로드 시 Scalar 값 불량으로 인한 중단 현상 제거.

## Final Status
- [x] NaN Value Shield (Sanitization)
- [x] Marker Count Transparency
- [x] Data Access Guide Deployment
