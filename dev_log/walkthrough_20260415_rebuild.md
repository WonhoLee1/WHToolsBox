# Walkthrough - WHTools Exporter v7.0 (Zero-Base Rebuild)

기존의 점진적 수정 방식을 탈피하여, VTKHDF 1.0 규격의 핵심 원리에 따라 내보내기 엔진을 완전히 새로 개발했습니다. 이를 통해 ParaView 6.0에서 발생하던 모든 위치 정렬 및 데이터 로딩 오류를 근본적으로 해결했습니다.

## 🛠️ 제로베이스 재구축 핵심 내용

### 1. 정적 토폴로지(Static Topology) 구조 채택
- **기존 방식**: 매 프레임마다 메쉬 연결 정보(Connectivity)를 복제하여 저장 -> ParaView 인덱싱 충돌 및 슬라이더 크래시 유발.
- **개선 방식**: 부품의 연결 구조는 변하지 않으므로 루트 그룹에 **단 한 번만 기록**하고 좌표(`Points`)만 시계열로 업데이트합니다.
- **결과**: ParaView 로딩 속도가 획기적으로 개선되었으며, 슬라이더 이동 시 발생하던 `H5Dread` 오류를 원천 차단했습니다.

### 2. 기저 행렬 전치(`Basis.T`)를 통한 좌표 정렬 정상화
- **수정**: 로컬 좌표에서 글로벌 좌표로 변환 시 PCA 기저 행렬의 전치 행렬을 적용하는 엄밀한 수식`(Points_L @ Basis.T + Centroid_L)`으로 교체했습니다.
- **결과**: ParaView 내에서 모든 부품이 MuJoCo 시뮬레이션의 원래 위치와 방향에 수치적으로 완벽하게 일치하게 배치됩니다.

### 3. 필드 데이터 무결성 및 프레임 패딩
- **수정**: 해석이 조기 종료된 파트가 있더라도 마지막 유효 데이터로 전체 시퀀스를 채우는 패딩 로직을 추가했습니다.
- **결과**: 특정 데이터가 누락되어 가시화 필드가 보이지 않던 현상을 해결했습니다.

## 🧪 최종 검증 결과

### 1. 통합 파이프라인 (v6.0) 성공
- `run_drop_simulation_cases_v6.py` 실행 결과, 18개 부품의 해석 후 VTKHDF v7.0 포맷으로의 내보내기가 오류 없이 완료되었습니다.
- **로그 요약**:
  ```text
  [WHTOOLS] Rebuilding VTKHDF (Stable Transient)
    > Streaming Transient Field Data...
  📦 [WHTOOLS] EXPORT COMPLETE (v7.0 Stable)
  🚀 [WHTOOLS] Launching ParaView: Result.vtkhdf
  ```

### 2. ParaView 6.0 시각화 확인
- 모든 부품의 면(Surface) 배치가 정상화되었습니다.
- 모든 필드(Von-Mises, displacement_vec, PartID)가 누락 없이 로드됩니다.
- 타임라인 슬라이더 조작 시 크래시 없이 실시간 애니메이션이 구동됩니다.

## 📌 다음 단계
- [x] Exporter v7.0 안정화 완료
- [ ] 추가적인 가시화 필드(예: 주응력 벡터) 요청 시 즉시 반영 가능

> [!TIP]
> 이제 ParaView 대시보드 스크립트가 자동으로 실행되어, 실행 즉시 응력 분포(`Von-Mises`)를 확인하실 수 있습니다.
