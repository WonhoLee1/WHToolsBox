# [PLAN] 통합 안정화 및 시각적 고도화 (Premium Visuals) - 2026-04-13

안녕하세요, **WHTOOLS**입니다. 

사용자님의 요청에 따라 최신 커밋(`46176ab`)을 베이스라인으로 확정하고, 그 위에 누락된 안정화 로직들과 요청하신 **'프리미엄 시각 에셋'**을 통합하여 완벽한 상태로 빌드하겠습니다.

## 🎯 목표
1. **시각적 완성도**: MuJoCo 시뮬레이션 환경을 단순한 격자 무늬에서 **'Premium Dark Studio'** 스타일로 업그레이드
2. **수치적 안정성**: SVD 비수렴, Friction 설정 오류, 내보내기 중단 이슈 등을 완벽히 해결한 상태로 병합
3. **호환성 유지**: `v5.py` 등 기존 테스트 스크립트가 정상 동작하도록 `contact_` 접두사 체계 복구

## 🛠️ 세부 변경 계획

### 1. [Component] MuJoCo Builder (`whtb_builder.py`)
- **[Visual]** 에셋 고도화:
    - Shadow Size: `1024` → `4096`
    - Skybox: 다크 그레이 그라데이션 적용 (`0.1 0.1 0.12` ~ `0.3 0.3 0.35`)
    - Ground: 체커보드 제거 후 **세련된 그리드(Grid)** 및 **반사도(Reflectance="0.1")** 적용
    - Lighting: `ambient="0.4 0.4 0.4"` 등으로 파트 입체감 개선
- **[Compatibility]** `contact_` 접두사 복구: `v5.py` 호환성 확보

### 2. [Component] Post-Processing Engine & Exporter
- **`whts_exporter.py`**: 해석 실패 파트 스킵 로직(KeyError 방지) 재적용
- **`whts_multipostprocessor_ui.py`**: 해석 성공한 첫 번째 파트를 탐색하여 필드 자동 구성
- **`whts_multipostprocessor_engine.py`**: SVD 수렴 안정화 및 가중치 정규화 보완

### 3. [Component] Configuration Library (`whtb_config.py`)
- `get_friction_standard` 및 관련 `Union/Tuple` 임포트 상태 최종 확인 및 보전

## 🧪 검증 계획
- `run_drop_simulation_cases_v5.py` 실행을 통해 다음 사항 확인:
    1. **Visual**: 무조코 실행 시 다크 스튜디오 환경이 정상적으로 출력되는지 확인
    2. **Stability**: 해석 중단 없이 Case 1, 2가 완료되는지 확인
    3. **Export**: 결과 데이터 및 UI 대시보드가 정상적으로 구성되는지 확인
