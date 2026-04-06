# [WHTOOLS] ISTA 6-Amazon 규격 반영 및 좌표계 통합 수정 계획 (Rev.2)

본 계획은 `TVPackageMotionSim` 내의 모든 모듈이 동일한 좌표계 정의를 공유하고, ISTA 6-Amazon (Type G/H) 규제에 맞는 낙하 면 번호를 지원하도록 전체 로직을 동기화합니다.

## User Feedback Reflected

1. **코드 이관 및 운용**: `_D260406` 파일에서 수정 및 검증을 완료한 후, 최종 로직을 메인 모듈인 `whts_mapping.py`로 이관하여 프로젝트의 표준으로 확정합니다.
2. **자동 회전(Pose Initialization) 이해**: 
    - 빌더는 `target_pt`가 전역 Z축 하단(`[0, 0, -1]`)을 향하도록 박스를 회전시켜 배치합니다. 
    - 이 로직 덕분에 물리 모델 정의(`Y=Height`)와 무조코 뷰어상의 낙하 자세가 성공적으로 결합됩니다.

## Proposed Changes

### 1. [run_discrete_builder] 좌표계 및 낙하 로직 수정

#### [MODIFY] [whtb_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_utils.py)
- **ISTA 6-Amazon 규격 분리 정의**:
    - **Parcel (Type G)**: Face 1/2(Y), 3/4(X), **5/6(Z)**
    - **LTL (Type H)**: Face 1/2(Y), **3/4(Z)**, 5/6(X)
- **축 매핑 고정**: `Top/Bottom` = Y축, `Front/Rear` = Z축, `Sides` = X축.

### 2. [run_drop_simulator] 매핑 및 분석 로직 수정 및 이관

#### [MODIFY] [whts_mapping_D260406.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping_D260406.py)
- **`get_face_index_logic`**: `Top/Bottom`을 Y축(index 1)으로 매핑.
- **SVD 투영**: 법선 벡터 방향에 따른 평면(X-Z, X-Y, Y-Z) 보정.

#### [FINAL EXPORT] [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)
- `_D260406`에서 검증된 최상위 버전을 `whts_mapping.py`로 덮어쓰기하여 최종 교체.

---

## 작업 순서 (Task List)

1. [x] 구현 계획 승인 및 상세 검토
2. [ ] 대상 파일 백업 (`.bak` 생성)
3. [ ] `whtb_utils.py` 수정: ISTA 번호 체계 및 Y-Up 축 적용
4. [ ] `whts_mapping_D260406.py` 수정 및 시뮬레이션 결과(v5) 검증
5. [ ] **코드 이관**: `whts_mapping_D260406.py` -> `whts_mapping.py` (Overwrite)
6. [ ] 최종 리포트 및 대시보드(V2) 확인

## Verification Plan

### Automated Tests
- `python run_drop_simulation_cases_v5.py` 실행
- 로그에 출력되는 `target_pt` 좌표값이 박스 치수(`box_w, box_h, box_d`)와 축 대칭이 맞는지 확인

### Manual Verification
- V2 대시보드의 **Contour Plot** 방향이 실제 제품의 상/하/좌/우와 일치하는지 육안 검사
