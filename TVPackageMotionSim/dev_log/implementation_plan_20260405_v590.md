# [Goal] 설정 기반(Config-driven) 3D-2D 분리형 전문가 대시보드 개편 (v5.9.0)

유동적인 분석 환경을 위해 UI의 모든 초기 상태를 코드로 제어할 수 있는 `GuiConfig` 시스템을 도입하고, 3D와 2D 영역을 스플리터로 분리하는 대규모 구조 개편을 수행합니다.

## Proposed Changes

- **GuiConfig**: 3D 필드, 2D 레이아웃 및 개별 슬롯 플롯 설정을 코드로 제어 가능한 구조체 도입
- **Layout**: QSplitter (좌: 3D, 우: 2D) 분리, 상단 애니메이션 툴바, 하단 상태바 배치
- **Banner**: 3D 제어 패널(Group Box) 내 좌측 상단으로 이동
- **2D Plot Engine**: 1x1 ~ 3x2 동적 레이아웃 및 "Add Plot" 전용 다이얼로그 시스템
- **Animation Step**: 1~10 프레임 건너뛰기 기능 추가

## Verification Plan

- `run_post_only_v5.py`에서 `GuiConfig`를 통한 복합 레이아웃 초기화 동작 확인
- 레이아웃 축소/확대 시 플롯 데이터 유지 검증
- 애니메이션 스텝 및 2D 플롯 동시 업데이트 성능 확인
