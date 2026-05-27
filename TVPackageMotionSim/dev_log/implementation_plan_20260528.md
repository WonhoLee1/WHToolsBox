# dev_log 통합 개발 역사 및 기술 명세 백서 편찬 계획

## 프로젝트 개요

`TVPackageMotionSim\dev_log` 디렉토리에는 2026년 3월부터 5월 말까지 수개월간의 격동의 시뮬레이터 개발 및 튜닝 역사가 **200개가 넘는 아티팩트와 마크다운 로그(Implementation Plan, Walkthrough, Task)**에 나뉘어 축적되어 있습니다.

본 계획은 이 분산된 수많은 로그들의 핵심 정보(버전 마일스톤, 수치 불안정성 디버깅, 공학 수학 알고리즘, 물리 계수 튜닝, 최신 모니터 UI 등)를 논리적이고 체계적인 구조로 분석하고 단일 백서 파일로 완벽하게 통합·정리하는 것을 목표로 합니다.

더불어, 통합이 완벽히 검증되고 나면 **기존의 자잘하고 오랜 기간 누적된 레거시 마크다운 로그 파일들을 안전하고 과감하게 삭제(정리)**하여 디렉토리 청결성을 확보하겠습니다.

## User Review Required

> [!IMPORTANT]
> - 통합 정리된 백서의 파일명을 **`TVPackageMotionSim\dev_log\comprehensive_dev_history.md`**로 결정할 것을 제안합니다. 이에 동의하시는지 확인해 주세요.
> - **기존 파일 안전 정리 범위**:
>   - **삭제 대상**: 옛날 날짜의 모든 `implementation_plan_*.md`, `task_*.md`, `walkthrough_*.md` 파일들과 `history_*.md`, `engineering_knowledge.md`, `issue_tracker.md` 등 (모두 새 백서에 완벽히 통합 편입됨).
>   - **보존 대상**: 이미지 자산(`*.png`), 스크립트(`*.py`), 최신 백서(`comprehensive_dev_history.md`), 현재 작업용 문서(`implementation_plan_20260528.md` 등)

## Proposed Changes

### [Component Name] [NEW] [comprehensive_dev_history.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/comprehensive_dev_history.md)

- `dev_log` 아래의 모든 누적 역사와 기술 명세를 종합한 단일 대형 백서를 작성합니다. 
- 구성 체계 제안:
  
  ```mermaid
  graph TD
      A[comprehensive_dev_history.md] --> B["I. WHToolsBox 소개 및 개요"]
      A --> C["II. 통합 개발 역사 및 버전 마일스톤 (V1 ~ V7+)"]
      A --> D["III. 핵심 공학 알고리즘 및 물리 엔진 기술 명세"]
      A --> E["IV. 트러블슈팅 및 버그 픽스 아카이브 (Gotchas)"]
      A --> F["V. 통합 데이터 추출기 & UI 뷰어 매뉴얼"]
  ```

#### 세부 구성 항목:

1. **I. WHToolsBox 소개 및 개요**:
   - 패키지 낙하 시뮬레이터 프레임워크의 개발 취지와 핵심 가치 정리.
2. **II. 통합 개발 역사 및 버전 마일스톤**:
   - 2026년 3월 초기 이산 요소(Cushion, Paper) 접촉 모델링 설계.
   - V4, V5, V6, V7+로 발전해 온 과정(LTL 모드, 코너 낙하, SVD, JAX 가속 통합 해석, ParaView VTKHDF/GLB 자동 내보내기).
3. **III. 핵심 공학 알고리즘 및 물리 엔진 기술 명세**:
   - 상대 회전 행렬 분해(Bending, Twist) 기구학 이론.
   - 정밀 공기 역학(항력, Squeeze Film 효과) 레이놀즈 방정식 근사.
   - 소성 변형(Strain-based Plasticity v3) 듀얼 트리거 및 영구 압착 수식.
   - 관성 텐서 자동 보정(Auto-balancing) Delta-Inertia 이론.
4. **IV. 트러블슈팅 및 버그 픽스 아카이브 (Gotchas)**:
   - 개발 과정에서 디버깅한 핵심 수치 불안정성 이슈(SVD NaN Guard, UI 프리징 해결, UTF-8 인코딩 등) 및 주의점 정리.
5. **V. 통합 데이터 추출기 & UI 뷰어 매뉴얼**:
   - 최근 추가된 `wht_export_sim_result.py`와 `wht_plotwindowutil.py`의 구조 및 실제 터미널 실행법 안내.

### [Component Name] [DELETE] [Legacy Markdown Logs]

- 백서 작성 및 검증이 완벽히 끝나면, `dev_log` 하위에 수백 개가 쌓여 있는 옛날 `implementation_plan_*.md`, `task_*.md`, `walkthrough_*.md`, `history_*.md`, `engineering_knowledge.md`, `issue_tracker.md` 등의 마크다운 자산을 일괄 영구 안전 삭제 처리하여 폴더를 정리합니다.

## Verification Plan

### Manual Verification
- 에디터를 통해 새로 편찬된 백서인 [comprehensive_dev_history.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/comprehensive_dev_history.md)를 **Preview mode**로 열어, 목차와 수식 렌더링, 텍스트가 인코딩(UTF-8) 깨짐 없이 완벽한 가시성을 보여주는지 심도 깊게 점검합니다.
- 삭제 완료 후 `dev_log/` 폴더를 조회하여, 이미지와 코드 및 백서 본문 파일만 완벽하게 필터링되어 깨끗하게 정리되었는지 검증합니다.
