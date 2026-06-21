# WHTOOLS TV Drop Motion Simulator v6.0 설명서 PPTX 빌드 워크스루

본 문서에서는 **WHTOOLS TV Drop Motion Simulator v6.0**의 프로젝트 상세설명 및 사용 설명서를 라이트 테마 기반의 PPTX 파일로 성공적으로 생성한 과정을 정리합니다.

## 변경된 주요 내용

### 1. 신규 스크립트 작성
- **파일명**: [make_manual_pptx.py](file:///c:/Users/GOODMAN/WHToolsBox/make_manual_pptx.py)
- **디자인 컨셉**: 화이트 및 밝은 블루그레이 배경(`C_BG`, `C_CARD_BG`, `C_ACCENT`)에 진한 로열 블루(`C_PRIMARY`)와 스카이 블루(`C_SECONDARY`)를 조합한 고품질 라이트 테마. 본문은 가독성이 뛰어난 다크 차콜 그레이(`C_TEXT`) 적용.
- **슬라이드 구성 (총 11장)**:
  - **Slide 1**: 타이틀 (WHTools TV Drop Motion Simulator v6.0 설명서)
  - **Slide 2**: 개요 및 목적 (물리 시험 한계, WHTools 솔루션 및 기대 가치)
  - **Slide 3**: 시스템 아키텍처 및 데이터 흐름 (4개 레이어 구조 및 파이프라인 흐름)
  - **Slide 4**: 핵심 기능 1 - Simulation Control Center (GUI 제어 패널, 파라미터 및 외부 도구 관리)
  - **Slide 5**: 핵심 기능 2 - JAX 기반 자율 구조 해석 (Kirchhoff 박판 솔버, Von-Mises 응력장, PCA 자율 정렬)
  - **Slide 6**: 핵심 기능 3 - ISTA-6 Amazon 배치 해석 (병렬 실행, 결과 수집 및 리포팅 자동화)
  - **Slide 7**: 사용법 1 - 프로그램 실행 및 기본 설정 (Miniconda 가상환경 활성화, INI 설정 및 파이썬 진입점 실행)
  - **Slide 8**: 사용법 2 - 모델 설정 및 물리 파라미터 튜닝 (Config Tree 조작, 관성 보정 및 Weld Blocks 강도 설정)
  - **Slide 9**: 사용법 3 - OpenRadioss FEM 해석 및 후처리 (최대 충격 스냅샷 캡처, Penetration 보정, VTKHDF/GLB 익스포트)
  - **Slide 10**: 주요 성과 및 차별성 (상용 FEM 솔버와의 4개 항목 정량/정성 비교 테이블)
  - **Slide 11**: 마무리 및 Q&A (연락처 및 상세 기술 스택 요약)

### 2. 레이아웃 개선 및 텍스트 겹침 버그 해결 (2026-06-13 추가)
- **카드 박스 내 겹침 개선**:
  - 카드당 단 하나의 큰 텍스트 박스만 생성하고, 내부의 타이틀 및 불릿 목록 항목들은 모두 텍스트 프레임 내 문단(`tf.add_paragraph()`)으로 추가하도록 `add_card_box` 함수를 리팩토링하여 텍스트 겹침 버그 해결.
- **일반 텍스트 상자 개행 대응**:
  - `add_text` 함수 내에서도 텍스트에 `\n` 개행문자가 포함된 경우, 기존에 단일 `run.text`에 통째로 기입하던 방식에서 `text.split('\n')`을 통해 개별 문단(`add_paragraph()`)으로 분산 배치하도록 로직 보강. 이로써 다중 행 텍스트의 겹침 현상을 원천 차단함.
- **Segoe UI 서체 통일**:
  - 모든 텍스트 렌더링 서체를 한글과 영문 모두에서 가장 널리 쓰이고 수려한 디자인을 보여주는 **Segoe UI**로 전면 교체하여 레이아웃의 시각적 완성도 극대화.

### 3. 빌드 결과물 생성
- **결과물 파일**: `WHTools_DropSimulator_Manual.pptx`
- **저장 경로**: [WHTools_DropSimulator_Manual.pptx](file:///c:/Users/GOODMAN/WHToolsBox/WHTools_DropSimulator_Manual.pptx)

---

## 검증 결과 및 확인 방법

### 자동 실행 검증
1. 콘다 가상환경(`vdmc`)에서 `python-pptx` 모듈을 정상적으로 인식하여 PPTX 빌드 완료.
2. 실행 명령어 및 결과 로그:
   ```powershell
   conda run -n vdmc python make_manual_pptx.py
   ```
   **출력 로그**:
   ```
   [SUCCESS] Saved PPTX manual to -> C:\Users\GOODMAN\WHToolsBox\WHTools_DropSimulator_Manual.pptx
   ```

### 수동 확인 방법
- 생성된 [WHTools_DropSimulator_Manual.pptx](file:///c:/Users/GOODMAN/WHToolsBox/WHTools_DropSimulator_Manual.pptx) 파일을 열어 다음 사항을 최종 검토해 주십시오:
  1. 슬라이드가 와이드(16:9) 비율로 설정되었는지 확인
  2. 밝은 블루와 화이트를 활용한 라이트 테마 배색이 정상 반영되었는지 확인
  3. 10번 슬라이드의 3열 비교 테이블이 레이아웃 깨짐 없이 올바르게 그려졌는지 확인
