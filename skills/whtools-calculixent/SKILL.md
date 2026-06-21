---
name: whtools-calculixent
description: WHTOOLs CalculiX & Python FEA multi-physics simulation agent skill. Extends capabilities in geometry, meshing, solver assembling, and result processing with full SQLite/Vector RAG support.
---

# 🛠️ WHTOOLs Calculixent - Integrated Agent Skill

본 파일은 **Antigravity** 에이전트 시스템 및 다양한 LLM 프레임워크가 유기적으로 가져다 쓸 수 있도록 패키징된 **WHTOOLs Calculixent** 공식 스킬 규약서입니다.

---

## 🧭 1. 에이전트 의사결정 및 검색 우선순위 (Agent Decision Tree)

AI 에이전트는 해석 모델을 설계하거나 연동 코드를 구축할 때, 다음의 **3단계 탐색 규칙**에 의거하여 논리적 사고를 전개합니다.

1. **LLM 내장 지식 우선 활용 (Fast-Path)**:
   * Abaqus/CalculiX의 핵심 키워드 표준 사양(`*MATERIAL`, `*ELASTIC`, `*BOUNDARY` 등)이나 수치해석 일반 공식(강성 조립, 형상 함수 등) 등은 내장 지식을 우선적으로 사용하여 고속 구현합니다.
2. **모호성 직면 시 RAG 및 예제 지식 서칭 (Search-Path)**:
   * 다중물리 해석의 특이점(예: 접촉 접합, 전자기 필드 세팅, FLUID 관망 설계 등)이나 해석 설정의 모호함이 나타나면 본 스킬의 `resources/calculix_rag.db` SQLite 데이터베이스나 `examples/` 폴더 소스를 최우선 검색하여 검증된 API 제어 기법을 모사합니다.
3. **사용자 의사결정이 필수적인 경우 대안(Options) 제시 (Ask-Path)**:
   * 격자 해상도 조절, 불완전 구속 해결을 위한 강체 모드 억제 기법 등 엔지니어의 공학적 판단이 반드시 필요한 경계에 다다르면 단정적 가정을 금지하고, 2가지 이상의 실행 가능한 안(Options)을 수립하여 질문을 던집니다.

---

## 📝 2. Abaqus / CalculiX 입력 덱 및 다중물리 표준 규약

CalculiX ccx는 Abaqus 호환 `.inp` 파일 형식을 준수하므로, 파일 생성 시 다음 규칙을 철저히 적용합니다.

1. **계층적 Include 구조 보증**:
   * 멀티 파트 및 대규모 격자 제어를 위해, 격자 파일(`mesh.inp`)과 하중/경계조건/해석 속성을 정의한 마스터 파일(`job.inp`)을 물리적으로 완벽히 분리하고 `*INCLUDE, INPUT=mesh.inp`로 참조합니다.
2. **이름(Name) 정규화 및 인코딩**:
   * 모든 노드세트(`NSET`), 엘리먼트세트(`ELSET`)명은 영문 대문자, 숫자, 언더바(`_`)만 허용하며 공백을 정밀 제거합니다. 모든 파일은 **BOM 없는 UTF-8**로 인코딩하여 저장합니다.
3. **요소(Element) 차수 준수**:
   * CalculiX의 해석 신뢰성을 극대화하기 위해 1차(Linear) 요소 대신 **2차(Quadratic) 고차 요소(예: C3D10, C3D20R 등)**를 기본 요소형으로 강제 채택합니다.
4. **해석 도메인별 스텝 카드 빌딩**:
   * **정적 구조 해석 (*STATIC)**: 대변형 해석 시 `*STEP, NLGEOM=YES` 카드를 사용하여 기하학적 비선형 해석을 활성화합니다.
   * **고유진동 해석 (*FREQUENCY)**: `*FREQUENCY` 밑에 연산할 고유치 개수를 지정합니다.
   * **과도 및 모달 동해석 (*DYNAMIC / *MODAL DYNAMIC)**: 구조물에 작용하는 시간 이력 과도 응답 해석 시 가동하며, 모달 감쇠 비(`*DAMPING`)를 동시 세팅합니다.
   * **열전달 해석 (*HEAT TRANSFER)**: 온도 분포 조회를 위해 `*HEAT TRANSFER` 카드와 전도/대류/복사 경계 조건(FILM, DFLUX) 카드를 조립합니다.
   * **유체 관망 해석 (*FLUID)**: 유체 파이프 관망 유동압 트래킹을 위해 `*FLUID SECTION`과 압력 경계 카드를 매핑합니다.
   * **전자기 해석 (*ELECTROMAGNETICS)**: 맥스웰 전자기 필드 및 줄 가열, 로렌츠 힘 계산을 위해 `*ELECTROMAGNETICS` 및 유도 경계 조건을 삽입합니다.

---

## 🏗️ 3. 실행형 에이전트 API 기능 명세 (Implemented Skills)

에이전트는 본 스킬 패키지 하위에 내장된 파이썬 모듈 및 RAG 유틸리티를 활용하여 FEM 해석을 지능적으로 제어합니다.

1. **Mesh Transposer & INP Assembler Skill**: GMSH에서 추출한 격자 토폴로지를 ccx 격자로 트랜스포즈하여 마스터 `.inp` 덱으로 조립 및 익스포트하는 API (`InpAssembler.write_multi_physics_deck`).
2. **ccx Solver Runner Skill**: Windows PowerShell 환경에서 멀티스레딩(OMP)을 지원하는 `ccx_static.exe`를 비동기로 구동하고 실시간 수렴성(`*.sta`, `*.cvg`)을 트래킹하는 실행 API (`CalculixSolverRunner.run_solve`).
3. **FEA Output Parser Skill**: 해석 완료 후 DAT 파일에서 고유진동수 및 참여 계수를 추출하고, ASCII FRD 파일에서 각 노드별 변위(DISP) 및 응력(STRESS) 벡터 성분을 고속 해독하는 파이서 API (`FeaOutputParser.parse_frd_displacements`).
4. **ParaView Style Interactive PyVista Viewer Skill**: 10% warped 형상 스케일링, Black 배경 ParaView 비주얼 스타일 준수 및 마우스 우클릭으로 XY/YZ/ZX 투영 뷰를 실시간 토글 제어하는 3D 뷰어 API (`InteractiveModeShapeViewer`).
5. **SQLite FTS5 & Vector RAG Skill**: 스킬 하위 `resources/calculix_rag.db` 파일에 대해 SQLite FTS5 및 TF-IDF 벡터 연산으로 관련 C/Fortran 서브루틴 및 .inp 예제 조각을 비동기로 Top-K 코사인 유사도 검색하는 엔진 (`LightweightCodeRAG.query_local_rag`).

---
Assisted by WHTOOLs Calculixent
