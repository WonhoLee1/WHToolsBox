---
name: openradioss
description: OpenRadioss simulation skill referring to whtools-openradiossent guidelines, D:\PythonCodeStudy\WHT_Calculixent\wht_openradiossent_doc documents, and Altair official help documentation.
---

# 🛠️ OpenRadioss Simulation Skill

본 파일은 **Antigravity** 에이전트가 OpenRadioss 시뮬레이션 카드 생성, 해석 유효성 검증, 물리 엔진(MuJoCo) 연동을 수행할 때 준수해야 하는 행동 지침 및 스킬 규약서입니다.

이 스킬의 도움을 받아 개발된 모든 공학 소프트웨어와 아티팩트에는 다음 서명을 명기합니다:
**Assisted by WHTOOLs OpenRadiossent**

---

## 🧭 1. 레퍼런스 및 외부 문서 최우선 준수 규칙

AI 에이전트는 해석 모델 생성 로직을 변경하거나 신규 코드를 구현할 때 반드시 다음 자료들을 최우선으로 검토하고 반영해야 합니다.

1. **로컬 스킬 규약 (`whtools-openradiossent`)**:
   - `skills/whtools-openradiossent/SKILL.md` 파일에 정의된 규약을 상시 참고합니다.
   - 물리적 계층 분리(스타터 덱 `_0000.rad`과 엔진 덱 `_0001.rad`의 분리) 보장.
   - 디폴트 단위계로 **[mm - ms - g - N - MPa]** 사용.
   - 비선형 탄소성을 위한 `LAW2 (PLAS_JOHNS)` 및 접촉 강성 확보를 위한 `TYPE7` 접촉면 설정 가이드라인 준수.
   - 질량 스케일링(Mass Error < 2%) 및 Hourglass 에너지(< 5%)의 적합성 평가 진단.

2. **외부 설명 문서 (`wht_openradiossent_doc`)**:
   - `D:\PythonCodeStudy\WHT_Calculixent\wht_openradiossent_doc` 하위 폴더 내에 포함된 설명 마크다운(`.md`) 파일들을 참고하여 OpenRadioss 적용 가이드라인과 설계 팁을 숙지하고 이를 코드 구현에 적용합니다.

3. **Altair 공식 도움말 웹 문서 (Altair Radioss Help Document)**:
   - 개발 및 코드 작성 과정에서 모르는 키워드, 카드 형식, 옵션 또는 에러 코드가 발견될 경우 반드시 공식 온라인 문서를 검색 및 참고해야 합니다.
   - **URL:** [Altair Radioss Help Home](https://help.altair.com/hwsolvers/rad/index.htm)
   - 주요 카드 구문(`/BCS`, `/INIVEL`, `/TRANSFORM` 등)의 유효 매개변수 및 서식 규칙은 공식 문서를 대조하여 문법 에러(Syntax Error)를 사전에 방지합니다.

---

## 🏗️ 2. 핵심 구현 및 검증 수칙

- **중복 정의 금지**: `/BCS`, `/INIVEL` 등 경계 조건 및 초기 속도 카드 정의 시 동일 노드 세트에 중복된 ID 또는 중복 카드가 선언되지 않도록 필터링 로직을 엄격하게 유지합니다.
- **LS-DYNA (.k) 1:1 대응 포맷팅**: LS-DYNA 포맷 출력 시 `*INITIAL_VELOCITY_GENERATION` 카드를 포함한 카드별 파라미터 규격을 공식 레퍼런스 서식에 일치시켜 작성합니다.
- **인코딩 규약**: 모든 관련 파이썬 모듈 및 카드 덱 파일 생성/수정 시 반드시 **UTF-8 (BOM 없음)** 인코딩을 명시(`encoding='utf-8'`)하여 한글 폰트 및 데이터 깨짐 현상을 방지합니다.
