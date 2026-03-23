# Implementation Plan - run_discrete_builder.py 구문 오류 수정 (2026-03-23)

## 1. 개요
`run_discrete_builder.py` 파일 로드 시 발생의 `SyntaxError`를 해결하고, 코드의 구조적 정합성을 확보합니다.

## 2. 문제 분석
- `calculate_inertia` (498행): `BaseDiscreteBody` 클래스의 메서드임에도 들여쓰기가 누락되어 전역 함수로 인식되고 있습니다.
- `get_worldbody_xml_strings` (609행): 
    - 694행에 `을 기반으로 클래스 참조`라는 오염된 문자열이 포함되어 있습니다.
    - 695행부터 753행까지 과거 버전 또는 중복된 코드가 병합 오류로 인해 남아 있습니다.

## 3. 수정 단계
### 3.1. `calculate_inertia` 들여쓰기 수정
- 498행부터 607행까지의 모든 코드를 4칸 들여쓰기하여 `BaseDiscreteBody` 클래스 내부로 이동시킵니다.

### 3.2. `get_worldbody_xml_strings` 정형화
- 609행부터 시작되는 최신 로직(Single-Body 및 Multi-Body 대응)을 유지합니다.
- 694행의 가비지 문자열(`return xml_outs을 기반으로 클래스 참조`)을 올바른 `return xml_outs`로 수정하거나 삭제합니다.
- 695행부터 754행 이전까지의 중복된 하위 로직 및 구 버전 코드를 삭제합니다.

### 3.3. 검증
- `python -m py_compile run_discrete_builder.py` 명령을 통해 문법 오류가 없는지 확인합니다.
- `run_drop_simulation.py`를 실행하여 임포트가 정상적으로 이루어지는지 확인합니다.

---
> [!IMPORTANT]
> 기존 코드의 기능을 손상시키지 않으면서 구조적 문제만 해결합니다.
