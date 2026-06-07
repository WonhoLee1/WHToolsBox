# Walkthrough: Radioss Model Builder 배치 자동화 연동 (2026-06-06)

## 🎯 작업 개요
요청해주신 "배치로(no viewer) 모델과 해석을 진행하는 파이프라인에서 Radioss 파일을 생성하는 기능을 호출할 수 있도록 연계해달라"는 내용에 대한 작업이 모두 완료되었습니다. 이전 세션에서 선조치된 `whts_radioss_builder.py`의 `NameError` 핫픽스 이후, 이를 실제 파이프라인(`run_drop_simulation_cases_v6.py`) 내에 매끄럽게 통합했습니다.

---

## 🛠️ 주요 변경 사항

### 1. `run_digital_twin_pipeline_v6` 내 Radioss 연계 기능 추가
MuJoCo 시뮬레이션이 성공적으로 완료되고, 결과 파일(`.pkl`) 및 `DropSimResult` 객체가 확보된 직후에 `RadiossModelBuilder`가 자동 호출되도록 구현하였습니다.

- **낙하 자세(Posture) 추출**: 사용자가 `cfg`에 정의한 `drop_direction` 혹은 `LTL` 기반의 초기 낙하 위치 및 틸트(기울기) 각도는 첫 프레임(`result.pos_hist[0]` 및 `result.quat_hist[0]`)의 물리 엔진 데이터를 이용해 추출됩니다. 
- **자동 역변환 연동**: MuJoCo 쿼터니언을 $3\times3$ 회전 행렬(`R_mat`)과 병진 행렬(`t_vec`)로 완벽하게 변환하여 Radioss의 `/TRANSFORM` 카드 생성 로직에 전달하도록 구현했습니다.

### 2. No Viewer (Headless) 모드의 배치 케이스 추가 (`test_case_batch_radioss_setup`)
인터랙티브 뷰어가 켜지지 않고 백그라운드 연산으로만 모든 프로세스를 완료한 후, 모델 및 `.rad` 파일을 일괄 생성할 수 있도록 특화된 테스트 케이스를 구성했습니다.

- 기존의 기준 케이스인 `test_case_1_setup`을 래핑(Wrapping)하여 `use_viewer=False`가 주입되도록 수정했습니다.
- `__main__` 실행 시 곧바로 위 배치 케이스를 실행하도록 진입점을 수정하였습니다.

---

## ✅ 검증 내역
1. `python run_drop_simulation_cases_v6.py` 실행 시:
   - 뷰어 표시 없이 곧바로 $t=2.0$s 까지 시뮬레이션 데이터가 Headless 모드로 순식간에 계산되었습니다.
   - 마지막에 `🚀 [WHTOOLS] Generating Radioss Explicit Models...` 및 성공 로그(`Radioss Models generated successfully`)가 출력되는 것을 확인했습니다.
   - `RadiossModelBuilder`가 `/TRANSFORM` 등 필요한 파트 형상을 정상적으로 빌드하도록 지원합니다.

> 💡 **Tip:** 향후 다른 케이스(`test_case_2_setup` 등)를 정의하시더라도, `run_digital_twin_pipeline_v6` 파이프라인을 통과하기만 하면 언제나 동일하게 초기 낙하 자세 기반의 Radioss 모델 세트가 자동 추출됩니다.

### 3. 기준 좌표계(Transform Mode) 선택 플래그 추가
export_radioss_transform_mode 플래그를 추가하여, 추출된 위치와 기울기를 어디에 적용할지 선택할 수 있도록 개선했습니다.
- 'parts': 바닥(Ground)은 고정된 채 패키지 세트를 기울이고 이동시킵니다.
- 'ground': 패키지 세트는 정방향(원점)을 유지하고, 바닥면의 좌표를 역으로 회전 및 병진이동시켜 충돌 상황을 모사합니다.
