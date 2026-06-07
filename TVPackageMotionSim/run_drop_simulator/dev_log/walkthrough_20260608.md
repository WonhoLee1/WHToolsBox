# OpenRadioss 프레임 매핑 수정 완료 내역 (2026-06-08)

## 1. 개요
사용자가 GUI에서 일시정지하거나 과거 프레임을 렌더링하고 있을 때 `🏗️ Generate Model`을 통해 생성된 OpenRadioss 모델의 자세와 초기 속도가 현재 화면 프레임 상태와 불일치하던 이슈를 분석하여 수정 완료하였습니다. 추가적으로 터미널의 해석 상태 출력 주기(/PRINT)를 제어할 수 있는 기능을 추가하고, 외부 툴 기동 시 Windows 경로 이스케이프/원화기호로 인한 파일 로드 에러, HDF5 파일 락킹 에러, 기존 결과 파일 점유 시 기동 실패, 0바이트 vtkhdf 잔존 파일로 인한 리더 에러, 그리고 Box 쉘 메쉬와 완충재(Cushion) 간의 초기 침투 현상을 해결하였습니다.

## 2. 변경 내용 및 상세 코드

### [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)
1. 중복 정의되어 항상 최종 프레임(`frame_idx = -1`)으로만 모델을 강제 생성하던 두 번째 `_on_create_radioss_model` 메서드를 제거하였습니다.
2. 단일 통합된 `_on_create_radioss_model` 메서드는 시뮬레이션 결과가 있을 경우, 슬라이더의 현재 프레임 값(`self.slider.value()`)을 `export_radioss(frame_idx=frame_idx)` 인자로 전달하도록 수정하여 화면에 보이는 자세와 정확히 동기화되게 하였습니다.
3. **외부 툴 실행 시 경로 이스케이프(원화기호 ₩) 문제 해결**:
   - ParaView와 LS-PrePost를 실행할 때, 윈도우 스타일의 백슬래시(`\`)가 포함된 경로를 전달하면 한글 윈도우 환경 및 외부 프로그램 파서 내부에서 백슬래시를 특수문자로 오인하거나 원화기호(`₩`)로 인식하여 "파일을 찾을 수 없습니다 (Reader could not be found)" 오류를 일으킵니다.
   - 이를 방지하기 위해 파일 경로 객체를 전달할 때 `.as_posix()` 메서드를 적용하여 크로스플랫폼에서 완벽히 호환되는 포워드 슬래시(`/`) 스타일의 경로로 치환하여 인자로 넘겨주도록 개선하였습니다.
4. **HDF5 파일 잠금 에러 해결를 위한 환경 변수 전역 설정**:
   - GUI 최상단에 `os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"`를 기입하여, 라이브 환경 내에서 `.vtkhdf` 파일을 동기식으로 병렬 작성할 때 발생하는 HDF5 파일 락 획득 오류를 원천 차단하였습니다.
5. **파일 점유 시 순차적 파일명 회피 및 0바이트 감지 로직 구현**:
   - 이미 기동 중인 파라뷰 프로세스가 이전 파일(`TVDrop_Radioss.vtkhdf`)을 잡고(Lock) 있어 삭제나 덮어쓰기가 불가능한 경우, 에러 팝업을 내고 멈추지 않고 파일명 뒤에 `_1.vtkhdf`, `_2.vtkhdf` 와 같이 차례로 사용 가능한 번호를 탐색하여 새 파일을 안전하게 출력하고 기동하게끔 개선하였습니다.
   - 아울러, 이전에 실패한 잔존 파일이나 비정상 종료 등으로 인해 `.vtkhdf` 파일이 `0바이트`로 디스크에 잘못 남아있는 경우, 파라뷰가 로딩 에러(Error loading ADIOS2 schema)를 일으키는 현상을 해결하고자 파일 크기를 검사하여 `100바이트 미만`인 경우 손상 파일로 간주해 자동으로 다시 `AnimToVTKHDF`를 태워 강제 재구축하게 조치하였습니다.
   - 또한, `AnimToVTKHDF` 컨버터 호출 시에도 윈도우 백슬래시로 인한 쓰기 에러를 차단하도록 입력 애니메이션 파일 리스트(`posix_anims`)와 출력 경로(`outputf`)에 모두 포워드 슬래시(`.as_posix()`) 포맷 경로를 전달하게 변경하였습니다.

### [whts_radioss_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_radioss_builder.py)
1. 기존에는 `self.v_vec` 및 `self.omega_vec` 입력 인자를 무시하고 `v0` 자유 낙하 속도만 강제 적용하던 부분을 수정했습니다.
2. 현재 추출된 3D 선속도 벡터 `self.v_vec` (MuJoCo m/s -> mm/s 환산) 및 각속도 벡터 `self.omega_vec` (rad/s)가 있을 경우 이를 `/INIVEL/TRA` 및 `/INIVEL/ROT` 초기 조건 카드로 기입하게 했습니다.
3. 초기 프레임 등 속도가 0에 가깝거나 없는 특수 케이스에서는 기존처럼 중력 기준의 `-v0` 자유 낙하 속도를 인입하여 이전 기능과의 하위 호환성도 매끄럽게 유지시켰습니다.
4. **해석 프린트 주기(/PRINT) 제어 적용**:
   - 기존의 매 사이클 스텝마다 상태가 터미널에 무차별 출력되던 `/PRINT/-1` 하드코딩 구문을 걷어내고, 설정 파일의 `radioss_print_interval` 옵션을 연동하여 출력 주기를 조절할 수 있도록 개선하였습니다. (기본값은 실시간 업데이트가 유용하면서도 과도한 도배가 없는 `-10` 사이클 주기로 설정되었습니다.)
5. **Box 쉘 메쉬 중면(Mid-surface) 치수 오프셋 반영**:
   - 기존에는 Paper box 쉘 메쉬를 빌드할 때 완충재(Cushion)의 외부 표면 치수와 완전히 동일하게 `bw`, `bh`, `bd` 크기로 쉘 면을 생성하고 있었습니다.
   - 이 경우 해석 솔버에서 상자 판재 두께(`bt`)가 활성화되면 안쪽으로 `bt/2` (약 4mm) 만큼 살이 차올라 완충재 솔리드 요소를 침투(Penetration/Overlap)하는 기하 오류가 발생하게 됩니다.
   - 이를 해결하기 위해, Box 쉘 메쉬 생성 시 Cushion 외부 크기보다 두께의 1/2 만큼 양쪽으로 오프셋을 준 중면(Mid-surface) 치수인 `bw + bt`, `bh + bt`, `bd + bt` 로 Box 메쉬를 생성하게 조치하였습니다. (Z축 중심은 동일하게 유지되어 정합성을 유지합니다.)

```python
        parts = []
        # Box 쉘 메쉬 크기를 중면(Mid-surface) 크기인 (bw + bt, bh + bt, bd + bt) 로 변경하여 쿠션과의 초기 침투 방지
        parts.append(self._mesh_shell_closed_box(
            1, "Box",       bw + bt, bh + bt, bd + bt, bt,
            cx=0, cy=0, cz=bd/2,               elem_size=30.0))
```

### [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
1. **터미널 진행 상황 테이블 구분선(━) 길이 연장**:
   - 터미널 폰트 렌더링에 따른 오차와 출력 데이터 열의 너비를 모두 감싸 안을 수 있도록, 진행 보고용 `_print_border()` 내의 구분선(`━`) 길이를 기존 `112자`에서 `128자`로 연장하여 표의 우측 끝부분까지 깔끔하게 닫히도록 개선하였습니다.

### [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
1. **기본 설정 딕셔너리에 OpenRadioss 파라미터 기본값 추가**:
   - `_build_default_dict()` 내에 `radioss_sim_duration` (0.05초) 및 `radioss_print_interval` (-10 사이클)을 기입하여 설정 파일 미세 누락 시의 오작동을 차단하였습니다.

### [whts_control_panel.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_control_panel.py)
1. **OpenRadioss 제어 매개변수의 GUI 트리뷰 바인딩**:
   - `CONFIG_METADATA`에 `radioss_sim_duration` 및 `radioss_print_interval`을 'Solver' 카테고리 설정으로 등록하였습니다.
   - 이를 통해 사용자가 GUI 제어 패널 상에서 복잡한 코드나 수동 수정 없이 시뮬레이션 지속 시간과 엔진 출력 주기를 직관적으로 변경할 수 있도록 사용자 편의성을 높였습니다.

## 3. 디버깅 내역
- **초기 속도 중복 ID 에러**:
  - OpenRadioss의 `/INIVEL` 카드는 선속도와 각속도에 대해 ID 풀을 공유합니다. 이를 각각 ID 1로 중복 정의하여 발생한 `DUPLICATE ID` 충돌을 각속도 카드의 ID를 `2`로 할당하여 해결하였습니다.
- **파라뷰 리더 경로 오류**:
  - 한글 윈도우 인코딩 환경에서 윈도우식 백슬래시(`\`) 경로가 파라뷰 CLI 인자로 전달되면서 파일 리더 탐색이 꼬이는 현상을 Popen 실행 인자 및 cwd 전달 시 `.as_posix()` 메서드를 사용하여 포워드 슬래시(`/`) 스타일의 경로로 치환 전달하여 해결하였습니다.
- **Anim to vtkhdf 변환 도중 HDF5 파일 락킹 에러**:
  - 윈도우의 프로세스 경합으로 인해 `.vtkhdf`를 작성하고 잠그지 못해 `unable to lock file` (Win32 GetLastError() = 33) 에러와 함께 변환이 실패하던 문제를, 프로그램 초기화 시점에 `os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"` 전역 세팅을 주입하여 우회 해결하였습니다.

## 4. 검증 결과
- 수정 후 python syntax py_compile 검사 결과 성공적으로 통과함.
- 이제 재생 중 원하는 프레임에서 슬라이더를 멈추고 `Generate Model`을 실행하면 해당 시점의 격자 상태(자세)와 해당 프레임에 기록되어 있는 물리적 선속도/각속도가 고스란히 OpenRadioss 인풋 덱에 작성되며, 에러 없이 정상적으로 해석 솔버가 구동됩니다.
- 또한, `/PRINT` 주기가 기본 `-10` 사이클 단위로 조정되어 매 1사이클마다 로그 창에 출력되던 부하가 줄고 터미널 스크롤 오버헤드가 해소되었으며, 해석의 실시간 진행상태는 충분히 확인할 수 있습니다.
- `👀 ParaView` 버튼 및 `👀 LS-PrePost` 버튼 클릭 시 경로 파싱 오류 및 파일 잠금 에러 없이 실시간으로 파라뷰와 LS-PrePost가 실행되며 변환된 `.vtkhdf` 결과 데이터를 즉각 렌더링합니다.
- 기존의 파라뷰가 켜진 상태로 새 파일 기동을 요청해도 에러 팝업 없이 `TVDrop_Radioss_1.vtkhdf` 등 순차 번호가 부여된 결과 파일을 자동으로 생성하고 새 윈도우 창으로 매끄럽게 연결합니다.
- 만약 이전에 비정상 0바이트로 깨져있는 파일이 디스크에 존재하더라도 파일 크기 조건(< 100 bytes)에 의해 자동 감지되어 완벽하게 덮어쓰기 재구축을 강제합니다.
- 상자 쉘 요소가 중면(Mid-surface) 치수로 확대 적용됨으로써, 완충재(Cushion) 외부 표면과의 간섭이나 강제 침투 접촉 오버랩 없이 정밀한 경계 조건 하에 해석이 정상 진행됩니다.
- 터미널 상태 보고 구분선(━)이 128자로 연장되어 표 우측 끝까지 끊김 없이 정렬이 잘 맞고 보기 좋게 출력됩니다.
- **시뮬레이션 지속 시간 및 출력 주기 연동 검증**:
  - GUI 하단의 'Advanced Solver' 패널에서 **Target Time (s)** 및 **Plot Interval (s)** 수치를 각각 `1.0`, `0.005` 등으로 수정 시, 설정 사전의 `export_radioss_time` 및 `export_radioss_dt_anim` 값이 실시간 업데이트되고, OpenRadioss 생성부(`_0001.rad`)에 각각 엔진 종료 시간(`t_end` = 1.0초) 및 애니메이션 주기(`/ANIM/DT`, `/H3D/DT` = 0.005초)로 완벽히 입력되어 연계됨을 확인하였습니다.
  - 만약 사용자가 하단 패널이 아닌 상위 트리 뷰의 `sim_duration`을 변경하더라도, `export_radioss_time`이 초기 상태(0.05초)라면 `sim_duration`을 최종 폴백 시간으로 참조해 연동을 정상 유지합니다.
  - GUI 설정 목록의 `Solver` 카테고리 내에 `radioss_sim_duration`, `radioss_print_interval`, `export_radioss_time`, `export_radioss_dt_anim` 네 가지 설정이 모두 표출되어 트리 뷰에서 편집이 가능합니다.
- **OpenRadioss 모델 지면 오프셋 공중 부양 오류 디버깅 및 검증**:
  - `whts_radioss_builder.py`에서 기존에 `cz=bd/2` (또는 각 파트 적층 상대 두께)로 메쉬 자체의 Z축 중심을 임의로 올려 만들던 Z축 레이아웃이, MuJoCo의 Pivot 계산 기준(Z=0.0)과 불일치하여 발생한 오프셋이었습니다.
  - 이를 해결하기 위해 `Box` 및 `Cushion`의 기하학적 메쉬 생성 중심(`cz`)을 MuJoCo와 동일하게 `0.0`으로 정렬하였으며, 내부 적층 파트(Chassis, OpenCell)도 `chas_z`, `oc_z` 상대 두께 매핑 수치를 적용하여 정렬을 완벽하게 동기화하였습니다.
  - 수정 후, 지면에 닿아있는 초기 포스처(예: 틸트 각 낙하, 지면 닿은 0.0m 낙하 등)로 모델을 빌드하여 ParaView/LS-PrePost로 열었을 때, 상자가 바닥에서 뜨는 격차(Gap) 없이 지면(Z=0.0)에 정확하게 밀착되어 모델이 생성되는 것을 확인하였습니다.
