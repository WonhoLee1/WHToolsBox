import sys
import numpy as np
import mujoco
import subprocess
from pathlib import Path

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from TVPackageMotionSim.run_drop_simulator.whts_data import DropSimResult
from TVPackageMotionSim.run_drop_simulator.whts_radioss_builder import RadiossModelBuilder

def generate_radioss_from_result(pkl_path: str, target_time: float = 0.05, transform_mode='parts'):
    """
    기존에 성공적으로 저장된 simulation_result.pkl 파일을 읽어서
    시뮬레이션 재실행 없이 Radioss (.rad) 파일을 생성합니다.
    """
    pkl_file = Path(pkl_path)
    if not pkl_file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {pkl_path}")
        return

    print(f"📂 결과 파일 로드 중: {pkl_path}...")
    res = DropSimResult.load(str(pkl_file))
    
    # 1. 대상 프레임 검색
    times = np.array(res.time_history)
    frame_idx = int(np.argmin(np.abs(times - target_time)))
    actual_time = times[frame_idx]
    print(f"⏱️ 타겟 시간: {target_time}s -> 추출 프레임: {frame_idx} (실제 시간: {actual_time:.4f}s)")
    
    # 2. Root Body ID 찾기 (일반적으로 BPackagingBox)
    rid = -1
    # body_index_map은 {BodyID: Name} 형태
    for b_id, b_name in res.body_index_map.items():
        if "PackagingBox" in str(b_name) or "Chassis" in str(b_name):
            rid = b_id
            break
            
    if rid == -1:
        # 매핑이 없으면 0번 인덱스 사용
        rid = 0
        print("⚠️ 명시적인 기준 바디를 찾지 못해 기본 인덱스(0)를 사용합니다.")
    else:
        print(f"🎯 기준 바디 ID 발견: {rid} ({res.body_index_map.get(rid)})")
        
    # 3. 위치, 회전, 속도 추출 (엔진의 로직과 동일)
    quat_mj = res.quat_hist[frame_idx][rid]
    R_flat = np.zeros(9)
    mujoco.mju_quat2Mat(R_flat, quat_mj)
    R_mat = R_flat.reshape(3, 3)
    t_vec = res.pos_hist[frame_idx][rid].copy()
    
    # Radioss 초기 관통(Initial Penetration) 에러 방지를 위해 Z축으로 약간 올려줍니다. (51.4mm 관통이 보고되었으므로 60mm 상향)
    t_vec[2] += 0.060

    
    cvel = res.vel_hist[frame_idx]
    omega_vec = cvel[:3]
    v_vec = cvel[3:]
    
    # 4. Radioss 모델 빌더 호출
    h = res.config.get("drop_height", 0.5)
    name = res.config.get("model_name", "TVDrop_Radioss")
    out_dir = pkl_file.parent
    
    print(f"🚀 Radioss 모델 생성 시작 (출력 폴더: {out_dir})...")
    builder = RadiossModelBuilder(
        config=res.config,
        output_dir=out_dir,
        R_mat=R_mat,
        t_vec=t_vec,
        v_vec=v_vec,
        omega_vec=omega_vec,
        transform_mode=transform_mode,
        drop_height_m=h,
        model_name=name,
    )
    starter = builder.build()
    print("✅ 생성 완료!")
    return starter

def check_radioss_syntax(starter_file: Path):
    print(f"\n🔍 OpenRadioss Starter를 실행하여 문법 무결성 검사를 시작합니다...")
    _gui_dir = str(Path(r"D:\OpenRadioss_win64\OpenRadioss\openradioss_gui"))
    if _gui_dir not in sys.path:
        sys.path.insert(0, _gui_dir)
    
    try:
        from runopenradioss import RunOpenRadioss
    except ImportError:
        print("❌ OpenRadioss GUI 모듈을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    command = [
        str(starter_file),   # [0] input file
        '4',                 # [1] OpenMP threads
        '1',                 # [2] MPI processes
        'dp',                # [3] precision
        'no',                # [4] anim_to_vtk
        'no',                # [5] th_to_csv
        'yes',               # [6] starter_only (문법 검사용)
        'no',                # [7] anim_to_d2plot
        'no',                # [8] anim_to_vtkhdf
        '',                  # [9] mpi_path
    ]
    
    runner = RunOpenRadioss(command, debug=0)
    runner.batch_run()
    
    # .out 파일 파싱하여 에러 검사
    out_file = starter_file.with_name(starter_file.stem + ".out")
    if out_file.exists():
        content = out_file.read_text(encoding='utf-8', errors='replace')
        errors = []
        warnings = []
        for line in content.splitlines():
            upper_line = line.upper()
            if "ERROR" in upper_line:
                if "0 ERROR" not in upper_line and "NO SYNTAX ERROR" not in upper_line and "ERROR(S) SUMMARY" not in upper_line:
                    errors.append(line)
            if "WARNING" in upper_line:
                if "0 WARNING" not in upper_line and "WARNING(S) SUMMARY" not in upper_line:
                    warnings.append(line)
        
        print("\n📊 [문법 검사 결과]")
        if errors:
            print(f"❌ {len(errors)}개의 에러가 발견되었습니다:")
            for e in errors[:5]:
                print(f"   {e.strip()}")
            if len(errors) > 5:
                print("   ... (더 많은 에러는 .out 파일을 확인하세요)")
        else:
            print("✅ 문법 에러가 발견되지 않았습니다. (정상)")
            print("🚀 엔진(Engine)을 실행합니다 (전체 시뮬레이션)...")
            
            _gui_dir = str(Path(r"D:\OpenRadioss_win64\OpenRadioss\openradioss_gui"))
            if _gui_dir not in sys.path:
                sys.path.insert(0, _gui_dir)
            try:
                from runopenradioss import RunOpenRadioss
                command = [
                    str(starter_file),   # [0] input file
                    '4',                 # [1] OpenMP threads
                    '1',                 # [2] MPI processes
                    'dp',                # [3] precision
                    'no',                # [4] anim_to_vtk
                    'no',                # [5] th_to_csv
                    'no',                # [6] starter_only (no = run engine)
                    'no',                # [7] anim_to_d2plot
                    'yes',               # [8] anim_to_vtkhdf (yes = convert to vtkhdf automatically)
                    '',                  # [9] mpi_path
                ]
                # debug=0 disables GUI but batch_run() executes it
                runner = RunOpenRadioss(command, debug=0)
                runner.batch_run()
                
                target_dir = starter_file.parent
                engine_out = target_dir / "TVDrop_Radioss_0001.out"
                if engine_out.exists():
                    print("\n[엔진 실행 결과 요약]")
                    with open(engine_out, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                        print("".join(lines[-40:]))
            except Exception as e:
                print(f"엔진 실행 중 오류 발생: {e}")
            
        if warnings:
            print(f"⚠️ {len(warnings)}개의 경고가 발견되었습니다. (.out 파일을 참고하세요)")
    else:
        print(f"❌ 출력 파일(.out)을 찾을 수 없습니다: {out_file.name}")

if __name__ == "__main__":
    # 예시: 가장 최근에 성공한 폴더 경로를 입력하세요.
    # rds-YYYYMMDD_HHMMSS 형식의 가장 최근 폴더를 자동으로 찾습니다.
    results_dir = Path(r"c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\results")
    
    # 폴더 내의 가장 최신 rds- 폴더 찾기
    subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("rds-")])
    if subdirs:
        latest_dir = subdirs[-1]
        pkl_path = latest_dir / "simulation_result.pkl"
        
        # 원하는 출력 시간 및 모드 설정
        starter_file = generate_radioss_from_result(
            pkl_path=str(pkl_path), 
            target_time=0.0, 
            transform_mode='parts'
        )
        if starter_file:
            check_radioss_syntax(starter_file)
    else:
        print("❌ results 폴더 내에 시뮬레이션 결과가 없습니다.")
