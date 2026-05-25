import os
import sys
import pickle
import glob
from typing import Any

# [WHTOOLS] 패키지 경로 추가
sys.path.append(os.getcwd())

# [WHTOOLS] 모듈 임포트
from run_drop_simulation_cases_v5 import run_analysis_and_dashboard

def get_latest_result_dir():
    """가장 최근에 생성된 시뮬레이션 결과 디렉토리를 찾습니다."""
    dirs = glob.glob("results/rds-*")
    if not dirs:
        return None
    # 디렉토리 생성 시간 순으로 정렬
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]

if __name__ == "__main__":
    print("\n" + "="*85)
    print(" 🛠️  WHTOOLS Post-Only Analysis Mode (v5.3.5)")
    print("="*85)
    
    target_dir = None
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = get_latest_result_dir()
        
    if not target_dir or not os.path.exists(target_dir):
        print(f"❌ Error: No valid result directory found. (Found: {target_dir})")
        sys.exit(1)
        
    pkl_path = os.path.join(target_dir, "simulation_result.pkl")
    if not os.path.exists(pkl_path):
        print(f"❌ Error: 'simulation_result.pkl' not found in {target_dir}")
        sys.exit(1)
        
    print(f"📂 Loading data from: {target_dir}")
    try:
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)

        # 구버전 pkl에 components가 없으면 XML에서 복구
        if not getattr(result, 'components', None):
            xml_path = os.path.join(target_dir, "simulation_model.xml")
            if os.path.exists(xml_path):
                import mujoco as _mj
                _model = _mj.MjModel.from_xml_path(xml_path)
                _prefixes = ['bpaper', 'bcushion', 'bchassis', 'bopencell', 'inertiaaux', 'autobalance']
                _comps, _extents = {}, {}
                for _i in range(_model.nbody):
                    _name = _mj.mj_id2name(_model, _mj.mjtObj.mjOBJ_BODY, _i)
                    if not _name: continue
                    for _p in _prefixes:
                        if _p in _name.lower():
                            _comps.setdefault(_p, {})
                            _parts = _name.split('_')
                            try: _idx = (int(_parts[-3]), int(_parts[-2]), int(_parts[-1])) if len(_parts) >= 4 else (0, 0, 0)
                            except: _idx = (0, 0, 0)
                            _comps[_p][_idx] = _i
                            if _model.body_geomnum[_i] > 0:
                                _extents[_i] = _model.geom_size[_model.body_geomadr[_i]].copy()
                            break
                result.components = _comps
                if not getattr(result, 'block_half_extents', None):
                    result.block_half_extents = _extents
                print(f"  ✅ components recovered: { {k: len(v) for k,v in _comps.items()} }")

        # 분석 및 대시보드 실행
        run_analysis_and_dashboard(result)
        
    except Exception as e:
        print(f"❌ Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
