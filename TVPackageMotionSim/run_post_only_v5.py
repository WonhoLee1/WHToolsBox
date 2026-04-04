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
    dirs = glob.glob("rds-*")
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
        
        # 분석 및 대시보드 실행
        run_analysis_and_dashboard(result)
        
    except Exception as e:
        print(f"❌ Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
