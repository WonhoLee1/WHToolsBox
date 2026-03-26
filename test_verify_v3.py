
import sys
import os
import time

# test_box_mujoco 경로 추가
sys.path.append(os.path.abspath("test_box_mujoco"))

from run_drop_simulation_v3 import DropSimulator
from run_discrete_builder import get_default_config

def verify_v3():
    print("Starting verification of run_drop_simulation_v3.py...")
    
    cfg = get_default_config()
    cfg.update({
        "sim_duration": 0.1,    # 짧은 시뮬레이션
        "use_viewer": False,     # 헤드리스 모드
        "drop_height": 0.3,
        "plot_results": True
    })
    
    sim = DropSimulator(config=cfg)
    try:
        sim.setup()
        sim.simulate()
        sim.plot_results()
        
        # 출력 폴더 확인
        out_dir = sim.output_dir
        print(f"Checking output directory: {out_dir}")
        
        files = os.listdir(out_dir)
        print(f"Generated files: {files}")
        
        required_files = ["rds-impact_g_trace.png", "rds-ground_impact.png"]
        missing = [f for f in required_files if f not in files]
        
        if not missing:
            print("✅ Core results generated successfully.")
        else:
            print(f"❌ Missing files: {missing}")
            
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_v3()
