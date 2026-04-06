
import sys
import os

# 패키지 경로 추가
sys.path.append(os.getcwd())

from run_discrete_builder.whtb_config import get_default_config
from run_discrete_builder.whtb_builder import create_model as create_v2
from run_discrete_builder.whtb_builder_backup_20260405 import create_model as create_legacy

def compare_xmls():
    # 1. 동일한 설정 생성
    cfg = get_default_config({"drop_mode": "PARCEL", "drop_direction": "front-bottom-left", "include_paperbox": False})
    
    # 2. V2 XML 생성
    v2_io = create_v2("v2_temp.xml", config=cfg)
    v2_xml = v2_io[0]
    
    # 3. Legacy XML 생성
    legacy_io = create_legacy("legacy_temp.xml", config=cfg)
    legacy_xml = legacy_io[0]
    
    # 4. 차이점 출력
    if v2_xml == legacy_xml:
        print("✅ XMLs are IDENTICAL.")
    else:
        print("❌ XMLs are DIFFERENT.")
        
        # 간단한 라인별 비교
        v2_lines = v2_xml.splitlines()
        leg_lines = legacy_xml.splitlines()
        
        max_idx = max(len(v2_lines), len(leg_lines))
        diff_count = 0
        for i in range(max_idx):
            v_line = v2_lines[i] if i < len(v2_lines) else "MISSING"
            l_line = leg_lines[i] if i < len(leg_lines) else "MISSING"
            if v_line != l_line:
                print(f"Line {i+1}:")
                print(f"  V2 : {v_line}")
                print(f"  LEG: {l_line}")
                diff_count += 1
                if diff_count > 10:
                    print("... too many differences.")
                    break

if __name__ == "__main__":
    compare_xmls()
