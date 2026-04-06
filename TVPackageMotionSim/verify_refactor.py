# -*- coding: utf-8 -*-
import os
from run_discrete_builder import get_default_config, create_model

def verify_refactor_v2():
    print("\n--- [Refactor V2 Verification] ---")
    
    # 1. Default Config 확인 (Case 1 수치 이식 여부)
    cfg = get_default_config()
    
    # 주요 수치 체크 (Case 1 사양)
    # chassis_weld_solref_damprr이 0.3인지 확인
    checks = {
        "chassis_weld_solref_damprr": cfg.get("chassis_weld_solref_damprr") == 0.3,
        "opencell_weld_solref_damprr": cfg.get("opencell_weld_solref_damprr") == 0.5,
        "cush_weld_solref_damprr": cfg.get("cush_weld_solref_damprr") == 0.8
    }
    
    all_pass = True
    for label, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        val = cfg.get(label, "N/A")
        print(f"[{status}] {label} (Val: {val})")
        if not passed: all_pass = False

    # 2. XML 생성 및 내용 정합성 확인
    test_xml = "verify_test_v2.xml"
    xml_content, _, _, _, _ = create_model(test_xml, config=cfg)
    
    # XML 내부의 물리 문자열 체크
    xml_checks = {
        "Chassis Solref (0.002 0.3)": 'solref="0.002 0.3"' in xml_content,
        "Cushion Conaffinity (31)": 'conaffinity="31"' in xml_content,
        "NoSlip Iterations": 'noslip_iterations="20"' in xml_content
    }
    
    for label, passed in xml_checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] XML Content: {label}")
        if not passed: all_pass = False

    if all_pass:
        print("\n✅ Verification Successful! V2 Refactor is stable.")
    else:
        print("\n❌ Verification Failed. Please check logs.")

if __name__ == "__main__":
    verify_refactor_v2()
