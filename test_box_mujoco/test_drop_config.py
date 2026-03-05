import os
from run_discrete_builder import create_model, get_default_config

cases = ["PARCEL", "LTL", "F", "F-T", "R-B-L", "T-L", "B-R"]

for case in cases:
    print(f"Testing drop mode: {case}")
    cfg = get_default_config()
    cfg["drop_mode"] = case
    cfg["include_paperbox"] = False
    
    out_file = f"test_drop_{case.replace('-', '')}.xml"
    create_model(out_file, config=cfg)
    
print("All drop parsing tests passed successfully.")
