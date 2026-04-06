import numpy as np
from run_discrete_builder.whtb_utils import parse_drop_target

def test_mapping():
    # 박스 치수 (W, H, D)
    bw, bh, bd = 1.0, 2.0, 0.5
    
    print("--- ISTA 6-Amazon Mapping Test (Y=Height, Z=Depth) ---")
    
    # 1. PARCEL (Type G)
    print("\n[PARCEL - Type G]")
    cases_g = [
        ("Face 1", "Top (+Y)"),
        ("Face 2", "Bottom (-Y)"),
        ("Face 5", "Front (+Z)"),
        ("Face 6", "Back (-Z)"),
        ("Face 3", "Right (+X)"),
        ("Face 4", "Left (-X)"),
    ]
    for cmd, desc in cases_g:
        pt = parse_drop_target("PARCEL", cmd, bw, bh, bd)
        print(f"  {cmd} ({desc}): {pt}")

    # 2. LTL (Type H)
    print("\n[LTL - Type H]")
    cases_h = [
        ("Face 1", "Top (+Y)"),
        ("Face 2", "Bottom (-Y)"),
        ("Face 3", "Front (+Z)"),
        ("Face 4", "Back (-Z)"),
        ("Face 5", "Right (+X)"),
        ("Face 6", "Left (-X)"),
        ("Corner 2-3-5", "Bottom-Front-Right (-Y, +Z, +X)")
    ]
    for cmd, desc in cases_h:
        pt = parse_drop_target("LTL", cmd, bw, bh, bd)
        print(f"  {cmd} ({desc}): {pt}")

if __name__ == "__main__":
    test_mapping()
