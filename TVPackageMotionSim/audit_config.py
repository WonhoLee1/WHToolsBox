import os
import re

def audit_config_keys():
    config_path = r"c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_discrete_builder\whtb_config.py"
    search_dirs = [
        r"c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_discrete_builder",
        r"c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulator"
    ]
    main_file = r"c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulation_cases_v4.py"

    # 1. Extract keys from whtb_config.py
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Simple regex to find "key": in the dictionary
    keys = re.findall(r'"([a-zA-Z0-9_]+)":', content)
    unique_keys = sorted(list(set(keys)))
    
    # 2. Collect all active python files
    py_files = []
    for d in search_dirs:
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith(".py") and "backup" not in f.lower() and f != "whtb_config.py":
                    py_files.append(os.path.join(root, f))
    py_files.append(main_file)

    print(f"Total Keys Found: {len(unique_keys)}")
    print(f"Total Files to Scan: {len(py_files)}")

    # 3. Check usage
    usage_count = {k: 0 for k in unique_keys}
    usage_details = {k: [] for k in unique_keys}

    for f_path in py_files:
        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
            f_content = f.read()
            for k in unique_keys:
                # Search for "key" or 'key'
                if f'"{k}"' in f_content or f"'{k}'" in f_content:
                    usage_count[k] += 1
                    usage_details[k].append(os.path.basename(f_path))

    # 4. Report
    print("\n[UNUSED KEYS REPORT]")
    print("-" * 50)
    unused = []
    for k in unique_keys:
        if usage_count[k] == 0:
            unused.append(k)
            print(f"❌ {k}")
    
    print("-" * 50)
    print(f"Summary: {len(unused)} unused keys found out of {len(unique_keys)}.")

if __name__ == "__main__":
    audit_config_keys()
