import os

def fix_and_rename(file_path):
    print(f"Processing {file_path}...")
    try:
        # Read with utf-8-sig to handle UTF-8 with/without BOM
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            
        # Perform replacements
        new_content = content.replace('cush_edge_solref', 'cush_corner_solref')
        new_content = new_content.replace('cush_edge_solimp', 'cush_corner_solimp')
        
        # Write back with utf-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")

files = [
    r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_drop_simulation.py",
    r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_discrete_builder\__init__.py",
    r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\dev_log\walkthrough_20260325.md"
]

for f in files:
    if os.path.exists(f):
        fix_and_rename(f)
