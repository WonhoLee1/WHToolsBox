import os

def final_safe_rename():
    files_to_update = [
        r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_drop_simulation.py",
        r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_discrete_builder\__init__.py",
        r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\dev_log\walkthrough_20260325.md",
        r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\dev_log\task_20260325.md"
    ]
    
    replacements = {
        'cush_edge_solref': 'cush_corner_solref',
        'cush_edge_solimp': 'cush_corner_solimp',
        'edge_solref': 'corner_solref',  # Internal properties too
        'edge_solimp': 'corner_solimp'
    }
    
    for file_path in files_to_update:
        if not os.path.exists(file_path):
            continue
            
        print(f"Updating {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = content
            for old, new in replacements.items():
                new_content = new_content.replace(old, new)
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"  Successfully updated.")
            else:
                print(f"  No changes needed.")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    final_safe_rename()
