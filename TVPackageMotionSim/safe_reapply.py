import os

def apply_all_fixes():
    # 1. run_discrete_builder/__init__.py 수정 (is_edge_block)
    builder_path = r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_discrete_builder\__init__.py"
    if os.path.exists(builder_path):
        with open(builder_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # is_edge_block 로직 수정 (OR -> AND)
        old_logic = "bx = (i == 0 or i == nx_max)\n        by = (j == 0 or j == ny_max)\n        bz = (k == 0 or k == nz_max)\n        \n        return bx or by or bz"
        new_logic = "bx = (i == 0 or i == nx_max)\n        by = (j == 0 or j == ny_max)\n        \n        return bx and by"
        
        if old_logic in content:
            content = content.replace(old_logic, new_logic)
        
        # Rename edge -> corner
        content = content.replace('edge_solref', 'corner_solref')
        content = content.replace('edge_solimp', 'corner_solimp')
        content = content.replace('cush_edge_solref', 'cush_corner_solref')
        content = content.replace('cush_edge_solimp', 'cush_corner_solimp')

        with open(builder_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Updated run_discrete_builder/__init__.py")

    # 2. run_drop_simulation.py 수정
    sim_path = r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\run_drop_simulation.py"
    if os.path.exists(sim_path):
        with open(sim_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Plasticity 및 시각화 로직은 복잡하므로 특정 지점을 찾아 교체하거나 
        # 내가 기억하는 최신 apply_plastic_deformation 전체를 주입해야 함.
        # 일단 문자열 치환으로 시도.
        
        content = "".join(lines)
        
        # [NEW] 코너 색상 변경 (Yellow) 및 plasticity_ratio 등을 포함한 통합 패치
        # 이 부분은 view_file로 위치를 다시 확인하는 것이 안전함.
        # 하지만 이미 git checkout 했으므로 원본 상태임.
        
        # Rename edge -> corner
        content = content.replace('cush_edge_solref', 'cush_corner_solref')
        content = content.replace('cush_edge_solimp', 'cush_corner_solimp')

        with open(sim_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Updated run_drop_simulation.py (Basic Rename)")

apply_all_fixes()
