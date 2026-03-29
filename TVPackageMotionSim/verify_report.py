
import sys
import os
sys.path.append(r'c:\Users\GOODMAN\WHToolsBox\test_box_mujoco')
import run_discrete_builder

cfg = run_discrete_builder.get_default_config()
cfg["chassis_aux_masses"] = [
    {"name": "UpperWeight", "pos": [0, 0.5, 0], "size": [0.2, 0.1, 0.1], "mass": 10.0},
    {"name": "SideWeight", "pos": [0.5, 0, 0], "size": [0.1, 0.2, 0.1], "mass": 5.0}
]
cfg["drop_mode"] = "PARCEL"
cfg["include_paperbox"] = True
cfg["include_cushion"] = True

output_file = r'c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\test_report_gen.xml'
run_discrete_builder.create_model(output_file, config=cfg)
