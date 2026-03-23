import mujoco
import numpy as np

def run_xml(path, target):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    max_hs = {i: 0.0 for i in range(11)}
    contacts = {i: False for i in range(11)}
    rebounded = {i: False for i in range(11)}
    
    # run for 1 second
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        for i in range(11):
            h = data.qpos[i*7 + 2] - 0.010
            if not contacts[i] and h < 0.010:
                contacts[i] = True
            
            if contacts[i]:
                if h > max_hs[i]: max_hs[i] = h
                if data.qvel[i*6 + 2] > 0.05: rebounded[i] = True
    
    print(f"Target: {target}, Timestep: {model.opt.timestep}")
    materials = ["Steel", "Aluminum", "Copper", "Glass", "Plastic_PC", "TPU", "Hard_Rubber", "Soft_Rubber", "Paper", "EPS_Foam", "PSA_Adhesive"]
    for k, v in max_hs.items():
        cor = np.sqrt(max(0, v) / 0.49) # Drop from 0.5 center means 0.49 distance to surface
        print(f"{materials[k]:12} COR: {cor:.3f} | max_h: {v:.4f}")

run_xml(r"D:\PythonCodeStudy\mujoco-3.5.0-windows-x86_64\model\WH_boxdrop\Material_Test_COR_0.5.xml", 0.5)

