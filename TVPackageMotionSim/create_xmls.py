import pandas as pd
import os

# Create output dir
output_dir = r"D:\PythonCodeStudy\mujoco-3.5.0-windows-x86_64\model\WH_boxdrop"
os.makedirs(output_dir, exist_ok=True)

# Read CSV
df = pd.read_csv("sphere_damping_mapping.csv", index_col="Material")

# Target CORs
targets = [0.2, 0.5, 0.8]

# Colors for visualization
colors = {
    "Steel": "0.3 0.3 0.3 1",
    "Aluminum": "0.8 0.8 0.8 1",
    "Copper": "0.8 0.4 0.1 1",
    "Glass": "0.5 0.8 0.9 0.5",
    "Plastic_PC": "0.1 0.3 0.8 1",
    "TPU": "0.8 0.1 0.1 1",
    "Hard_Rubber": "0.1 0.1 0.1 1",
    "Soft_Rubber": "0.2 0.2 0.2 1",
    "Paper": "0.9 0.9 0.9 1",
    "EPS_Foam": "0.95 0.95 0.8 1",
    "PSA_Adhesive": "0.9 0.9 0.9 0.5"
}

for target in targets:
    col_name = f"COR_{target:.1f}"
    
    # 이제 K 강성이 Softening (스케일다운) 되었으므로 실무적인 일반 0.001초 timestep을 사용합니다.
    xml_content = f"""<mujoco model="Material_COR_{target:.1f}_Test">
    <option integrator="implicitfast" timestep="0.001"/>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    </asset>

    <visual>
        <map force="0.1" zfar="30"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <scale forcewidth="0.05" contactwidth="0.1" contactheight="0.05"/>
    </visual>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="5 5 0.05" material="matplane" solref="0.001 0"/>
"""
    
    for i, (mat_name, row) in enumerate(df.iterrows()):
        p1 = row["p1_Value"]
        p2 = -row[col_name]
        density = row["Rho (kg/m3)"]
        
        # Position them in a line along X axis
        x_pos = (i - len(df)/2) * 0.1
        y_pos = 0.0
        z_pos = 0.5 # Drop from 0.5m
        
        color = colors.get(mat_name, "0.5 0.5 0.5 1")
        
        xml_content += f"""
        <body name="{mat_name}_body" pos="{x_pos:.3f} {y_pos:.3f} {z_pos:.3f}">
            <freejoint name="{mat_name}_joint"/>
            <geom name="{mat_name}_geom" type="sphere" size="0.01" density="{density:.1f}" 
                  solref="{p1:.1f} {p2:.4f}" solimp="0.8 0.95 0.001" solmix="1000" rgba="{color}"/>
        </body>"""
        
    xml_content += """
    </worldbody>
</mujoco>
"""
    
    file_path = os.path.join(output_dir, f"Material_Test_COR_{target:.1f}.xml")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    print(f"Created: {file_path}")
