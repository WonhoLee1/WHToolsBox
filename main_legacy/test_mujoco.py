import mujoco
import mujoco.viewer
import os

# Load the model from your XML file
xml_path = r"flex.xml"
# Ensure the file exists
if not os.path.exists(xml_path):
    print(f"Error: {xml_path} not found")
else:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Launch the interactive viewer
    with mujoco.viewer.launch_viewer(model, data) as viewer:
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            # Sync the viewer
            viewer.sync()

