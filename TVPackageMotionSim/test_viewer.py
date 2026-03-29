import mujoco
import mujoco.viewer
import sys
import os

def load_and_view(xml_path):
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        sys.exit(1)
        
    print(f"Loading {xml_path}...")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        print("Model loaded successfully. Starting viewer...")
        mujoco.viewer.launch(model, data)
        
    except Exception as e:
        print(f"Failed to load Mujoco model: {e}")

if __name__ == "__main__":
    xml_path = r"c:\Users\GOODMAN\WHToolsBox\test_box_mujoco\test_shapes_check.xml"
    load_and_view(xml_path)
