import os
import time
import mujoco
import mujoco.viewer

# Dummy model
xml = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size=".1" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Try environment variables (often MJ_VIEWER_WIDTH/HEIGHT or MUJOCO_WINDOW_WIDTH/HEIGHT)
os.environ["MJ_VIEWER_WIDTH"] = "800"
os.environ["MJ_VIEWER_HEIGHT"] = "600"

print("Launching viewer...")
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"Viewer launched. Please observe window size.")
        time.sleep(5)
        viewer.close()
except Exception as e:
    print(f"Error: {e}")
