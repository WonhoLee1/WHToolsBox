import mujoco
xml_str = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE mujoco [
    <!ENTITY commonWeld 'solref="0.05 1.0"'>
]>
<mujoco>
  <worldbody>
    <body name="b1" pos="0 0 1"><geom size="0.1"/><site name="s1"/></body>
    <body name="b2" pos="0 0 2"><geom size="0.1"/><site name='s2'/></body>
  </worldbody>
  <equality>
    <weld site1="s1" site2="s2" &commonWeld; />
  </equality>
</mujoco>
"""
try:
    m = mujoco.MjModel.from_xml_string(xml_str)
    print("SUCCESS: ENTITY SUPPORTED")
except Exception as e:
    print("FAILURE: ENTITY NOT SUPPORTED")
    print(e)
