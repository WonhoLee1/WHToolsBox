import mujoco
xml1 = """<mujoco><worldbody><geom name="g" type="sphere" size="1" solref="-200000 0.0"/></worldbody></mujoco>"""
xml2 = """<mujoco><worldbody><geom name="g" type="sphere" size="1" solref="-200000 -0.0"/></worldbody></mujoco>"""
xml3 = """<mujoco><worldbody><geom name="g" type="sphere" size="1" solref="-200000 -0.0001"/></worldbody></mujoco>"""

for i, x in enumerate([xml1, xml2, xml3]):
    try:
        m = mujoco.MjModel.from_xml_string(x)
        print(f"xml{i+1}:", m.geom_solref[0])
    except Exception as e:
        print(f"xml{i+1}: error", e)
