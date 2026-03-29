import mujoco
xml = """
<mujoco>
    <worldbody>
        <geom name="gpos" type="sphere" size="1" solref="200000 200"/>
        <geom name="gneg" type="sphere" size="1" solref="-200000 -200"/>
    </worldbody>
</mujoco>
"""
try:
    model = mujoco.MjModel.from_xml_string(xml)
    print("Positive solref:", model.geom_solref[0])
    print("Negative solref:", model.geom_solref[1])
except Exception as e:
    print(e)
