import sys
import os
import mujoco

# Add current directory to path to import the module
sys.path.append(os.getcwd())

from mujoco_secvd_boxmotionsim_v0_0_3 import BoxDropInstance, SimulationManager, CONFIG

def test_xml_generation():
    print("Testing XML Generation...")
    
    # Initialize a simulation with one box
    sim = SimulationManager(enable_air_cushion=True, enable_plasticity=True, solver_mode="2")
    
    inst = BoxDropInstance(
        uid=0,
        position_offset=[0, 0, 0],
        drop_type="corner",
        box_params={'L': 1.8, 'W': 1.2, 'H': 0.22, 'MASS': 30.0},
        label="Test Box"
    )
    
    sim.add_instance(inst)
    
    # Generate XML
    xml_str = sim.generate_full_xml()
    
    # Save to failed_model.xml for inspection
    with open("failed_model.xml", "w", encoding='utf-8') as f:
        f.write(xml_str)
    
    print("XML dumped to failed_model.xml")
    
    # Try to load it into MuJoCo to check for syntax errors
    try:
        model = mujoco.MjModel.from_xml_string(xml_str)
        print("✅ SUCCESS: XML is valid MuJoCo MJCF!")
    except Exception as e:
        print(f"❌ FAILURE: XML is invalid: {e}")

if __name__ == "__main__":
    test_xml_generation()
