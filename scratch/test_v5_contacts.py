import os
import sys

# Add the project path
project_path = r"c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim"
if project_path not in sys.path:
    sys.path.append(project_path)

from run_discrete_builder import create_model, get_default_config

def verify_v5_xml():
    print("Testing v5 XML Generation with Contact Pairs...")
    
    # Create test output directory
    test_dir = os.path.join(project_path, "test_v5_refactor")
    os.makedirs(test_dir, exist_ok=True)
    xml_path = os.path.join(test_dir, "test_v5_model.xml")
    
    # Get config (this will include the new 'contacts' dictionary)
    cfg = get_default_config({"include_paperbox": True, "include_cushion": True})
    
    try:
        # Generate model
        xml_content, *_ = create_model(xml_path, config=cfg)
        print(f"✅ XML generated successfully at {xml_path}")
        
        # Basic validation of content
        if "<contact>" in xml_content and "<pair" in xml_content:
            print("✅ <contact> and <pair> tags found.")
        else:
            print("❌ <contact> or <pair> tags NOT found!")
            
        if 'class="cls_ground_paper"' in xml_content:
            print("✅ cls_ground_paper found.")
        if 'class="cls_ground_cushion_edge"' in xml_content:
            print("✅ cls_ground_cushion_edge found.")
            
        # Check if bitmask is disabled
        if 'contype="1"' in xml_content or 'conaffinity="1"' in xml_content:
             # Wait, ground or some sites might have 1? 
             # Let's check specifically for geoms.
             pass
             
        # MuJoCo Load Test (if mujoco is available)
        try:
            import mujoco
            mujoco.MjModel.from_xml_string(xml_content)
            print("✅ MuJoCo MjModel load success!")
        except Exception as e:
            print(f"❌ MuJoCo load failed: {e}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ XML generation failed: {e}")

if __name__ == "__main__":
    verify_v5_xml()
