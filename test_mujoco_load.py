"""Test _fix_msh41_single_block + MuJoCo flexcomp load."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

# Generate fresh MSH 4.1 for box
import gmsh
gmsh.initialize()
gmsh.model.add('BoxTest')
occ = gmsh.model.occ
outer = occ.addBox(-100, -70, -12.5, 200, 140, 25)
inner = occ.addBox(-95, -65, -7.5, 190, 130, 15)
occ.synchronize()
result, _ = occ.cut([(3,outer)], [(3,inner)], removeObject=True, removeTool=True)
occ.synchronize()
gmsh.option.setNumber('Mesh.Algorithm3D', 4)
gmsh.option.setNumber('Mesh.CharacteristicLengthMax', 40.0)
gmsh.option.setNumber('Mesh.MshFileVersion', 4.1)
gmsh.option.setNumber('Mesh.SaveAll', 1)
gmsh.option.setNumber('Mesh.Binary', 0)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.removeDuplicateNodes()
gmsh.model.removePhysicalGroups()
gmsh.write('test_box_msh/box_test_fix.msh')
gmsh.finalize()
print('Written box_test_fix.msh')

# Check before fix
with open('test_box_msh/box_test_fix.msh', 'r') as f:
    lines = f.readlines()
for line in lines:
    if line.strip() == '$Nodes':
        idx = lines.index(line)
        print(f'BEFORE fix - Nodes header: {lines[idx+1].strip()}')
        break

# Apply fix
from tv_packaging_gmsh import _fix_msh41_single_block
_fix_msh41_single_block('test_box_msh/box_test_fix.msh')

# Check after fix
with open('test_box_msh/box_test_fix.msh', 'r') as f:
    lines = f.readlines()
for line in lines:
    if line.strip() == '$Nodes':
        idx = lines.index(line)
        print(f'AFTER fix  - Nodes header: {lines[idx+1].strip()}')
        print(f'             Block header: {lines[idx+2].strip()}')
        break

# MuJoCo load test
import mujoco
os.chdir('test_box_msh')
xml = (
    '<mujoco><compiler meshdir="."/><worldbody>'
    '<body name="b"><freejoint/>'
    '<flexcomp name="f" type="gmsh" file="box_test_fix.msh" dim="3">'
    '<edge stiffness="5000" damping="10"/>'
    '<contact condim="3" selfcollide="none"/>'
    '</flexcomp></body></worldbody></mujoco>'
)
try:
    m = mujoco.MjModel.from_xml_string(xml)
    print(f'flexcomp OK: nflex={m.nflex}, nflexvert={m.nflexvert}')
except Exception as e:
    print(f'ERROR: {e}')
