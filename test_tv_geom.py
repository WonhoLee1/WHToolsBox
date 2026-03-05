"""Quick geometry test for tv_packaging_gmsh.py"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import gmsh

gmsh.initialize()
gmsh.model.add('Test')
occ = gmsh.model.occ

try:
    # Box hollow
    outer = occ.addBox(-1000, -700, -125, 2000, 1400, 250)
    inner = occ.addBox(-995, -695, -120, 1990, 1390, 240)
    occ.synchronize()
    box_res, _ = occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
    occ.synchronize()
    print('Box hollow OK, vols:', [t for d,t in box_res if d==3])

    # Z positions (Parcel, contentCenter=0)
    chD, cohT, dD = 40.0, 2.0, 5.0
    total = chD + cohT + dD  # 47
    half = total / 2.0       # 23.5
    cz = 0.0
    z_ch  = cz - half + chD / 2.0
    z_coh = cz - half + chD + cohT / 2.0
    z_dp  = cz - half + chD + cohT + dD / 2.0
    print(f'Z: chassis={z_ch:.2f}, coh={z_coh:.2f}, disp={z_dp:.2f}')

    # Chassis
    chassis = occ.addBox(-950, -650, z_ch - chD/2, 1900, 1300, chD)
    occ.synchronize()
    print('Chassis OK, tag:', chassis)

    # Display
    disp = occ.addBox(-950, -650, z_dp - dD/2, 1900, 1300, dD)
    occ.synchronize()
    print('Display OK, tag:', disp)

    # DispCoh (picture frame)
    cohW = 20.0
    coh_outer = occ.addBox(-950, -650, z_coh - cohT/2, 1900, 1300, cohT)
    coh_inner = occ.addBox(-950+cohW, -650+cohW, z_coh - cohT/2, 1900-2*cohW, 1300-2*cohW, cohT)
    occ.synchronize()
    coh_res, _ = occ.cut([(3, coh_outer)], [(3, coh_inner)], removeObject=True, removeTool=True)
    occ.synchronize()
    print('DispCoh OK, vols:', [t for d,t in coh_res if d==3])

    # Cushion
    cush = occ.addBox(-995, -695, -120, 1990, 1390, 240)
    occ.synchronize()
    tools = []
    for ent in [(3, chassis), (3, disp)] + coh_res:
        tools.extend(occ.copy([ent]))
    occ.synchronize()
    cush_res, _ = occ.cut([(3, cush)], tools, removeObject=True, removeTool=True)
    occ.synchronize()
    print('Cushion OK, vols:', [t for d,t in cush_res if d==3])

    all_vols = gmsh.model.getEntities(3)
    print(f'Total volumes: {len(all_vols)}')
    print('ALL GEOMETRY OK')

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'ERROR: {e}')

gmsh.finalize()
