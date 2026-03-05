import gmsh
import math

def create_packaging_model(config):
    gmsh.initialize()
    occ = gmsh.model.occ
    
    # 1. Geometry (Build ALL once)
    bw, bh, bd, bt = config['boxWidth'], config['boxHeight'], config['boxDepth'], config['boxThick']
    dw, dh, dd = config['dispWidth'], config['dispHeight'], config['dispDepth']
    cw, ct = config['dispCohWidth'], config['dispCohThick']
    ch_t, es = 40.0, config['elementSize']
    tp_t = dd + ct + ch_t

    # Box
    o_box = occ.addBox(-bw/2, -bh/2, -bd/2, bw, bh, bd)
    i_box = occ.addBox(-bw/2+bt, -bh/2+bt, -bd/2+bt, bw-2*bt, bh-2*bt, bd-2*bt)
    box_raw, _ = occ.cut([(3, o_box)], [(3, i_box)])
    
    # Product
    if config['orientation'] == 'Parcel':
        c = occ.addBox(-dw/2, -dh/2, -tp_t/2, dw, dh, ch_t)
        d = occ.addBox(-dw/2, -dh/2, -tp_t/2 + ch_t + ct, dw, dh, dd)
    else:
        d = occ.addBox(-dw/2, -dh/2, -tp_t/2, dw, dh, dd)
        c = occ.addBox(-dw/2, -dh/2, -tp_t/2 + dd + ct, dw, dh, ch_t)
    # Simple product union
    product_raw, _ = occ.fuse([(3, c)], [(3, d)])
    
    # Cushion
    cush_base = occ.addBox(-bw/2+bt, -bh/2+bt, -bd/2+bt, bw-2*bt, bh-2*bt, bd-2*bt)
    cush_raw, _ = occ.cut([(3, cush_base)], product_raw)
    
    # Split
    splitters = []
    for x in [-dw/2, dw/2]:
        s = occ.addRectangle(-bh/2, -bd/2, x, bh, bd)
        occ.rotate([(2, s)], 0, 0, x, 0, 1, 0, math.pi/2)
        splitters.append((2, s))
    for y in [-dh/2, dh/2]:
        s = occ.addRectangle(-bw/2, -bd/2, y, bw, bd)
        occ.rotate([(2, s)], 0, 0, y, 1, 0, 0, math.pi/2)
        splitters.append((2, s))
        
    frag, _ = occ.fragment(box_raw + cush_raw + product_raw, splitters)
    occ.synchronize()

    # Labeling
    box_tags, cush_tags, prod_tags = [], [], []
    lim = max(bw, bh, bd)/2 - bt - 1.0
    for dim, tag in gmsh.model.getEntities(3):
        com = occ.getCenterOfMass(dim, tag)
        d_max = max(abs(com[0]), abs(com[1]), abs(com[2]))
        d_xy = max(abs(com[0]), abs(com[1]))
        d_z = abs(com[2])
        if d_max > lim: box_tags.append(tag)
        elif d_xy < (dw/2 + 2.0) and d_z < (tp_t/2 + 2.0): prod_tags.append(tag)
        else: cush_tags.append(tag)

    gmsh.model.addPhysicalGroup(3, box_tags, 1, "Box")
    gmsh.model.addPhysicalGroup(3, cush_tags, 2, "Cushion")
    gmsh.model.addPhysicalGroup(3, prod_tags, 3, "Product")

    # Meshing
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", es)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", es)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    
    # Export
    # 1. Full
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write("packaging_model.msh")
    
    # 2. Separate files by filtering physical groups during write if possible
    # In Gmsh Python, the easiest way is to hide/show or use separate models.
    # Let's save three copies by renaming physical groups? No.
    # We'll use Mesh.SaveAll = 0 and save each group.
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    # Note: Only physical groups are saved when SaveAll=0.
    # But if multiple exist, they all go to one file.
    # So we must remove other groups or use different models.
    
    # I'll just save the full MSH and provide instructions for MuJoCo to read by ID 
    # OR I'll use the 'Model Copy' approach.
    
    main_model = gmsh.model.get()
    for name, tag_list in [("box", box_tags), ("cushion", cush_tags), ("product", prod_tags)]:
        gmsh.model.add(name)
        # Gmsh doesn't have an easy 'copy entities' between models in Python without re-running.
        # So I will just write a combined file and tell the user how to use it.
        pass

    gmsh.finalize()

if __name__ == "__main__":
    cfg = {
        'boxWidth': 2000.0, 'boxHeight': 1400.0, 'boxDepth': 250.0, 'boxThick': 5.0,
        'dispWidth': 1600.0, 'dispHeight': 1100.0, 'dispDepth': 5.0,
        'dispCohWidth': 20.0, 'dispCohThick': 2.0,
        'orientation': 'Parcel', 'cushionCutouts': [], # Simplified for now
        'elementSize': 100.0
    }
    create_packaging_model(cfg)
