"""
Unified VTKHDF Converter for CalculiX and OpenRadioss
Assisted by WHTOOLs Calculixent & OpenRadiossent
"""
import os
import subprocess
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
import logging

try:
    import vtk
    from vtkmodules.vtkIOMinimalEnsemble import vtkVTKHDFWriter
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
    from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
except ImportError:
    print("Warning: VTK 패키지를 찾을 수 없습니다. (pip install vtk)")

class NativeFRDToVTKHDFConverter:
    @staticmethod
    def parse_frd_and_export_vtkhdf(frd_path: str, output_vtkhdf: str):
        """
        CalculiX FRD 파일을 읽어 VTKHDF로 다이렉트 저장합니다.
        (현재는 Hexa20 등 2차 요소에 최적화된 단일 스텝 데모 구현입니다)
        """
        if not os.path.exists(frd_path):
            raise FileNotFoundError(f"FRD 파일을 찾을 수 없습니다: {frd_path}")

        print(f"Parsing {frd_path} and writing directly to {output_vtkhdf} using native VTK...")
        
        data = {}
        with open(frd_path, 'r', encoding='utf-8', errors='ignore') as f:
            line = f.readline()
            while line:
                if line.startswith('    1C'):
                    data['calc name'] = line[6:].strip()
                elif line.startswith('    2C'):
                    number_of_nodes = int(line[7:37])
                    data['nodes'] = []
                    for _ in range(number_of_nodes):
                        line = f.readline()
                        if line.startswith(' -1'):
                            node_number = int(line[3:13])
                            node_x = float(line[13:25])
                            node_y = float(line[25:37])
                            node_z = float(line[37:49])
                            data['nodes'].append([node_number, [node_x, node_y, node_z]])
                    line = f.readline() # skip -3 
                elif line.startswith('    3C'):
                    number_of_elements = int(line[7:37])
                    line = f.readline()
                    data['elements'] = []
                    for _ in range(number_of_elements):
                        if line.startswith(' -1'):
                            element_number = int(line[3:13])
                            element_type = int(line[13:18])
                            node_numbers = []
                            line = f.readline()
                            while line.startswith(' -2'):
                                for i in range(3, min(len(line)-1, 103), 10):
                                    val = line[i:i+10].strip()
                                    if val:
                                        node_numbers.append(int(val))
                                line = f.readline()
                            data['elements'].append([element_number, element_type, node_numbers])
                elif line.startswith('  100C'):
                    number_of_values = int(line[24:36])
                    line = f.readline()
                    if line.startswith(' -4'):
                        name = line[3:17].strip()
                        number_of_components = 0
                        line = f.readline()
                        while line.startswith(' -5'):
                            if line[5:8] != 'ALL':
                                number_of_components += 1
                            line = f.readline()
                        
                        array = {name: []}
                        for _ in range(number_of_values):
                            if line.startswith(' -1'):
                                node_number = int(line[3:13])
                                values = []
                                for comp in range(number_of_components):
                                    val = float(line[(12*comp+13):(12*(comp+1)+13)])
                                    values.append(val)
                                array[name].append([node_number, values])
                            line = f.readline()
                        data['arrays'] = array
                        break # Parse 1 timestep for simplicity
                line = f.readline()

        my_vtk_dataset = vtkUnstructuredGrid()
        points = vtkPoints()
        node_map = {}
        
        for i, (nid, coords) in enumerate(data.get('nodes', [])):
            points.InsertPoint(i, coords)
            node_map[nid] = i
        my_vtk_dataset.SetPoints(points)
        
        elements = data.get('elements', [])
        my_vtk_dataset.Allocate(len(elements))
        
        for eid, etype, enodes in elements:
            if etype == 4: # HEX20 in CalculiX
                mapped_nodes = [node_map[n] for n in enodes if n in node_map]
                if len(mapped_nodes) == 20:
                    node_numbers = [
                        mapped_nodes[0], mapped_nodes[1], mapped_nodes[2], mapped_nodes[3],
                        mapped_nodes[4], mapped_nodes[5], mapped_nodes[6], mapped_nodes[7],
                        mapped_nodes[8], mapped_nodes[9], mapped_nodes[10], mapped_nodes[11],
                        mapped_nodes[16], mapped_nodes[17], mapped_nodes[18], mapped_nodes[19],
                        mapped_nodes[12], mapped_nodes[13], mapped_nodes[14], mapped_nodes[15]
                    ]
                    # 28 is VTK_QUADRATIC_HEXAHEDRON
                    my_vtk_dataset.InsertNextCell(28, 20, node_numbers)

        if 'arrays' in data:
            for name, array_values in data['arrays'].items():
                vtk_array = vtkDoubleArray()
                vtk_array.SetNumberOfComponents(len(array_values[0][1]))
                vtk_array.SetNumberOfTuples(points.GetNumberOfPoints())
                vtk_array.SetName(name)
                for nid, vals in array_values:
                    if nid in node_map:
                        idx = node_map[nid]
                        vtk_array.SetTuple(idx, vals)
                my_vtk_dataset.GetPointData().AddArray(vtk_array)

        try:
            writer = vtkVTKHDFWriter()
            writer.SetFileName(output_vtkhdf)
            writer.SetInputData(my_vtk_dataset)
            writer.Write()
            print(f"VTKHDF 저장 완료: {output_vtkhdf}")
        except Exception as e:
            print(f"VTKHDF 변환 실패: {e}")
            print("현재 설치된 VTK 모듈이 vtkVTKHDFWriter를 지원하지 않을 수 있습니다. (VTK 9.2 이상 권장)")

class PVDToVTKHDFConverter:
    """
    Convert a ParaView PVD + VTU time series to a transient VTKHDF file.

    Supports static mesh topology with time-varying PointData (e.g. CalculiX
    implicit dynamic results from ccx2paraview).

    Key format note (VTK source vtkHDFUtilities.txx):
      VTKHDF/Steps/NSteps  must be an HDF5 *attribute*, not a dataset.
      Writing it as a dataset causes "NSteps attribute not found" at load time.

    Usage::
        PVDToVTKHDFConverter.convert("workspace/job.pvd", "workspace/job.vtkhdf")
    """

    @staticmethod
    def convert(pvd_path: str | Path, output_vtkhdf: str | Path) -> bool:
        """
        Read PVD index, load each VTU timestep, write transient VTKHDF.

        Parameters
        ----------
        pvd_path       : path to the .pvd file produced by ccx2paraview
        output_vtkhdf  : destination .vtkhdf path

        Returns
        -------
        bool : True on success, False on failure
        """
        pvd_path      = Path(pvd_path)
        output_vtkhdf = Path(output_vtkhdf)

        try:
            import h5py
            import numpy as np
            from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
        except ImportError as exc:
            print(f"[PVDToVTKHDF] Missing dependency: {exc}")
            return False

        # ── Parse PVD ────────────────────────────────────────────────────────
        tree       = ET.parse(pvd_path)
        collection = tree.getroot().find("Collection")
        if collection is None:
            print("[PVDToVTKHDF] Malformed PVD: no <Collection> element.")
            return False

        entries = sorted(
            [
                (float(ds.get("timestep", 0.0)),
                 pvd_path.parent / ds.get("file", ""))
                for ds in collection.findall("DataSet")
            ],
            key=lambda x: x[0],
        )
        if not entries:
            print("[PVDToVTKHDF] No <DataSet> entries found in PVD.")
            return False

        nsteps = len(entries)
        times  = np.array([t for t, _ in entries], dtype=np.float64)
        print(f"[PVDToVTKHDF] {nsteps} timesteps  |  t=[{times[0]:.3f}, {times[-1]:.3f}] s")

        # ── Read static mesh topology from first VTU ──────────────────────────
        def _read_vtu(path: Path):
            rdr = vtkXMLUnstructuredGridReader()
            rdr.SetFileName(str(path))
            rdr.Update()
            return rdr.GetOutput()

        g0    = _read_vtu(entries[0][1])
        npts  = g0.GetNumberOfPoints()
        ncells = g0.GetNumberOfCells()

        pts = np.array([g0.GetPoint(i) for i in range(npts)], dtype=np.float64)

        conn, offs, types = [], [0], []
        for ci in range(ncells):
            cell = g0.GetCell(ci)
            ids  = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            conn.extend(ids)
            offs.append(len(conn))
            types.append(g0.GetCellType(ci))

        conn  = np.array(conn,  dtype=np.int64)
        offs  = np.array(offs,  dtype=np.int64)
        types = np.array(types, dtype=np.uint8)

        pd0         = g0.GetPointData()
        field_names = [pd0.GetArrayName(i) for i in range(pd0.GetNumberOfArrays())]

        # ── Collect transient field data ──────────────────────────────────────
        fd: dict[str, list] = {n: [] for n in field_names}
        for step_idx, (_t, vtu_file) in enumerate(entries):
            g  = _read_vtu(vtu_file)
            pd = g.GetPointData()
            for name in field_names:
                arr = pd.GetArray(name)
                if arr is not None:
                    nc   = arr.GetNumberOfComponents()
                    data = np.array(
                        [[arr.GetComponent(i, c) for c in range(nc)]
                         for i in range(npts)],
                        dtype=np.float64,
                    )
                else:
                    data = np.zeros((npts, 1), dtype=np.float64)
                fd[name].append(data)
            if (step_idx + 1) % 20 == 0:
                print(f"[PVDToVTKHDF]   loaded {step_idx + 1}/{nsteps} steps…")

        # ── Write VTKHDF ──────────────────────────────────────────────────────
        with h5py.File(output_vtkhdf, "w") as f:
            grp = f.create_group("VTKHDF")
            grp.attrs["Type"]    = np.bytes_("UnstructuredGrid")
            grp.attrs["Version"] = np.array([2, 0], dtype=np.int64)

            grp["Points"]                  = pts
            grp["Connectivity"]            = conn
            grp["Offsets"]                 = offs
            grp["Types"]                   = types
            grp["NumberOfPoints"]          = np.array([npts],      dtype=np.int64)
            grp["NumberOfCells"]           = np.array([ncells],    dtype=np.int64)
            grp["NumberOfConnectivityIds"] = np.array([len(conn)], dtype=np.int64)

            # Steps group
            # NSteps MUST be an HDF5 attribute (not dataset).
            # vtkHDFUtilities::GetAttribute() looks for attrs, not datasets.
            sg = grp.create_group("Steps")
            sg.attrs["NSteps"]          = np.int64(nsteps)
            sg["TimeValues"]            = times
            sg["NumberOfParts"]         = np.ones(nsteps,  dtype=np.int64)
            sg["PartOffsets"]           = np.zeros(nsteps, dtype=np.int64)
            sg["PointOffsets"]          = np.zeros(nsteps, dtype=np.int64)
            sg["CellOffsets"]           = np.zeros(nsteps, dtype=np.int64)
            sg["ConnectivityIdOffsets"] = np.zeros(nsteps, dtype=np.int64)

            # PointData: each field stored in a subgroup containing "Values".
            # vtkHDFUtilities.cxx does H5Dopen(group, "Values") — a direct
            # dataset under PointData/ causes "Cannot open Values".
            if field_names:
                pdg  = grp.create_group("PointData")
                spd  = sg.create_group("PointData")
                spdo = spd.create_group("Offsets")
                for name in field_names:
                    if fd[name]:
                        cat = np.concatenate(fd[name], axis=0)
                        # Each field: PointData/{name}/Values
                        pdg.create_group(name)["Values"] = cat
                        # per-step start index within the concatenated array
                        spdo[name] = np.arange(nsteps + 1, dtype=np.int64) * npts

        print(f"[PVDToVTKHDF] Written: {output_vtkhdf}  ({nsteps} steps, {npts} nodes)")
        return True


class RadiossVTKHDFConverter:
    @staticmethod
    def convert_anim_to_vtkhdf(anim_prefix: str, output_vtkhdf: str, static_mesh: bool = True):
        """
        OpenRadioss Anim 파일을 단일 VTKHDF 파일로 변환합니다.
        """
        anim_files = sorted(glob(f"{anim_prefix}[0-9][0-9]*"))
        if not anim_files:
            raise FileNotFoundError(f"'{anim_prefix}'로 시작하는 Anim 파일을 찾을 수 없습니다.")
            
        print(f"총 {len(anim_files)}개의 Anim 파일을 {output_vtkhdf}로 변환 시작...")
        
        cmd = ["python", "-m", "animtovtkhdf"]
        if not static_mesh:
            cmd.append("--nostatic")
            
        cmd.extend(anim_files)
        cmd.append(output_vtkhdf)
        
        # Windows PowerShell UTF-8 출력 처리 대응
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8', env=env)
            print(f"VTKHDF 변환 완료: {output_vtkhdf}")
            if result.stdout.strip():
                print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"VTKHDF 변환 실패: {e.stderr}")
            raise
