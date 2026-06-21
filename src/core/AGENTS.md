<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-04 | Updated: 2026-06-04 -->

# src/core

## Purpose
Six single-responsibility modules that implement each step of the AutoCalculix pipeline, plus a configuration module. Each class owns exactly one pipeline stage and communicates through file paths in `workspace/`.

## Key Files

| File | Description |
|------|-------------|
| `config.py` | `CALCULIX_EXE` path, `WORKSPACE_DIR`, `ModalAnalysisConfig` and `TrayGeometryConfig` dataclasses |
| `mesher.py` | `GmshMesher` — creates rectangular tray geometry, meshes it with Gmsh, converts 2D plane elements → CalculiX shell elements (S3/S4), exports `mesh.inp` |
| `model_builder.py` | `CalculixModelBuilder` — writes the master INP file with `*INCLUDE`, material card, shell section, and `*FREQUENCY` step |
| `solver.py` | `CalculixSolver` — invokes `ccx_static.exe` as a subprocess and monitors completion |
| `dat_parser.py` | `CalculixDatParser` — parses the `.dat` text output to extract eigenfrequency list `[{"mode": int, "hz": float}]` |
| `frd_converter.py` | `FrdToVtuConverter` — calls `ccx2paraview` to convert `.frd` → per-mode `.NN.vtu` files (1-based, 2-digit) |
| `viewer.py` | `ModeShapeViewer` — loads a VTU file, warps by displacement vector scaled to 10% of max model dimension, opens PyVista window |
| `mesh_loader.py` | `ExternalMeshLoader` — reads Abaqus INP or OptiStruct/Nastran FEM/BDF files and normalises them to a CalculiX-compatible mesh INP |

## For AI Agents

### Working In This Directory
- All units are mm / MPa / ton — do not mix unit systems
- `config.py` is the single source of truth for the CalculiX executable path and default material properties (Steel: E=210 GPa, ν=0.3, ρ=7.85×10⁻⁹ t/mm³)
- Shell element conversion in `mesher.py`: Gmsh outputs CPS/CPE → must be rewritten as S3/S4 for CalculiX
- FRD output from CalculiX expands S3 → wedge elements with z=±thickness/2 nodes; the viewer handles this transparently via VTU

### Testing Requirements
- Run `python src/pipeline.py` from the repo root; all six steps should print and complete
- If only testing a single module, instantiate its class directly and point it at files in `workspace/`

### Common Patterns
- Each class constructor accepts a `workspace: Path` argument and creates the directory if absent
- Pipeline steps communicate via files, not in-memory objects — enables step-by-step debugging by inspecting intermediate files in `workspace/`
- Free-free modal analysis produces ~7 near-zero rigid body modes; first flexible mode is typically index 8

## Dependencies

### Internal
- `config.py` — imported by all other modules in this package

### External
- `gmsh` — `mesher.py`
- `pyvista` — `viewer.py`
- `numpy` — `viewer.py`, `dat_parser.py`
- `ccx2paraview` — `frd_converter.py`

<!-- MANUAL: -->
