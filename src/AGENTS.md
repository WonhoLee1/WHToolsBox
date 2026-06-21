<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-04 | Updated: 2026-06-04 -->

# src

## Purpose
Main Python source for the AutoCalculix pipeline. Contains the orchestrator (`pipeline.py`), the external API (`autocalculix_api.py`), and two sub-packages: `core/` (the six pipeline steps) and `utils/` (RAG index builders and format converters).

## Key Files

| File | Description |
|------|-------------|
| `pipeline.py` | `AutoCalculixPipeline` class — three entry points: `run_with_meshing`, `run_from_inp`, `run_from_external`; also the `__main__` example runner |
| `autocalculix_api.py` | Programmatic API for external projects; accepts raw node/element dicts and returns frequencies + VTU paths |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `core/` | Six single-responsibility modules covering each pipeline step (see `core/AGENTS.md`) |
| `utils/` | RAG database builders and auxiliary format converters (see `utils/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Import with `from src.core.xxx import ...` — `BASE_DIR` is always the repo root; both `pipeline.py` and `autocalculix_api.py` append it to `sys.path` on startup
- `autocalculix_api.py` is the integration point for WHT_LightChassisModel; its function signature is the public contract — do not break it without coordinating with that project
- `pipeline.py` is also the `__main__` entry point; keep the three example cases at the bottom commented-out-by-default

### Testing Requirements
- Run `python src/pipeline.py` from the repo root to execute case 1 (Gmsh tray mesh)
- Confirm printed steps 1–5 complete and a PyVista window opens

### Common Patterns
- Both `pipeline.py` and `autocalculix_api.py` resolve `BASE_DIR = Path(__file__).resolve().parent.parent` to locate `workspace/` and config

## Dependencies

### Internal
- `src/core/` — all six pipeline modules
- `src/core/config.py` — `CALCULIX_EXE`, `WORKSPACE_DIR`, dataclass configs

### External
- `gmsh`, `pyvista`, `numpy`, `ccx2paraview`

<!-- MANUAL: -->
