# Implementation Plan - Centralizing Simulation Output Paths (Backup)

The goal is to redirect all generated `rds-` (Raw Data Set) and `export_` directories into a centralized `results/` folder for better project organization.

## Proposed Changes

### [run_drop_simulator](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator)

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- Update `DropSimulator.__init__` to prefix `self.output_dir` with `results/`.
- Ensure the parent `results/` directory is created automatically.

#### [MODIFY] [whts_postprocess_engine_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_engine_v2.py)
- Update `get_result_files` to perform a recursive search (or look into one-level-deep subdirectories) for `.pkl` files within `results/`. This ensures simulation results nested in `rds-` folders are still discoverable.

---

### [Simulation Scripts](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/)

#### [MODIFY] [run_drop_simulation_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_v2.py)
- Update `run_simulation` function to prefix `output_dir` with `results/`.

#### [MODIFY] [run_drop_simulation_cases_v6.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v6.py)
- Update `run_analysis_and_dashboard_minimal` to prefix `export_path` with `results/`.

---

### [Dev Logs & Documentation](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log)

#### [NEW] [implementation_plan_2026-04-11.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_2026-04-11.md)
- Backup of this implementation plan as per USER rule.

## Verification Plan

### Automated Tests
- Run `run_drop_simulation_cases_v5.py` (which uses the engine) and verify that a `results/rds-.../` folder is created.
- Run `run_drop_simulation_cases_v6.py` and verify `results/export_.../` is created.

### Manual Verification
- Check the `results/` directory structure to ensure it matches the requested hierarchy.
- Open the integrated control UI (`whts_postprocess_ui_v2.py`) and verify that it can still list and analyze results saved in the new structure.
