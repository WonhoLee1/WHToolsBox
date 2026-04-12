# Implementation Plan - Restructuring Component Configuration & Enhancing Result Management (Backup v3)

Unify component-level meshing, constraints, and mass settings into a dictionary-driven structure and enhance the existing `DropSimResult` class for more professional data processing.

## Proposed Changes

### 1. Configuration Restructuring (Core Task)

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- Introduce a standardized `components` dictionary in `get_default_config()`.
- Group `div`, `use_weld`, and `mass` settings into this dictionary using part names as keys (`paper`, `cushion`, `opencell`, `chassis`, etc.).
- Update `sync_phys_config()` to handle synchronization between the new `components` dictionary and legacy flat keys for backward compatibility.

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- Update code to consume parameters from `config["components"][part_name]` instead of individual keys.
- Streamline part instantiation logic by iterating over or directly accessing the `components` dict.

#### [MODIFY] [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py)
- Refactor test case setups to use the new `components` unified dictionary style.

### 2. Result Management Enhancement

#### [MODIFY] [whts_data.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_data.py)
- Enhance `DropSimResult` with professional analysis methods:
  - `apply_cfc_filter()`: Apply standard hardware test filters (CFC 60/180).
  - `get_status_summary()`: Logic to determine PASS/FAIL/WARNING based on customizable thresholds.
  - `export_csv_summary()`: Export key metrics to professional CSV reports.

---

## Dev Logs & Documentation

### [NEW] [implementation_plan_2026-04-11_v3.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_2026-04-11_v3.md)
- Backup of this finalized plan.

## Verification Plan

### Automated Tests
- Run `scratch/test_v5_contacts.py` to ensure XML builder correctly reads the new dictionary structure.
- Execute a representative case in `run_drop_simulation_cases_v5.py` to verify end-to-end flow.

### Manual Verification
- Verify that `results/rds-.../summary_report.txt` correctly reflects mass and meshing details from the new dictionary.
- Confirm compatibility with the existing Post-Processing UI.
