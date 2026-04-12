# Implementation Plan - Goal: Transform configuration and results into a component-centric structure (Backup v5)

## Core Objectives
1.  **Config Consolidation**: Group meshing (`div`), constraint (`use_weld`), and `mass` settings into a unified `components` dictionary.
2.  **Result Class Evolution**: Evolve `DropSimResult` into an active dynamics analysis engine capable of re-extracting, filtering, and deriving kinematics.

## Proposed Changes

### 1. Configuration: Component-Centric Restructuring

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- Standardize `components` dictionary in `get_default_config()`.
- Group `div`, `use_weld`, and `mass` settings by part name.
- Update `sync_phys_config()` for backward compatibility.

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- Refactor builder to use `config["components"][part_name]` for model generation.

#### [MODIFY] [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py)
- Update case setups to the dictionary-based configuration.

### 2. Output: Evolution of DropSimResult (Dynamics Analysis Engine)

#### [MODIFY] [whts_data.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_data.py)
- Enrich `DropSimResult` with specialized dynamics methods:
  - `compute_kinematics_by_diff()`: Derive velocity and acceleration from position history using numerical differentiation (e.g., `np.gradient`).
  - `verify_accelerations()`: Compare recorded signal vs. derived signal for numerical stability checks.
  - `apply_cfc_filter(cfc_level=180)`: Integrated physical data filtering (CFC 60/180/1000).
  - `get_performance_summary()`: Generate PASS/FAIL/PEAK summaries.
  - `extract_sub_dataset(bodies: List[int])`: Extract 6DOF time-series for specific components.

---

## Dev Logs & Documentation

### [NEW] [implementation_plan_2026-04-11_v5.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_2026-04-11_v5.md)
- Backup of this finalized v5 plan.

---

## Verification Plan

### Automated Tests
- Run updated `scratch/test_v5_contacts.py` to verify the builder correctly interprets the new dictionary structure.
- Verify `DropSimResult.compute_kinematics_by_diff()` by comparing its output with recorded `vel_hist`.

### Manual Verification
- Verify generated XMLs reflect correct meshing/constraint settings.
- Confirm updated pkl objects are compatible with existing dashboard UIs.
