# Implementation Plan - Goal: Transform configuration and results into a component-centric structure (Final Backup)

## Core Objectives
1.  **Config Consolidation**: Group meshing (`div`), constraint (`use_weld`), and `mass` settings into a unified `components` dictionary.
2.  **Result Class Evolution**: Evolve `DropSimResult` from a passive dataclass into an active analysis engine capable of filtering, scaling, and summarizing results.

## Proposed Changes

### 1. Configuration: Component-Centric Restructuring

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- Update `get_default_config()` to include a `components` dictionary:
  ```python
  "components": {
      "paper": {"div": [5, 5, 1], "use_weld": True, "mass": 4.0},
      "cushion": {"div": [3, 3, 3], "use_weld": True, "mass": 2.0},
      ...
  }
  ```
- Update `sync_phys_config()` to ensure seamless mapping between the new dictionary and any legacy code still expecting flat keys.

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- Refactor model construction logic to pull parameters directly from `config["components"][part_name]`.

#### [MODIFY] [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py)
- Update case setups to the streamlined dictionary-based configuration.

### 2. Output: Evolution of DropSimResult

#### [MODIFY] [whts_data.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_data.py)
- Enrich `DropSimResult` with functional methods:
  - `apply_unit_scaling(to="mm")`: Internalize unit management.
  - `apply_cfc_filter(cfc_level=180)`: Integrate physical data filtering.
  - `get_performance_summary()`: Logic for structural and impact analysis summary.
  - `check_safety_margins()`: Threshold-based PASS/FAIL logic.

---

## Dev Logs & Documentation

### [NEW] [implementation_plan_2026-04-11_final.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_2026-04-11_final.md)
- Backup of this finalized plan.

---

## Verification Plan

### Automated Tests
- Run updated `scratch/test_v5_contacts.py` to verify the builder correctly interprets the new `components` dictionary.
- Verify that `DropSimResult.apply_unit_scaling()` correctly updates history arrays.

### Manual Verification
- Compare older and newer generated XMLs to ensure no regressions in meshing/constraints.
- Check that the Post-Processing UI displays nested result summaries correctly.
