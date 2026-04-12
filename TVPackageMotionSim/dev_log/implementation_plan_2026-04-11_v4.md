# Implementation Plan - Goal: Transform configuration and results into a component-centric structure (Backup v4)

## Core Objectives
1.  **Config Consolidation**: Group meshing (`div`), constraint (`use_weld`), and `mass` settings into a unified `components` dictionary.
2.  **Result Class Evolution**: Evolve `DropSimResult` from a passive dataclass into an active engine capable of loading, re-extracting, and analyzing simulation data.

## Proposed Changes

### 1. Configuration: Component-Centric Restructuring

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- Standardize `components` dictionary in `get_default_config()`.
- Group `div`, `use_weld`, and `mass` settings by part name.
- Update `sync_phys_config()` for backward compatibility with individual keys.

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- Refactor builder to use `config["components"][part_name]` for model generation.

#### [MODIFY] [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py)
- Update case setups to the dictionary-based configuration.

### 2. Output: Evolution of DropSimResult (Enhanced Analysis & Extraction)

#### [MODIFY] [whts_data.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_data.py)
- Enhance `DropSimResult` with functional methods:
  - `load_from_pkl(path)`: Robust loading with version compatibility checks.
  - `extract_time_series(part_name, metric)`: Specialized data extraction from history arrays.
  - `apply_cfc_filter(cfc_level=180)`: Integrated physical data filtering for re-analysis.
  - `get_summary_report()`: Generate engineering summaries (PASS/FAIL/PEAK).
  - `recompute_structural_metrics()`: Ability to re-run analysis from raw marker data stored in the object.

---

## Dev Logs & Documentation

### [NEW] [implementation_plan_2026-04-11_v4.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_2026-04-11_v4.md)
- Backup of this finalized v4 plan.

---

## Verification Plan

### Automated Tests
- Run updated `scratch/test_v5_contacts.py` to verify the builder correctly interprets the new dictionary structure.
- Verify `DropSimResult.load()` and `extract_time_series()` using an existing `simulation_result.pkl`.

### Manual Verification
- Verify generated XMLs reflect correct meshing/constraint settings.
- Confirm updated pkl objects are compatible with existing dashboard UIs.
