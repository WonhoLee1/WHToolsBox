# Implementation Plan - Goal: Transform configuration and results into a JAX-accelerated component-centric structure (Backup v6)

## Core Objectives
1.  **Config Consolidation**: Group meshing (`div`), constraint (`use_weld`), and `mass` settings into a unified `components` dictionary.
2.  **Result Class Evolution**: Evolve `DropSimResult` into an active JAX-accelerated dynamics analysis engine.

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

### 2. Output: Evolution of DropSimResult (JAX-Accelerated Dynamics Engine)

#### [MODIFY] [whts_data.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_data.py)
- Integrate **JAX** for high-speed analysis:
  - `compute_kinematics_jax()`: Use JAX to derive velocity/acceleration with auto-vectorization.
  - `apply_cfc_filter_jax()`: JIT-compiled CFC filtering for instant processing.
- Add engineering judgment methods:
  - `get_performance_summary()`: Generate PASS/FAIL/PEAK summaries.
  - `export_dataset(format="csv/xlsx")`: High-speed data export.
- Maintain fallback: Use standard NumPy if JAX is not available/configured in the environment.

---

## Dev Logs & Documentation

### [NEW] [implementation_plan_2026-04-11_v6.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_2026-04-11_v6.md)
- Backup of this finalized v6 plan.

---

## Verification Plan

### Automated Tests
- Run updated `scratch/test_v5_contacts.py` to verify the builder correctly interprets the new dictionary structure.
- Benchmarking `DropSimResult.compute_kinematics_jax()` against NumPy fallback.

### Manual Verification
- Verify generated XMLs reflect correct meshing/constraint settings.
- Confirm updated pkl objects are compatible with existing dashboard UIs.
