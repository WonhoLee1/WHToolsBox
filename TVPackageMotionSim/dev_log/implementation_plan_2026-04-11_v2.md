# Implementation Plan - Restructuring Component Configuration & Result Management (Backup)

User wants to unify component-level meshing and constraint settings into a dictionary-driven structure (similar to `contacts` and `welds`) and confirm the existence/role of a specialized simulation results class.

## Proposed Changes

### 1. Configuration Restructuring

#### [MODIFY] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- Introduce a default `components` dictionary in `get_default_config()`.
- Group `div`, `use_weld`, and `mass` settings into this dictionary using part names as keys (`paper`, `cushion`, `opencell`, `chassis`, etc.).
- Update `sync_phys_config()` to prioritize values from the `components` dictionary while maintaining backward compatibility with flat keys (e.g., `box_div`).

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- Refactor `get_single_body_instance()` and `create_model()` to consume parameters from the new `components` dictionary.
- Standardize the mapping between dictionary keys and `BaseDiscreteBody` subclasses.

#### [MODIFY] [run_drop_simulation_cases_v5.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_cases_v5.py)
- Update test case setups to use the new `components` dictionary instead of individual meshing/mass keys.

### 2. Specialized Result Management

#### [NEW] [whts_result.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_result.py)
- Define `WHToolsResultManager` class.
- Implement methods for:
  - Unit scaling (m to mm).
  - Component-wise metric extraction (Peak G, PBA, RRG).
  - Filter application (CFC frequency filtering).
  - Visualization support (preparing data for Plotly/Matplotlib).

#### [MODIFY] [whts_engine.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_engine.py)
- Integrate the new `WHToolsResultManager` to handle simulation finalization and data export.

---

## Dev Logs & Documentation

### [NEW] [implementation_plan_2026-04-11_v2.md](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/implementation_plan_2026-04-11_v2.md)
- Backup of this plan.

## Verification Plan

### Automated Tests
- Run `scratch/test_v5_contacts.py` (updated for the new config) to verify XML generation integrity.
- Execute a sample simulation in `run_drop_simulation_cases_v5.py` to confirm end-to-end functionality.

### Manual Verification
- Inspect generated XML files to ensure `div` and `weld" settings are correctly applied from the new dictionary structure.
- Verify that result (.pkl) files still load correctly in the Post-Processing UI.
