# Refactoring: Integrated Simulation Control UI (V2)

The objective is to transform the legacy `whts_postprocess` module from a standalone analysis tool into an **Integrated Control Center** for MuJoCo simulations. This involves moving to **PySide6** for UI consistency and refocusing the features on simulation management (config editing and execution).

## User Review Required

> [!IMPORTANT]
> **Functional Pivot**: The new UI will primarily handle simulation execution and configuration. Detailed physical analysis (SSR/Contour) will be offloaded to the `QtVisualizerV2` (from `plate_by_markers_v2.py`), which the Control Center will launch.

> [!WARNING]
> **Engine Cleanup**: All JAX/SSR code will be removed from the legacy engine to avoid redundancy and dependency bloat, as the new `ShellDeformationAnalyzer` handles this more accurately.

## Proposed Changes

### 1. Structural Analysis Engine (v2)
`whts_postprocess_engine_v2.py` will serve as the 'headless' logic layer.

#### [NEW] [whts_postprocess_engine_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_engine_v2.py)
- **ConfigManager**: Methods to load/save/modify the `cfg` dictionary used by `DropSimulator`.
- **SimulationRunner**: A wrapper for `sim.simulate()` that supports non-blocking execution (for UI integration).
- **SummaryExtractor**: Updated `extract_global_summary_data` that pulls RRG/Stress/PBA from `.pkl` files without requiring SSR re-calculation.

### 2. Integrated Control UI (v2)
`whts_postprocess_ui_v2.py` will provide the PySide6-based interface.

#### [NEW] [whts_postprocess_ui_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_ui_v2.py)
- **MainWindow**: Standardized with `D2Coding` font and the dark/premium aesthetic of WHTOOLS.
- **Tab 1: Config Editor**:
    - Categorized parameter groups (Geometry, Drop Environment, Physics, Mass).
    - Validation for input values.
- **Tab 2: Simulation Console**:
    - Real-time logging of MuJoCo progress.
    - "Run Simulation" and "Force Stop" buttons.
- **Tab 3: History & Analysis**:
    - List of previous simulation runs (`.pkl` files).
    - Summary table of key metrics (Max Stress, RRG).
    - **[Launch Analysis]** button to open the 3D Dashboard for a selected run.

---

## Open Questions

- **Execution Mode**: Should the "Run Simulation" button launch a completely separate process to prevent UI hang, or use a Python Thread within the same process? (Thread is easier for log redirection, Process is safer for stability).
- **Config persistence**: Should we save modified configs as separate `.json` files or overwrite the default dictionary in memory?

## Verification Plan

### Automated Tests
- Verify `ConfigManager` can round-trip a simulation setup to JSON.
- Test `SimulationRunner` with a short 0.1s dummy simulation to ensure thread safety.

### Manual Verification
- Open the new UI, modify 'Drop Height', run simulation, and verify that the 3D Dashboard can be launched with the new result.
