# Walkthrough: Integrated Simulation Control UI (V2)

The legacy post-processing tool has been fully refactored into a modern **Control Center** for the WHTOOLS MuJoCo simulation environment.

## Key Features Implemented

### 1. Advanced Parameter Editor (Settings Tab)
- **Dynamic Configuration**: Users can modify over 40+ simulation parameters (Scenario, Physics, Materials) directly from the UI.
- **Auto-Sync**: Changes are validated and synchronized with the `DropSimulator` configuration before execution.

### 2. Real-time Simulation Console (Execution Tab)
- **Live Logging**: Redirection of `stdout/stderr` allows users to monitor MuJoCo's solver progress and logs within the application.
- **Asynchronous Execution**: The simulation runs in a dedicated thread, keeping the UI responsive and enabling "Force Stop" capabilities.
- **MuJoCo Interface Control**: Easily toggle the native MuJoCo Viewer on/off.

### 3. Integrated Post-Analysis Pipeline
- **History Management**: Browse past results with automated summary metrics (Max Stress, RRG, PBA).
- **Dashboard Handshake**: The "Analyze In 3D Dashboard" button launches the `QtVisualizerV2` with the selected result file pre-loaded.

---

## Technical Details

### [whts_postprocess_ui_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_ui_v2.py)
- **Framework**: PySide6 (Qt)
- **UI Logic**: Threaded `SimulationThread` for MuJoCo execution.
- **Inter-process Communication**: `subprocess` for launching the 3D Dashboard.

### [whts_postprocess_engine_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_engine_v2.py)
- **Cleanup**: Removed `jax`, `ssr`, and `psr` legacy dependencies to prevent overhead.
- **Data API**: Robust `.pkl` summary extraction.

---

## Verification Results

- [x] **Config Editor**: Verified that modifying 'Drop Height' successfully propagates to the `DropSimulator`.
- [x] **Log Redirection**: Verified that `print()` statements from `DropSimulator` appear in the green-on-black console.
- [x] **Interoperability**: Verified that the 3D Dashboard opens automatically with the result of a fresh simulation run.

> [!TIP]
> Use the **"3D Analysis"** navigation button for a quick launch of the visualization environment without loading a specific file.
