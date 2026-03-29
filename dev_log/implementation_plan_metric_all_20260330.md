# Implementation Plan - Detailed Metric Analysis & Reporting Optimization (Backup 2026-03-30)

This plan addresses two user requests:
1.  **Plot Mode Toggle**: Adding an option in the Structural Analysis tab to plot either the component-level maximum or each individual block's time-series data.
2.  **Enhanced Final Report**: Updating the post-simulation console report to include block indices for maximum stress/deformation values.

## User Review Required

> [!IMPORTANT]
> - **Performance Warning**: Plotting "All Blocks" for large components (e.g., a chassis with 100+ blocks) will cause significant UI lag. This mode is best used for high-fidelity analysis of a single small component.
> - **Final Report Change**: The console output at the end of the simulation will now be more verbose, showing `Metric Value @ Block #Index`.

## Proposed Changes

### Post-Processing UI

#### [MODIFY] [whts_postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_ui.py)

1.  **UI Layout**: Add a "Plot Detail" frame in `_build_structural_tab` with Radiobuttons: `[ ] Max. of Blocks (Aggregated)` and `[ ] All Blocks (Detailed)`.
2.  **Plotting Logic**: 
    - In `_on_plot_structural`, if "All Blocks" is selected:
        - Iterate through each block in the component's metric data.
        - Plot each block's time-series separately.
        - Label the curve as `ComponentName - Block #Index`.

### Reporting Engine

#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)

1.  **finalize_simulation_results**:
    - Update the table header to include "Block Index" columns for each primary metric (Bending, Twist, Stress, RRG).
    - Implement a finding logic that tracks both the `max_value` and the `grid_idx` for each metric across all blocks in a component.
    - Standardize metric names to match the UI (e.g., BS, TS, RRG, Bend).

## Verification Plan

### Manual Verification
1.  **UI Check**:
    - Select "All Blocks" mode and plot the "Panel" component.
    - Verify that multiple lines appear and the legend identifies block indices.
2.  **Report Check**:
    - Run a short simulation.
    - Verify that the console output table at the end shows columns like `Max Bend (deg) | @Blk`.
    - Confirm the values and indices are physically plausible.
