# Implementation Plan - Detailed Metric Analysis & Multi-Subplot Optimization (Backup 2026-03-30)

This plan covers all three recent requests to improve WHTOOLS simulation analysis.

## User Review Required

> [!IMPORTANT]
> - **Plot Detail**: "All Blocks" is intended for high-fidelity study of one component at a time. Plotting hundreds of blocks across several components may cause significant lag in the UI.
> - **Tooltip Fix**: The fix in `mpl_extension.py` will correctly handle multiple subplots in the Kinematic and Structural analysis views by isolating hover labels per-axes.

## Proposed Changes

### 1. Structural Analysis Plotting (Tab 2)
#### [MODIFY] [whts_postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_ui.py)
- **UI**: Add "Plot Mode" (Max vs All) radio buttons to Section 1 (Metric Selection).
- **Logic**: Update `_on_plot_structural` to iterate and plot individual blocks if "All" is selected. Update legend formatting to `[Comp] - B#[Idx]`.

### 2. Final Report (Console Output)
#### [MODIFY] [whts_reporting.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_reporting.py)
- **Reporting**: Update `finalize_simulation_results` to find and print `@Blk [Index]` for Max Bending, Max Twist, Max Stress (BS/TS), and Max RRG.
- **Alignment**: Ensure the text table is properly aligned for better readability in the PowerShell window.

### 3. Interactive Analysis (Subplot Tooltips)
#### [MODIFY] [mpl_extension.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/mpl_extension.py)
- **Bug Fix**: Change `self._hover_labels[fig]` to `self._hover_labels[(fig, ax)]` to support multiple subplots within a single figure.
- **Logic**: Hide hover labels of other axes in the same figure when one axes becomes active.

## Verification Plan

### Manual Verification
1. **Plotting**: Select "All Blocks" mode and plot the "Panel" component. Verify correct legend labeling.
2. **Report**: Run a case and check the console output table at the end for block indices.
3. **Crosshair**: Open the Kinematic Data window (multiple subplots). Verify that hover tooltips appear in whichever subplot the mouse is actually over.
