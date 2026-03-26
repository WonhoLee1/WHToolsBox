# Walkthrough: Post-Processing UI & Mapping Refinement (v11)

This update matures the professional analysis suite with granular component selection, synchronized coloring, and interactive physical plots.

## Key Enhancements

### 1. Granular Body Selection
The **WHTOOLS Post-Processing Explorer** now includes a component selection tool:
- **Combobox Integration**: Choose a specific body (e.g., `bcushion`, `bopencell`) to analyze.
- **Contextual Visualization**: The 2D Distortion Map now reflects the specific geometry and metrics of the selected component.

### 2. Synchronized "True" Heatmaps
Visual consistency between MuJoCo and Matplotlib is now guaranteed:
- **Matplotlib Colormap Sync**: MuJoCo rank-based coloring now uses the exact `RdYlBu_r` colormap from Matplotlib.
- **Unified Color Language**: RED always means maximum relative distortion, while BLUE indicates the safest (least distorted) zones across all viewers.

### 3. Physically Accurate 2D Maps (Equal Aspect)
- **Geometry Preservation**: All 2D heatmaps now enforce an **Equal Aspect Ratio** (`ax.set_aspect('equal')`).
- **Real-world Proportions**: Blocks are rendered based on their grid proportions, preventing visual stretching and aiding in accurate failure pattern identification.

### 4. Interactive Impact Analysis (Plots)
A new button **"Show Impact Analysis (Plots)"** provides immediate access to time-series data:
- **G-Force Trace**: View the assembly's deceleration profile in a pop-up window.
- **Kinematics (Z-Axis)**: Side-by-side analysis of Position, Velocity, and Acceleration.
- **Non-Blocking Windows**: Multiple plot windows can be opened simultaneously for comparative analysis.

## How to Verify
1.  **Run Simulation**: Wait for the simulation to complete.
2.  **Select Target**: Use the Combobox in the Post-Processing UI to select a body.
3.  **Check 2D Map**: Click "Show 2D Distortion Map" and verify:
    - The plot title matches the selected body.
    - The grid looks square/proportional (Equal Aspect).
4.  **Check MuJoCo Colors**: Click "Apply Distortion Heatmap" and verify the colors in MuJoCo match the 2D plot's `RdYlBu_r` legend.
5.  **Check Impact Plots**: Click "Show Impact Analysis" and confirm the G-force and Kinematics windows pop up.

---
> [!NOTE]
> The "Apply Heatmap" function now applies the `RdYlBu_r` spectrum to the MuJoCo viewer, ensuring a professional engineering aesthetic.
