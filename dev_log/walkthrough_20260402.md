# Walkthrough: Plate Deformation Analyzer (Qt + PyVista + Matplotlib)

We have successfully rebuilt the **Plate Deformation Analyzer** into a professional engineering application. This version integrates high-performance 3D visualization with detailed 2D/3D post-processing plots within a unified **Qt (PySide6)** window.

## Key Features

### 1. Dual-Visualizer Architecture
- **PyVista (Left)**: High-resolution 3D view of the plate.
  - **Dynamic Mesh**: Updates real-time as you scrub the slider.
  - **Marker Spheres**: Marker positions are rendered as black spheres to show the measurement points.
  - **Field Map**: Displays **Equivalent Strain** as the primary field.
- **Matplotlib (Right)**: Detailed analytical plots.
  - **Strain Contour**: 2D projection of the strain field for precision reading.
  - **Energy/Stress Curves**: Temporal plots of strain energy and max stress, synchronized with the current frame indicator.

### 2. Relative Deformation Algorithm
The solver now accurately follows the engineering requirement for relative measurement:
- **Frame 0 Alignment**: Initial marker positions are used to define the "Flat Reference Plane" using PCA.
- **Incremental Fitting**: For each frame, we align the markers to Frame 0 and calculate the local height change ($\Delta Z$).
- **Displacement Mapping**: This relative displacement is interpolated over the mesh grid, ensuring the plate's motion is strictly derived from marker variations.

### 3. Interactive Sync & Control
- **Scrub Slider**: Smoothly navigate through the entire time history.
- **Keyboard Control**: Use the `Left` and `Right` arrow keys for frame-by-frame stepping.
- **Dynamic Legend**: The color scale (legend) updates its range to reflect the current frame's data.

## Verification Results

- **Environment**: Verified `PySide6` and `pyvistaqt` installation in the `vdmc` environment.
- **Execution**: The analysis engine successfully processes multi-frame data and initializes the Qt event loop.
- **Aesthetics**: UI uses **8pt Noto Sans KR / Segoe UI** fonts for a sleek, technical appearance.

> [!TIP]
> To run the analyzer, simply execute the script within your `vdmc` environment:
> ```bash
> python plate_by_markers.py
> ```

---
*Developed by **Antigravity** for **WHTOOLS**.*
