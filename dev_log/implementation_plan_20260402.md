# Qt-Based Plate Deformation Analyzer (v2)

This plan integrates **PySide6** to create a professional engineering UI that synchronizes **PyVista** and **Matplotlib** visualizers. It also refines the mechanics algorithm to focus on *relative deformation* from an initial planar fit.

## User Review Required

> [!IMPORTANT]
> - **Qt Architecture**: I will use a `QMainWindow` with a horizontal split. Left: PyVista (3D View + Marker Spheres). Right: Matplotlib (Strain Field + Energy/Stress Plots).
> - **Unified Control**: A fixed slider at the bottom will control the global frame index.
> - **Algorithm Update**: 
>   - Frame 0 is used to define the **Reference Plane** via PCA.
>   - Displacement $\Delta Z$ is calculated as the local height change from Frame 0 markers to current markers.
>   - This $\Delta Z$ is interpolated using `RBFInterpolator` and mapped onto the flat grid.
> - **Visualization**: Marker positions will be rendered as **Sphere Glyphs** in the 3D view to clarify context.

## Proposed Changes

### [Component] `plate_by_markers.py`

#### [MODIFY] [plate_by_markers.py](file:///c:/Users/GOODMAN/WHToolsBox/plate_by_markers.py)

- **Import Update**:
    - `PySide6.QtWidgets, QtCore, QtGui`.
    - `pyvistaqt.QtInteractor`.
    - `matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`.
- **`PlateMechanicsSolver` Refinement**:
    - Add `initialize_reference(markers_0)`: Fits a plane and saves the reference mesh $S_0$.
    - `solve_relative(markers_current, markers_0)`:
        - Project current markers to reference coordinate system.
        - Calculate relative displacement $\Delta Z$.
        - Interpolate $\Delta Z$ over the grid.
        - Derive **Equivalent Strain** $\epsilon_{eq}$ from curvatures.
- **`PlateVisualizerApp` (New Class)**:
    - Inherit from `QMainWindow`.
    - `setup_ui()`:
        - `QSplitter` for PyVista and Matplotlib.
        - `QSlider` for frames.
        - `QLabel` for frame info (8pt, Noto Sans KR / Segoe UI).
    - `update_views(frame_idx)`:
        - Update PyVista mesh and markers (polydata).
        - Update Matplotlib canvas (re-render plots).
- **Algorithm Check**:
    - Ensure Kabsch alignment is performed *correctly* to maintain the frame-to-frame correspondence for relative displacement.

## Open Questions

> [!QUESTION]
> 1. **Visualizer Layout**: Do you prefer side-by-side or a Tabbed interface? I will start with side-by-side (using `QSplitter`) for simultaneous viewing.
> 2. **Initial Shape**: If Frame 0 markers are not perfectly flat, should I fit the "Best Plane" (average plane) or follow the initial marker positions literally but treat them as "Zero Stress"? I will assume **Best Plane** fit as requested.

## Verification Plan

### Automated Tests
- Script execution under `PySide6`.
- Slider scrubbing performance check.
- Verification of strain values against expected magnitudes.

### Manual Verification
- Visual inspection of marker spheres and their alignment with the plate.
- Interactive responsiveness check between PyVista and Matplotlib.
