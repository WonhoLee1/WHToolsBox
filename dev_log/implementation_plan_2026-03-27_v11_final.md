# Post-Processing UI Refinement & Mapping (v11)

This phase enhances the structural analysis tools with granular control, physically accurate mapping, and synchronized visual feedback using professional colormaps.

## Proposed Changes

### [Post-Processing UI]

#### [MODIFY] [run_drop_simulation_v3.py](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_drop_simulation_v3.py)

- **UI Layout Enhancement**:
    - Add a `ttk.Combobox` to `PostProcessingUI` to select which component (e.g., `bcushion`, `bopencell`) to analyze in the 2D map.
    - Add a new button **"Show Impact Analysis (Plots)"**:
        - This button will trigger a refined version of `plot_results()` that displays the G-Force and Kinematics plots in interactive Matplotlib windows.
- **2D Distortion Mapping**:
    - Force **Equal Aspect Ratio** (`ax.set_aspect('equal')`) in `plot_2d_distortion_map` for accurate geometric representation.
    - Filter data based on the selected component from the Combobox.
- **Synchronized MuJoCo Coloring**:
    - Update `apply_rank_distortion_heatmap` to use `matplotlib.cm.get_cmap('RdYlBu_r')` for MuJoCo geom coloring.
    - This ensures that the MuJoCo viewer and the 2D plots use the exact same sophisticated color scheme.

## Implementation Details

### Color Synchronization
```python
import matplotlib.cm as cm
cmap = cm.get_cmap('RdYlBu_r')
# For a normalized rank f:
rgba = cmap(f) # Returns (R, G, B, A) in 0~1
```

### Equal Aspect Ratio
```python
ax.imshow(...)
ax.set_aspect('equal', adjustable='box')
```

## Verification Plan

### Manual Verification
1. **Body Selection**: Select `bcushion` and then `bopencell` in the UI; verify the 2D map updates to reflect the selected body.
2. **Aspect Ratio Check**: Confirm the 2D map blocks look square (or proportional to grid) instead of being stretched to fit the window.
3. **Color Matching**: Verify the RED/BLUE/YELLOW transitions in MuJoCo match the 2D plot legend.
4. **Impact Plots**: Click the new button and verify the G-Force/Kinematics plots pop up as interactive windows.
