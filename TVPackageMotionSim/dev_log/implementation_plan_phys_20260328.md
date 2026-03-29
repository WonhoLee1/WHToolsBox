# Physical Dimension Mapping & True Aspect Ratio Implementation Plan (2026-03-28)

## Goal
Upgrade the 2D structural contour system to reflect actual product dimensions, maintain a 1:1 aspect ratio, and properly align legends for professional engineering reporting.

## User Review Required
> [!IMPORTANT]
> The physical mapping assumes that the `body_pos` of each block (`b_...`) reflects its design-time offset from the component root. If the model uses a different nesting structure (e.g., nested frames), a global coordinate transform may be needed, but local offsets are sufficient for contour mapping on a single component.

## Proposed Changes

### [Component] Post-Processing UI (postprocess_ui.py)

#### [MODIFY] [postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/postprocess_ui.py)
- **Data Extraction**:
    - Modify `_get_contour_grid_at` to also collect and return the physical X/Y coordinates of each block.
    - It will return `(X_grid, Y_grid, Value_grid)` where `X_grid` and `Y_grid` are 2D arrays of physical positions (m).
- **Visualization Engine**:
    - Update `_draw_single_contour` to:
        - Use physical `X` and `Y` meshgrids for `contourf`.
        - Set `ax.set_aspect('equal')` to ensure 1:1 physical aspect ratio.
        - Add proper axis labels (`m` or `mm`).
        - Implement right-side colorbar placement using `mpl_toolkits.axes_grid1.make_axes_locatable` to prevent layout distortion.
- **SSR Integration**:
    - Ensure the SSR (Thin Plate Spline) interpolation is performed over the physical coordinate space for maximum accuracy.

## Verification Plan

### Manual Verification
1.  Launch UI and select a wide component (e.g., Back Cover).
2.  Verify that the X/Y axes show physical dimensions (e.g., -0.7 to 0.7m) instead of grid indices (0 to 14).
3.  Verify that the aspect ratio correctly represents the product's actual shape (e.g., wide TV screen).
4.  Verify that the colorbar is neatly aligned on the right side of each subplot without overlapping.
