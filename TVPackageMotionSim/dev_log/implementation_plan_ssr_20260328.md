# SSR (Structural Surface Reconstruction) Implementation Plan (2026-03-28)

## Goal
Implement a high-fidelity 2D contour visualization engine using Thin Plate Spline interpolation (SSR) to reconstruct smooth deformation surfaces from discrete block data.

## User Review Required
> [!IMPORTANT]
> This feature requires `scipy` to be installed in the environment. If `scipy` is missing, the UI will fall back to standard linear interpolation (matplotlib default contourf).

## Proposed Changes

### [Component] Post-Processing UI (postprocess_ui.py)

#### [MODIFY] [postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/postprocess_ui.py)
- **State Management**:
    - Add `self._ssr_mode_var = tk.BooleanVar(value=False)` in `__init__`.
- **UI Enhancement**:
    - In `_build_contour_tab`, add a checkbox: `[ ] 고정밀 모드 보간 (SSR)`.
    - Update help text to explain SSR.
- **SSR Engine**:
    - Modify `_draw_single_contour` to implement the SSR logic:
        - Detect if SSR is enabled.
        - Use `scipy.interpolate.Rbf` with `function='thin_plate'` (Thin Plate Spline).
        - Generate a high-resolution mesh (e.g., 50x50 or 10x multiplier) for smooth rendering.
        - Handle edge cases (missing `scipy` or too few data points).
- **Bug Fix**:
    - Ensure all remaining `NameError: i` risks are mitigated in any plotting loops.

## Verification Plan

### Automated/Manual Verification
1.  Launch UI and navigate to the **2D Field Contour** tab.
2.  Select a component (e.g., Panel) and a metric (e.g., Bending).
3.  Click **매트릭스 컨투어 생성** with SSR off -> Verify blocky/standard contour.
4.  Check **고정밀 모드 보간 (SSR)** and click again -> Verify smooth, high-fidelity surface reconstruction.
5.  Test during animation (Live Sync).

---
### [Component] Simulator Backend (run_drop_simulation_v3.py)
- No changes required, as SSR is a post-processing visualization layer.
