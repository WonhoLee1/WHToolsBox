# Field Contour UI & Visualization Enhancement Plan (2026-03-28)

## Goal
Optimize the Field Contour tab to match the Structural Analysis layout, implement a robust help system with mathematical formulas, and ensure the visualization stays synchronized with the simulation timeline in a non-blocking manner.

## User Review Required
> [!IMPORTANT]
> - **Component Filtering**: I will exclude any components containing `inertiaux_single` or `aux` from the selection list to focus on primary structural bodies.
> - **Non-Modal Window**: The Matrix Contour window will be changed to a persistent, non-modal window. Moving the slider in the main UI will trigger a refresh in the open contour window.
> - **Help Documentation**: I will generate/update the mathematical expression images for Bending, Twisting, RRG, and PBA.

## Proposed Changes

### [Component] Post-Processing UI (postprocess_ui.py)

#### 1. UI Layout Reorganization
- **[MODIFY] `_build_contour_tab`**:
    - Swap the order: **1. 분석 지표 선택** (Metrics) -> **2. 대상 부품 선택** (Components).
    - Filter components: Remove names containing `inertiaux_single` or `aux`.
    - Add a `?` (Help) button next to the `[ Control ]` label.

#### 2. Enhanced Help System
- **[MODIFY] `_show_metric_detailed_help`**:
    - Implement a new popup window that displays:
        - Technical definition of the metric.
        - **Mathematical formula** (rendered as an image or formatted text).
        - **Conceptual diagram** (using generated assets).
        - SSR logic explanation.

#### 3. High-Fidelity Plotting Engine
- **[MODIFY] `_draw_single_contour`**:
    - **Min/Max Marking**: Automatically detect absolute min/max points. Add arrows (`ax.annotate`) pointing to these locations with 8pt labels.
    - **Font Standardization**: Set X/Y axis labels and tick labels to `size=8`.
    - **Robust Scaling**: Implement `vmax` based on data distribution (e.g., 98th percentile or absolute max) to ensure the colorbar range is meaningful.

#### 4. Dynamic Live Sync
- **[MODIFY] `_on_show_contour_frame`**:
    - Store the Toplevel window reference in `self._contour_popup`.
    - Ensure it is non-modal (`grab_set()` removed or handled carefully).
- **[MODIFY] `_on_time_slider_change`**:
    - If `self._contour_popup` exists and is visible, trigger `_update_popup_contours(step)`.

## Verification Plan

### Manual Verification
1.  **UI Filter Test**: Verify that `aux` parts are missing from the list.
2.  **Help System Test**: Click `?` and verify the formulas and images appear.
3.  **Visualization Test**: Open Matrix Contour, move the time slider, and verify the contour updates instantly.
4.  **Min/Max Test**: Check if arrows correctly point to the highest and lowest values in the 2D field.
5.  **Font Test**: Verify the 8pt font size on axes.
