# Implementation Plan - UI Layout Optimization (Backup 2026-03-30)

This plan optimizes the Post-Processing Explorer's UI by relocating the "Plot Detail" option for better logical flow and adding a scrollable container to the control pane to handle increasing UI density.

## User Review Required

> [!IMPORTANT]
> - **UI Relocation**: The "Plot Detail" (Max vs All) toggle will be moved from "1. 분석 지표 선택" to "2. 대상 부품 선택". It will be placed just above the primary "Generate Graph" button.
> - **Global Scrollbar**: A master vertical scrollbar will be added to the right-hand control panel in both the Structural and Field Contour tabs. This ensures that as we add more features, the UI remains accessible regardless of window size.

## Proposed Changes

### [MODIFY] [whts_postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_postprocess_ui.py)

#### 1. Relocate Plot Detail Options
- Remove `detail_f` from `_build_structural_tab` (Section 1).
- Add `detail_f` to Section 2, positioned between the scrollable component list (Body) and the utility buttons (Footer).
- Ensure consistency by applying a similar layout to the Field Contour tab if relevant (though RRG/PBA plotting is specific to Structural).

#### 2. Implement Scrollable Control Pane
- In `_build_structural_tab`, `_build_contour_tab`, and potentially others:
    - Wrap the current `ctrl` frame (which contains sections 1, 2, 3) inside a `tk.Canvas` with a vertical `ttk.Scrollbar`.
    - This allows the user to scroll through the "1. 지표 선택", "2. 부품 선택", "3. 상세 도움말" sections if they don't fit in the window.

#### 3. UX Polish
- Adjust padding and frames to ensure the scrollbar looks premium and integrates well with the existing theme.

## Verification Plan

### Manual Verification
1. Open the UI to the **Structural Analysis** tab.
2. Verify that **Plot Mode (Max vs All)** now appears in the **Section 2 (Target Component Selection)** grouping.
3. Resize the window to be smaller vertically. Verify that a scrollbar appears on the right side allowing access to all control groups.
4. Confirm that scrolling the right pane does not affect the central animation viewer.
