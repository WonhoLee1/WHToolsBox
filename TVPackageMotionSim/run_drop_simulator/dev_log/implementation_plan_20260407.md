# [WHTOOLS] Side-Face Fitting Stability Improvement Plan

## Goal
Resolve the "flying points" issue occurring on Top, Bottom, Left, and Right faces. This behavior is likely caused by numerical instability during polynomial fitting on narrow surfaces and unstable extrapolation at mesh boundaries.

## Root Cause Analysis
1.  **Over-Extrapolation**: Side faces (Top/Bottom/Left/Right) are narrow. High-degree polynomials (4th-5th degree) extrapolate wildly beyond the data range when a fixed 5mm margin is added. Even with near-zero deformation, the "tail" of the polynomial at the margin can fly away.
2.  **Numerical Instability (Normalization)**: Z-score normalization (`std`) for narrow faces results in a near-singular system matrix if `std` is extremely small. Tiny noise is amplified.
3.  **Over-fitting**: Using a high-degree polynomial (e.g., degree 4) in a dimension where there are only 2-3 markers (the thickness of a side plate) is mathematically unstable.

## Proposed Changes

### 1. [plate_by_markers_v2.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py)

#### [MODIFY] `AdvancedPlateOptimizer`
- **Robust Normalization**: Switch from Z-score (mean/std) to Min-Max scaling to map coordinates strictly to `[-1, 1]`.
- **Marker Density-Aware Degree**: Implement logic to automatically select `degree_x` and `degree_y` based on the number of unique markers in each dimension.
    - `degree = min(config_degree, num_unique_points - 1)`
    - This prevents overfitting and "flying" artifacts on narrow faces with sparse markers (e.g., side plates).
- **Ridge Regularization**: Increase the default Tikhonov regularization factor for better conditioning.

#### [MODIFY] `ShellDeformationAnalyzer.fit_reference_plane`
- **Relative Margin**: Change the 5mm fixed margin to a relative margin (e.g., 1% of the dimension) to prevent wild extrapolation on narrow side faces.
- **AspectRatio Awareness**: Automatically detect side faces based on W/H ratio and set conservative fitting parameters.

#### [MODIFY] `PlateMechanicsSolver.evaluate_batch`
- Update evaluation to use the new Min-Max normalization stats.

### 2. [whts_mapping.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping.py)

#### [MODIFY] `extract_face_markers`
- **Projection Check**: Ensure that for all faces, the local coordinate system alignment is robust, especially the `statistical` mode's SVD results.

## Verification Plan

### Automated Verification
- Run the simulation pipeline and check the `R-RMSE` and `F-RMSE` values reported in the terminal.
- Compare output contours for side faces between the old and new versions.

### Manual Verification
- Launch the dashboard and visually inspect the Top/Bottom/Left/Right faces to ensure the deformation contours are bounded and realistic (no "flying" spikes).
- Verify that the front/rear faces still look correct.

## Open Questions
- What are the typical dimensions of the narrowest faces in your current model? (e.g., thickness of the side plates).
- Should I prioritize physical accuracy (lowering degree) over smoothness for these narrow parts?
