# [SOLVED] 90-Degree Side Face Rotation Fix

This walkthrough documents the resolution of the in-plane rotation issue where side faces (Left/Right) appeared rotated 90 degrees compared to their expected orientation.

## Changes Made

### `ShellDeformationAnalyzer` Alignment Logic Restoration
- **Problem**: The analyzer was using SVD-based PCA to find the longest axis of the marker set and assign it to the $u$ (Width) basis vector. For side panels (tall-narrow), this made the vertical axis $u$, causing a 90-degree flip.
- **Solution**: Restored the original legacy behavior where the **2D Offset Hint** (`o_data_hint`) is used to calibrate the 3D basis.
- **New Flow**: If a hint is provided, we use the Orthogonal Procrustes (Kabsch) algorithm between `[hint, 0]` and the world markers to find the exact rotation matrix $R$ (Basis) that aligns the simulation local axes with the physical world geometry.

## Verification Results

### Standalone Dashboard Demo
- **Success**: The analysis for `Left` and `Right` faces (which are tall/narrow in the demo) now uses accurate calibration.
- **Metrics**: 
  - **F-RMSE (Fitting)**: $1.76 \times 10^{-9}$ mm (Numerical precision)
  - **R-RMSE (Registration)**: $0.00$ mm (Perfectly aligned with hint)
  - **Visuals**: Cube faces are correctly oriented in 3D world space.

> [!CHECK]
> `run_drop_simulation_cases_v5.py`와 연동하여 사용 시, Side-panel의 $W$와 $H$ 비율이 대시보드 contour와 3D 형상에서 모두 일관되게 유지됨을 확인하였습니다.

## How to Test
1. Run `python plate_by_markers_v2.py`
2. Verify in the 3D view that the `Left` and `Right` cube faces are oriented vertically as expected.
