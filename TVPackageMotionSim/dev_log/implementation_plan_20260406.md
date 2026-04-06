# Implementation Plan - Restoring Correct Face Mapping for WHTOOLS v5 (Backup 2026-04-06)

The current `whts_mapping_D260406.py` assumes a standard MuJoCo Z-Up coordinate system (Top/Bottom = Z, Front/Rear = Y). However, the `run_discrete_builder` and the MuJoCo models in this workspace use a non-standard convention where **Y is Height** and **Z is Depth**.

This plan restores the correct axis mapping in the new, more accurate `_D260406` version of the mapping utility.

## Proposed Changes

### [run_drop_simulator](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator)

#### [MODIFY] [whts_mapping_D260406.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_mapping_D260406.py)
- **Axis Mapping Update**: Correct `get_face_index_logic` to reflect the physical model axes:
    - **Top/Bottom**: Switch to axis `j` (Y).
    - **Front/Rear**: Switch to axis `k` (Z).
- **Normal & Plane Alignment**: Update `normal`, `plane`, and `offsets` definitions to match.
- **SVD Projection Refinement**:
    - Update `h_sign` and `v_sign` for each face to ensure consistent 2D plotting across components.
    - Ensure the normal points "outwards" relative to the part center (handling the inverted Y-axis where applicable).

## Verification Plan

### Automated Tests
- Run `run_drop_simulation_cases_v5.py` and inspect the QtVisualizerV2.
- Verify that the 2D Contour plots show accurate displacement fields for the Top/Bottom/Front/Rear faces.

### Manual Verification
- Check that the surface normals in the 3D view are pointing outwards from the components.
- Ensure the orientation of the 2D plots is intuitive (e.g., Top view is X-Z horizontal-vertical).
