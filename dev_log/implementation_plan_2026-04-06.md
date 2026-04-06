# [FIX] 90-Degree Rotation of Side Faces (Restore Legacy Calibration)

The side faces (Left/Right) were rotated 90 degrees in-plane because the PCA-based basis selection in `ShellDeformationAnalyzer` was independent of the 2D local coordinates (`o_data_hint`). This plan restores the robust calibration strategy found in the original `v2` implementation.

## Proposed Changes

### [ShellDeformationAnalyzer](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/plate_by_markers_v2.py#L404)

#### [MODIFY] `fit_reference_plane(m_data_init, o_data_hint=None)`
- If `o_data_hint` is provided:
  - Create a target 3D layout $P_{target} = [o\_data, 0]$.
  - Use SVD (Kabsch) to find the rotation $R$ that aligns $P_{target} \to (m\_data\_init - center)$.
  - Set `ref_basis = R`. Now $R[0]$ is strictly aligned with $X$ in `o_data_hint`.
- If `o_data_hint` is NOprovided:
  - Fallback to PCA (SVD on markers), but add a simple swap check if `W < H`.

#### [MODIFY] `analyze(m_data_hist, o_data_hint=None)`
- Update call to `fit_reference_plane` to pass the hint during the first-frame initialization.
- This ensures the basis is "locked" to the hint's orientation from the very beginning.

## Detailed Plan

1.  **Refactor `fit_reference_plane`**:
    - Centralize basis derivation.
    - Implement the Kabsch calibration ($P_{target} \cdot R \approx P_{world}$).
2.  **Verify Coordinate Consistency**:
    - Ensure `o_data` columns and `ref_basis` rows match the intended engineering axes.
3.  **Visualization Sync**:
    - Ensure `update_frame` uses this calibrated basis for stable reconstruction.

## User Review Required
> [!IMPORTANT]
> 이 방식은 사용자가 제공한 `o_data_hint`의 방향성을 **진리(Ground Truth)**로 간주합니다. 만약 힌트의 컬럼 순서($X, Y$)가 실제 파트의 형상과 물리적으로 일관되면, 대시보드에서의 90도 회전 현상은 완전히 사라집니다.

## Verification Plan
### Automated Tests
- `python .\TVPackageMotionSim\run_drop_simulator\plate_by_markers_v2.py`
- Verify side faces are taller than they are wide and aligned correctly in World space.
### Manual Verification
- `run_drop_simulation_cases_v5.py` 결과 대시보드에서 Left/Right 면의 정상 부착 여부 확인.
