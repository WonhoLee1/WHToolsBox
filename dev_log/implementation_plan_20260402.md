# 3-Way Split UI with Global & Local Views

This plan expands the UI to a 3-way split to visualize the **Global Motion** (Spatial) and **Local Deformation** (Material) side-by-side, along with the analytical plots.

## User Review Required

> [!IMPORTANT]
> - **3-Way Split Layout**: 
>   - **Panel 1 (Left)**: Global View (Spatial). Shows the raw rigid body motion (Rotation + Translation) of the plate and markers.
>   - **Panel 2 (Middle)**: Local View (Material). Shows only the deformation relative to the initial frame.
>   - **Panel 3 (Right)**: Matplotlib Analysis. Shows strain map and energy/stress curves.
> - **Coordinate Sync**: Both 3D views will be synchronized (if possible) or have optimized camera settings to show the scale of motion.
> - **Inverse Transformation**: The mesh in the Global View will be transformed using the inverse Kabsch result to match the raw marker positions.

## Proposed Changes

### [Component] `plate_by_markers.py`

#### [MODIFY] [plate_by_markers.py](file:///c:/Users/GOODMAN/WHToolsBox/plate_by_markers.py)

- **`QtVisualizer` Refactoring**:
  - Update `_init_ui` to add two `QtInteractor` widgets to the `QSplitter`.
  - Add `_setup_global_view()` and `_setup_local_view()`.
- **`update_frame` Logic**:
  - Calculate **Global Mesh Points**:
    - $P_{global} = (P_{local} \cdot L^{T} + \text{centroid}_0 - c_P) \cdot R + c_Q$.
  - Update the **Left Plotter** (Global) with $P_{global}$ and raw markers ($M_{global}$).
  - Update the **Middle Plotter** (Local) with $P_{local}$ and aligned markers ($M_{local}$).
  - Update Matplotlib as usual.
- **Camera Management**:
  - Set the Global View's camera or bounds based on the global raw data range to ensure the motion is visible within the frame.

## Open Questions

> [!QUESTION]
> 1. **Camera Synchronization**: Should the cameras of the Global and Local views move together (Sync)? Or should they be independent? (I recommend independent so you can zoom in on the deformation while seeing the overall motion from afar). 
> 2. **Colors**: Should the Global view also be colored by strain? (I recommend YES, to show the stress state even during large motion).

## Verification Plan

### Automated Tests
- Run with rich test data (rotation + translation).
- Observe if the left panel shows the plate flying/rotating while the middle panel shows it bending in place.

### Manual Verification
- Visual check of the 3-splitter layout stability.
- Verify the coordinate math by checking if marker spheres in the Global view align with the corners of the plate.
