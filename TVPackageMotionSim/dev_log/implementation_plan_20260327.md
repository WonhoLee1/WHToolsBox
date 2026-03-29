# Post-Processing UI & Rank-based Heatmaps (v10)

This phase introduces a dedicated analysis environment and a fairer, rank-based visualization strategy.

## Proposed Changes

### [MuJoCo Simulation]

#### [MODIFY] [run_drop_simulation_v3.py](file:///c:/Users/GOODMAN/WHToolsBox/test_box_mujoco/run_drop_simulation_v3.py)

- **Post-Processing Architecture**:
    - Add a `PostProcessingUI` class (Tkinter `Toplevel`).
    - Remove automatic MuJoCo coloring from `_finalize_simulation`.
    - Modify `_finalize_simulation` to open the `PostProcessingUI` upon completion.
- **Rank-based Distortion Coloring**:
    - Instead of linear scaling, sort blocks by `(Bend + Twist) / 2`.
    - Assign colors based on rank: $f = rank / (N - 1)$.
    - This ensures that exactly one block is pure RED and the rest are distributed across the full spectrum.
- **2D Distortion Mapping (Matplotlib)**:
    - Button "Distortion Map" in UI triggers a 10x5 figure.
    - Two subplots: `Bend` (Left) and `Twist` (Right).
    - Map `(i, j, k)` indices to a 2D grid (using Max/Sum along the Z-axis if 3D).
    - Apply `interp` (Bilinear/Cubic) for smooth transitions.
    - Set all fonts to 9pt.
- **UI Styling**:
    - Re-use the [WHTOOLS] Banner and button styles from `ConfigEditor`.

## Technical Details

### Rank-based Factor calculation
```python
scores = sorted(block_scores.items(), key=lambda x: x[1])
for rank, (grid_idx, score) in enumerate(scores):
    f = rank / (len(scores) - 1) if len(scores) > 1 else 1.0
    # Apply color interpolation...
```

### Matplotlib 2D Heatmap
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# Prepare grid data Z[ny, nx] from block_scores
im1 = ax1.imshow(grid_bend, interpolation='bilinear', cmap='Reds')
im2 = ax2.imshow(grid_twist, interpolation='bilinear', cmap='Reds')
# Set font properties to 9pt
```

## Verification Plan

### Manual Verification
1. **Post-UI**: Verify the UI appears only after simulation ends.
2. **Heatmap Contrast**: Confirm the MuJoCo blocks show a full spectrum regardless of how close the absolute values are.
3. **Matplotlib Plot**: Confirm the 2D plot appears with two subplots and the correct figure size/fonts.
