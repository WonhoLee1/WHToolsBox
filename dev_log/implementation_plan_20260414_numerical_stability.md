# Numerical Stability Patch for High-Scale Plate Analysis

Implementing coordinate normalization to resolve numerical explosion (NaN/Inf) occurring at the `mm` coordinate scale (~1000mm).

## User Review Required

> [!IMPORTANT]
> - **Coordinate Scaling**: The solver will internally map physical coordinates to a normalized range ([-1, 1]) for matrix inversion. 
> - **Result Consistency**: This change is purely numerical and will not change the physical output values (displacement, stress) in the UI, but will ensure they appear correctly for large parts.

## Proposed Changes

### High-Fidelity Analyzer Components

#### [MODIFY] [plate_by_markers.py](file:///c:/Users/GOODMAN/WHToolsBox/plate_by_markers.py)

1.  **PlateMechanicsSolver.setup_mesh**:
    - Calculate and store `x_scale` and `y_scale` based on the local coordinate bounds.
2.  **PlateMechanicsSolver.evaluate_batch**:
    - Normalize grid coordinates before evaluating the basis matrix.
    - Apply the **Chain Rule** to curvature calculation: $\kappa_{xx} = \frac{\Delta w}{\Delta x^2} = \frac{1}{S_x^2} \frac{\partial^2 w_{norm}}{\partial x_{norm}^2}$.
3.  **KirchhoffPlateOptimizer.get_hessian_basis**:
    - Fix the internal `stack` logic to ensure correct broadcasting during JAX `vmap` calls.
4.  **KirchhoffPlateOptimizer.solve_analytical**:
    - Implement internal normalization of reference points.
    - Scale the Hessian basis contributions ($B_{xx}, B_{yy}, B_{xy}$) to match the physical scale using the cached scale factors.

## Open Questions

- None at this stage. The normalization is a mathematically required step for high-order polynomial fitting on larger domains.

## Verification Plan

### Automated Tests
- Run `plate_by_markers.py` with the simulation data.
- **Success Criteria**: 
  - `Physical Scale Check` confirms mm range.
  - The 3D Surface Mesh becomes visible in the window.
  - The displacement values match the expected simulation range.

### Manual Verification
- Visual inspection of the dashboard to ensure the plate deformation logic (Ripple, Bending) is physically intuitive.
