# Implementation Plan - Full Inertia Tensor Balancing (2026-05-14)

The goal is to support the full 3x3 inertia tensor (6 unique components) in the assembly inertia calculation and the auto-balancing logic.

## Proposed Changes

### [MODIFY] [whtb_base.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_base.py)
- Update `calculate_inertia` to calculate and return 6 components: `[ixx, iyy, izz, ixy, ixz, iyz]`.
- Use the Parallel Axis Theorem for products of inertia: $I_{xy} = \sum m(x \cdot y)$ (relative to CoG).
- Update internal `_collect` function to handle these 6 components.

### [MODIFY] [whts_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulator/whts_utils.py)
- Update `calculate_required_aux_masses` to support 6-element `target_moi`.
- Implement an asymmetric distribution logic for 8 auxiliary masses to match the products of inertia.
- **Algorithm**:
    1. Calculate required $dx, dy, dz$ from diagonal terms (same as before).
    2. Solve for individual masses $m_1, \dots, m_8$ at $(\pm dx, \pm dy, \pm dz)$ to match the off-diagonal terms $I_{xy}, I_{xz}, I_{yz}$ and maintain CoG at the center.
    3. Ensure masses remain positive (clamping/scaling if necessary).

### [MODIFY] [whtb_physics.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_physics.py)
- Update `_print_physics_report` to display all 6 inertia components in the table.
- Adjust table column widths to accommodate the extra data.

### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- Ensure the XML output uses the correct inertial definition if needed.

## Verification Plan

### Manual Verification
- Run `run_drop_simulation_cases_v6.py` and verify `Final (Balanced)` inertia.
