# [Goal] Refactoring Inter-Component Weld Logic

Currently, the interface welding between `OpenCellCohesive` (Tape), `OpenCell`, and `Chassis` relies on exact index matching `(i, j, 0)`. This logic fails when:
1. Components have different `div` (e.g., Tape is 3x3 while Chassis is 4x4).
2. Components have multiple Z-layers (`div_z > 1`), as the code hardcodes `k=0`.

We will replace this with a robust spatial-proximity matching logic.

## User Review Required

> [!IMPORTANT]
> The new logic will match blocks based on their center coordinates `(cx, cy)`. If the resolutions are significantly different, one Tape block might weld to the nearest single block of the counterpart. This is generally preferred for engineering stability over missing welds entirely.

## Proposed Changes

### [Discrete Builder Component]

#### [MODIFY] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)

- **Identify Max Z-index**: Calculate `max_k_oc`, `max_k_occ`, `max_k_chas` for each component.
- **Implement Spatial Matching Loop**:
    - Iterate through all blocks in `b_occ`.
    - If a block is on the **top layer** of the Tape (`k == max_k_occ`):
        - Find the block in `b_opencell` with the closest `(cx, cy)` on its **bottom layer** (`k == 0`).
        - Create a `<weld>` between Tape's `PZ` site and OpenCell's `NZ` site.
    - If a block is on the **bottom layer** of the Tape (`k == 0`):
        - Find the block in `b_chassis` with the closest `(cx, cy)` on its **top layer** (`k == max_k_chas`).
        - Create a `<weld>` between Tape's `NZ` site and Chassis's `PZ` site.
- **Tolerance**: Use a distance threshold (e.g., block width/2) to ensure we don't weld far-away blocks.

## Open Questions

- Should we allow one Tape block to weld to *multiple* smaller blocks of OpenCell if the resolutions are very different? (Currently, 1-to-1 nearest neighbor is planned for XML simplicity).

## Verification Plan

### Automated Tests
- Run `python diag_markers.py` (updated to check welds if possible) or check the generated XML file manually.
- Run a short simulation and verify that the components don't drift apart (signaling missing welds).

### Manual Verification
- View the generated model in MuJoCo Viewer and select "Welds" in the rendering options to visualize the connections.
