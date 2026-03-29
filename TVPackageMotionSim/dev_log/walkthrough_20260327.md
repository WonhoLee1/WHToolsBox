# Walkthrough: Detailed Block-level Distortion Report (v9)

This update provides full transparency for structural analysis by reporting on every individual block.

## Key Enhancements

### 1. Granular Block-by-Block Reporting
In addition to the summary table, the simulation now outputs a **detailed breakdown** for every component:
- **Comprehensive Listing**: Every constituent block is listed with its grid coordinates `(i, j, k)`.
- **Individual Metrics**: `Max Bend(deg)` and `Max Twist(deg)` are reported for each block separately.
- **Traceability**: You can now pinpoint exactly which internal blocks are experiencing the highest stress, even if they aren't the component-wide maximum.

### 2. Relative Heatmap Scaling (v8 inherited)
- **Min-Max Contrast**: The visual heatmap remains active, scaling Original Color to RED relative to each component's distortion range.

### 3. Professional Terminal Layout
- **Component Grouping**: Detailed tables are grouped by component name with clear separators.
- **Perfect Alignment**: Column widths are fixed for readability across high-block-count components.

## How to Verify
1.  **Run Simulation**: After completion, scroll up in your terminal to see the **[Detailed Block-by-Block Distortion Breakdown]** section.
2.  **Verify Indices**: Cross-reference the `(i, j, k)` indices in the detailed table with the "Highlighted" block index in the summary table.
3.  **Confirm Alignment**: Verify that the detailed tables are clean, grouped, and properly aligned.

---
> [!NOTE]
> For large assemblies (e.g., 90+ blocks), the detailed report may be long. It is recommended to use a terminal with a scrollback buffer of at least 1000 lines.
