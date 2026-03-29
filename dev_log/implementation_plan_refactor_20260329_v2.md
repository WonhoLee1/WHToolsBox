# [Refactoring] run_discrete_builder Modularization (whtb_ Prefix version)

The current `run_discrete_builder/__init__.py` (1538 lines) is a "god file" that handles configuration, utilities, body definitions, and model building. This refactoring will split these responsibilities into separate, well-defined modules with the `whtb_` prefix for better maintainability and readability.

## User Review Required

> [!IMPORTANT]
> - This refactoring changes the internal structure of the `run_discrete_builder` package. 
> - External scripts using `run_discrete_builder.create_model` will still work fine as we will maintain the public API in `__init__.py`.
> - Any internal script directly importing classes (e.g., `from run_discrete_builder import BCushion`) should still work if we export them correctly in `__init__.py`.
> - All new file names will be prefixed with `whtb_`.

## Proposed Changes

### [run_discrete_builder] package

Separating responsibilities into specialized modules with `whtb_` prefix.

---

#### [NEW] [whtb_utils.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_utils.py)
- Move `get_local_pose` and `calculate_solref` here.
- These are pure mathematical utility functions used by the builder and config modules.

#### [NEW] [whtb_config.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_config.py)
- Move `get_default_config` and `parse_drop_target` here.
- Handles all parameter defaults, synchronization, and drop target parsing.

#### [NEW] [whtb_base.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_base.py)
- Move `DiscreteBlock` and `BaseDiscreteBody` classes here.
- This serves as the foundation for all 3D discrete bodies.

#### [NEW] [whtb_models.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_models.py)
- Move all specialized body classes here:
  - `BPaperBox`, `BCushion`, `BOpenCellCohesive`, `BOpenCell`, `BChassis`, `BAuxBoxMass`, `BUnitBlock`.
- Imports `whtb_base` for the parent classes and `DiscreteBlock`.

#### [NEW] [whtb_builder.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/whtb_builder.py)
- Move `create_model` and `get_single_body_instance` here.
- Acts as the orchestrator that assembles components using a config.
- Imports all components from `whtb_models.py` and configuration logic from `whtb_config.py`.

#### [MODIFY] [__init__.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_discrete_builder/__init__.py)
- Replace the massive implementation with selective imports from `whtb_*` modules.
- Expose `create_model`, `get_default_config`, and `get_single_body_instance` for backward compatibility.

---

## Open Questions

- Should we also keep the `if __name__ == "__main__":` block in `whtb_builder.py`?
  - *Recommendation*: Keep it for validation purposes.

## Verification Plan

### Automated Tests
- Run `python -m run_discrete_builder.whtb_builder` to verify XML generation still works.
- Verify the generated XML (`test_shapes_check.xml`) matches the expected structure.

### Manual Verification
- Check if `run_drop_simulation_v3.py` still runs correctly without import errors.
- Inspect the log output of `create_model` to ensure reports are still generated correctly.
