# Goal: Resize Post-Processing UI Banner

The user requested to reduce the size of the title banner in the `PostProcessingUI` to 1/3 of its current size for better layout efficiency.

## Proposed Changes

### [TVPackageMotionSim]

#### [MODIFY] [postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/postprocess_ui.py)

Update the `target_w` variable in `_build_ui()` to be 1/3 of the current value (900 -> 300).

## Verification Plan

### Manual Verification
- Run the post-processing UI and verify that the banner is significantly smaller (300px width).
- Ensure the layout remains balanced with the smaller banner.
