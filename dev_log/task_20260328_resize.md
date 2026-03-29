# Task: Fixing PostProcessingUI AttributeError & Resizing Banner

- [x] Research and Planning
    - [x] Identify the cause of `AttributeError: 'PostProcessingUI' object has no attribute '_build_sidebar'`
    - [x] Verify if `_build_sidebar` should be replaced by `_build_ui`
- [x] Implementation
    - [x] Update `postprocess_ui.py` to call `_build_ui` instead of `_build_sidebar`
    - [x] Ensure all other method calls in `__init__` are correct
- [x] Verification
    - [x] Run the script to verify the UI opens correctly

- [ ] Resizing Post-Processing UI Banner
    - [ ] Update `postprocess_ui.py` to reduce banner width to 1/3 (900 -> 300)
    - [ ] Create and update implementation plan and walkthrough
