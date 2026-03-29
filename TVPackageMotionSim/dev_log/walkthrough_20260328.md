# UI Global Font Application Walkthrough (2026-03-28)

## Changes Made
Implemented a robust font management system in `PostProcessingUI` to ensure a consistent look and feel across different UI themes.

### 1. Font Detection Improvement
- Updated `get_ui_font` to cache the detected font family.
- Added partial string matching to detect variations of 'D2Coding' (e.g., 'D2Coding v.1.3').
- Fallback to 'Malgun Gothic' remains as a secondary option.

### 2. Recursive Font Application
- Added `_apply_font_recursive` method that traverses the entire widget tree.
- This ensures even standard `tk` widgets (like labels inside frames) that might not perfectly follow `ttk.Style` are updated.

### 3. Option DB Integration
- Used `option_add("*Font", ...)` to set the default font for any newly created widgets.

## Verification Results
- **Font Consistency**: Verified that all labels, buttons, and text areas now use the same font family.
- **Theme Resilience**: Verified that changing the UI theme via the menu triggers a re-application of the font, maintaining consistency.

> [!TIP]
> If you install a new version of D2Coding, the UI will automatically pick it up upon the next restart thanks to the improved detection logic.
