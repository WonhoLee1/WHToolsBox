# UI Global Font Implementation Plan (2026-03-28)

## Goal
Ensure all UI widgets in `PostProcessingUI` use a consistent font ('D2Coding' if available, otherwise 'Malgun Gothic').

## User Review Required
- The font change will apply to all existing widgets and future ones (via `option_add`).
- Matplotlib font will remain 'Malgun Gothic' for better readability in graphs unless otherwise specified.

## Proposed Changes

### [Component] Post-Processing UI (postprocess_ui.py)

#### [MODIFY] [postprocess_ui.py](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/postprocess_ui.py)
- Improve `get_ui_font` to handle font family detection more reliably.
- Add `_apply_font_recursive(widget, font_tuple)` to update fonts of all children recursively.
- Update `_apply_custom_styles` to:
    - Use `tk.Toplevel.option_add` for global font defaults.
    - Call `_apply_font_recursive` to ensure standard `tk` widgets are updated.
    - Update `ttk.Style` global (`.`) font.

## Verification Plan

### Manual Verification
- Launch the Post-Processing UI.
- Verify that labels, buttons, and text areas use 'D2Coding' (if installed) or 'Malgun Gothic'.
- Switch themes via the menu and verify the font persists.
