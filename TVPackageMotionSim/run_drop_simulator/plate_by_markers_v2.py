# -*- coding: utf-8 -*-
"""
[DEPRECATED] WHTOOLS Plate Analysis v2 (Legacy Wrapper)
이 모듈은 더 이상 유지보수되지 않으며, 기능이 분할되어 다음 모듈로 이관되었습니다:
- Engine: whts_multipostprocessor_engine.py
- UI: whts_multipostprocessor_ui.py
- Launcher: whts_multipostprocessor.py

하위 호환성을 위해 내부 클래스들을 신규 모듈에서 임포트하여 노출하지만, 
신규 프로젝트에서는 whts_multipostprocessor_* 모듈을 직접 사용하십시오.
"""

import warnings
from .whts_multipostprocessor_engine import (
    PlateConfig,
    AlignmentManager,
    AdvancedPlateOptimizer,
    PlateMechanicsSolver,
    ShellDeformationAnalyzer,
    PlateAssemblyManager,
    scale_result_to_mm
)
from .whts_multipostprocessor_ui import (
    PlotSlotConfig,
    DashboardConfig,
    QtVisualizerV2
)

warnings.warn(
    "plate_by_markers_v2 is deprecated and will be removed in future versions. "
    "Please use whts_multipostprocessor_engine or whts_multipostprocessor_ui instead.",
    DeprecationWarning, stacklevel=2
)

if __name__ == "__main__":
    import sys
    from PySide6 import QtWidgets
    print("[WARNING] plate_by_markers_v2.py is a LEGACY wrapper. Redirecting to whts_multipostprocessor.py...")
    # 신규 통합 실행기 메인 로직 실행
    from .whts_multipostprocessor import main
    main()
