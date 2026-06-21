# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CALCULIX_EXE = r"D:\SOFTWARE\calculix_2.23_4win\ccx_static.exe"
WORKSPACE_DIR = BASE_DIR / "workspace"


@dataclass
class ModalAnalysisConfig:
    job_name: str = "modal_job"
    num_modes: int = 10
    # Material
    E: float = 210000.0       # MPa
    nu: float = 0.3
    rho: float = 7.85e-9      # ton/mm³
    # Shell
    thickness: float = 2.0    # mm
    elset_name: str = "ALL_SHELLS"


@dataclass
class TrayGeometryConfig:
    length: float = 100.0
    width: float = 50.0
    mesh_size: float = 5.0
    elset_name: str = "TrayShell"
