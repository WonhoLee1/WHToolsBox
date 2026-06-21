# -*- coding: utf-8 -*-
from pathlib import Path
from src.core.config import ModalAnalysisConfig


class CalculixModelBuilder:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def build_modal_analysis(self, mesh_inp_file: Path, config: ModalAnalysisConfig) -> Path:
        """
        메쉬 INP를 *INCLUDE하는 Master INP를 생성합니다.
        mesh_inp_file은 workspace 안에 있어야 합니다 (CalculiX가 상대경로로 INCLUDE).
        """
        master_inp_path = self.workspace / f"{config.job_name}.inp"

        with open(master_inp_path, 'w', encoding='utf-8') as f:
            f.write(f"*HEADING\nModal Analysis: {config.job_name}\n")
            f.write(f"*INCLUDE, INPUT={mesh_inp_file.name}\n\n")

            f.write(f"*MATERIAL, NAME=MAT1\n")
            f.write(f"*ELASTIC\n{config.E}, {config.nu}\n")
            f.write(f"*DENSITY\n{config.rho}\n\n")

            f.write(f"*SHELL SECTION, ELSET={config.elset_name}, MATERIAL=MAT1\n")
            f.write(f"{config.thickness}\n\n")

            f.write("*STEP\n*FREQUENCY\n")
            f.write(f"{config.num_modes}\n")
            f.write("*NODE FILE\nU\n")
            f.write("*END STEP\n")

        return master_inp_path
