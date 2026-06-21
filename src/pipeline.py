# -*- coding: utf-8 -*-
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.core.config import CALCULIX_EXE, WORKSPACE_DIR, ModalAnalysisConfig, TrayGeometryConfig
from src.core.mesher import GmshMesher
from src.core.model_builder import CalculixModelBuilder
from src.core.solver import CalculixSolver
from src.core.dat_parser import CalculixDatParser
from src.core.frd_converter import FrdToVtuConverter
from src.core.viewer import ModeShapeViewer


class AutoCalculixPipeline:
    """
    CalculiX 기반 고유진동수 해석 파이프라인.

    진입점
    ------
    run_with_meshing()   : Gmsh로 형상/메쉬 생성 후 해석 → 첫 유연체 모드 표시
    run_from_inp()       : 기존 메쉬 INP 파일로 해석   → 첫 유연체 모드 표시
    run_from_external()  : Abaqus INP / OptiStruct FEM·BDF 파일로 해석
                           → 모드 목록 출력 후 인터랙티브 선택 루프
    """

    def __init__(self, workspace: Path = WORKSPACE_DIR):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._solver    = CalculixSolver(CALCULIX_EXE)
        self._parser    = CalculixDatParser()
        self._converter = FrdToVtuConverter()
        self._viewer    = ModeShapeViewer()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_with_meshing(
        self,
        analysis: ModalAnalysisConfig = None,
        geometry: TrayGeometryConfig  = None,
        show_gui: bool = False,
    ):
        """Gmsh로 tray 메쉬를 생성한 뒤 전체 파이프라인을 실행합니다."""
        if analysis is None: analysis = ModalAnalysisConfig()
        if geometry is None: geometry = TrayGeometryConfig()
        analysis.elset_name = geometry.elset_name

        print("=== 1. Generating Mesh ===")
        mesher = GmshMesher(self.workspace)
        mesher.create_tray_geometry_and_mesh(geometry)
        mesh_inp = mesher.export_mesh_inp("mesh.inp", show_gui=show_gui)

        frequencies, vtu_base = self._execute_analysis(mesh_inp, analysis)
        first_flex = 8 if len(frequencies) >= 8 else 1
        self._viewer.show_mode(vtu_base, mode_number=first_flex)

    def run_from_inp(self, mesh_inp_file: Path, analysis: ModalAnalysisConfig = None):
        """기존 메쉬 INP 파일을 받아 고유진동수 해석을 수행합니다."""
        if analysis is None: analysis = ModalAnalysisConfig()
        mesh_inp_file = Path(mesh_inp_file)
        if not mesh_inp_file.exists():
            print(f"[Error] INP file not found: {mesh_inp_file}")
            return

        frequencies, vtu_base = self._execute_analysis(mesh_inp_file, analysis)
        first_flex = 8 if len(frequencies) >= 8 else 1
        self._viewer.show_mode(vtu_base, mode_number=first_flex)

    def run_from_external(self, mesh_file: Path, analysis: ModalAnalysisConfig = None):
        """
        Abaqus INP 또는 OptiStruct/Nastran FEM·BDF 파일을 입력으로
        모달 해석 후 인터랙티브 모드 선택 루프를 실행합니다.
        """
        from src.core.mesh_loader import ExternalMeshLoader

        if analysis is None: analysis = ModalAnalysisConfig()
        mesh_file = Path(mesh_file)

        print("=== 1. Loading External Mesh ===")
        loader = ExternalMeshLoader(self.workspace)
        mesh_inp, elset_name = loader.load(mesh_file)
        analysis.elset_name = elset_name
        if analysis.job_name == "modal_job":
            analysis.job_name = mesh_file.stem

        frequencies, vtu_base = self._execute_analysis(mesh_inp, analysis)
        self._interactive_loop(frequencies, vtu_base)

    # ------------------------------------------------------------------
    # Core analysis (Steps 2-5)
    # ------------------------------------------------------------------

    def _execute_analysis(self, mesh_inp: Path, config: ModalAnalysisConfig):
        """
        Steps 2-5 실행 후 (frequencies, vtu_base) 반환.
        """
        print("=== 2. Building Master INP ===")
        builder = CalculixModelBuilder(self.workspace)
        builder.build_modal_analysis(mesh_inp, config)

        print("=== 3. Running CalculiX Solver ===")
        self._solver.run(config.job_name, self.workspace)

        print("=== 4. Parsing DAT File ===")
        dat_file = self.workspace / f"{config.job_name}.dat"
        frequencies = self._parser.extract_frequencies(dat_file)
        for f_info in frequencies:
            print(f"  Mode {f_info['mode']:2d}: {f_info['hz']:10.3f} Hz")

        print("=== 5. Converting FRD to VTU ===")
        frd_file = self.workspace / f"{config.job_name}.frd"
        vtu_base = self._converter.convert(frd_file)

        return frequencies, vtu_base

    # ------------------------------------------------------------------
    # Interactive mode selection loop
    # ------------------------------------------------------------------

    def _interactive_loop(self, frequencies: list, vtu_base: Path):
        """모드 목록을 출력하고 사용자가 선택한 모드를 반복 시각화합니다."""
        n = len(frequencies)
        if n == 0:
            print("[Error] No frequencies found.")
            return

        print("\n" + "=" * 48)
        print(f"  {'Mode':>4}   {'Frequency (Hz)':>14}")
        print("-" * 48)
        for f in frequencies:
            print(f"  {f['mode']:4d}   {f['hz']:14.3f}")
        print("=" * 48)
        print("  Ctrl+C to exit")

        while True:
            try:
                s = input(f"\nSelect mode [1-{n}]: ").strip()
                mode = int(s)
                if 1 <= mode <= n:
                    self._viewer.show_mode(vtu_base, mode_number=mode)
                else:
                    print(f"  Please enter a number between 1 and {n}.")
            except ValueError:
                print("  Enter a valid mode number.")
            except KeyboardInterrupt:
                print("\nExiting.")
                break


# ----------------------------------------------------------------------
# 실행 예제 (원하는 케이스의 주석을 해제하여 실행)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = AutoCalculixPipeline()

    # ------------------------------------------------------------------
    # 케이스 1: Gmsh로 Tray 형상 직접 생성 → 모달 해석 → 첫 유연체 모드 표시
    # ------------------------------------------------------------------
    analysis_cfg = ModalAnalysisConfig(
        job_name  = "test_tray",
        num_modes = 10,
        thickness = 2.0,        # mm
        E         = 210000.0,   # MPa (Steel)
        nu        = 0.3,
        rho       = 7.85e-9,    # ton/mm³
    )
    geo_cfg = TrayGeometryConfig(
        length    = 100.0,      # mm
        width     = 50.0,       # mm
        mesh_size = 5.0,        # mm
    )
    pipeline.run_with_meshing(analysis=analysis_cfg, geometry=geo_cfg)

    # ------------------------------------------------------------------
    # 케이스 2: 기존 Abaqus INP 메쉬 파일 → 모달 해석 → 첫 유연체 모드 표시
    # ------------------------------------------------------------------
    # pipeline.run_from_inp(
    #     mesh_inp_file = r"D:\path\to\your_mesh.inp",
    #     analysis = ModalAnalysisConfig(
    #         job_name  = "my_model",
    #         num_modes = 10,
    #         thickness = 1.5,
    #         elset_name = "Shell_Part",   # INP 파일 내 *ELEMENT 의 ELSET 이름
    #     ),
    # )

    # ------------------------------------------------------------------
    # 케이스 3: 외부 메쉬 파일 (Abaqus INP / OptiStruct FEM·BDF)
    #           → 모달 해석 → 인터랙티브 모드 선택 루프 (Ctrl+C 로 종료)
    #
    # 입력 파일 최소 요건
    #   Abaqus INP  : *NODE + *ELEMENT (TYPE=S3/S4/S4R 등 셸 타입)
    #   OptiStruct  : GRID + CQUAD4/CTRIA3 (PSHELL·MAT·SPC 없어도 됨)
    # ------------------------------------------------------------------
    # pipeline.run_from_external(
    #     mesh_file = r"D:\path\to\your_model.fem",   # or .inp / .bdf
    #     analysis  = ModalAnalysisConfig(
    #         job_name  = "my_model",
    #         num_modes = 10,
    #         thickness = 1.5,
    #     ),
    # )
