# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path

class CalculixSolver:
    def __init__(self, executable_path: str):
        self.executable_path = executable_path
        
    def run(self, job_name: str, workspace: Path):
        """
        주어진 workspace 디렉토리에서 CalculiX 솔버를 실행합니다.
        """
        exe_path = Path(self.executable_path)
        if not exe_path.exists():
            print(f"[Error] CalculiX 실행 파일을 찾을 수 없습니다: {exe_path}")
            print("설정(config.py)의 CALCULIX_EXE 경로를 탐색기에서 다시 확인해 주세요.")
            return
            
        print(f"[Solver] Starting CalculiX job: {job_name}")
        command = [str(exe_path), job_name]
        
        try:
            # calculix는 job_name.inp 파일을 읽으므로 cwd를 workspace로 지정해야 함
            subprocess.run(command, cwd=workspace, capture_output=True, text=True, check=True)
            print("[Solver] CalculiX run completed successfully.")
        except subprocess.CalledProcessError as e:
            print("[Solver] CalculiX run failed!")
            print("=== STDOUT (Error Details) ===")
            print(e.stdout)
            print("=== STDERR ===")
            print(e.stderr)