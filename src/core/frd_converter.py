# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from ccx2paraview import Converter


class FrdToVtuConverter:
    def convert(self, frd_filepath: Path) -> Path:
        """
        ccx2paraview를 사용해 FRD → VTU 변환.
        출력 파일: <stem>.0.vtu, <stem>.1.vtu, ... (모드별 1개씩)
        반환값: 출력 파일들의 베이스 경로 (확장자 제외)
        """
        if not frd_filepath.exists():
            print(f"[Error] FRD file not found: {frd_filepath}")
            return None

        logging.basicConfig(level=logging.WARNING)
        c = Converter(str(frd_filepath), ['vtu'])
        c.run()

        return frd_filepath.with_suffix('')
