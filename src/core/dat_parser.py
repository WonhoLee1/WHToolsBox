# -*- coding: utf-8 -*-
from pathlib import Path

class CalculixDatParser:
    def extract_frequencies(self, dat_filepath: Path):
        """
        CalculiX .dat 파일에서 EIGENVALUE OUTPUT 섹션을 찾아 고유진동수를 추출합니다.
        """
        frequencies = []
        if not dat_filepath.exists():
            print(f"[Error] DAT file not found: {dat_filepath}")
            return frequencies

        with open(dat_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        parsing_active = False
        for line in lines:
            if "E I G E N V A L U E   O U T P U T" in line:
                parsing_active = True
                continue
            
            if parsing_active:
                parts = line.split()
                # 고유진동수 출력 줄 포맷: MODE NO | EIGENVALUE | FREQ(RAD/TIME) | FREQ(CYCLES/TIME) | MASS...
                if len(parts) >= 4 and parts[0].isdigit():
                    mode_no = int(parts[0])
                    
                    # 고유진동수 표는 1번부터 순차적으로 증가합니다. 
                    # 만약 다른 표(Participation factor 등)로 넘어가서 다시 1번 모드가 나타나면 파싱을 중단합니다.
                    if mode_no == len(frequencies) + 1:
                        hz = float(parts[3]) # 4번째 컬럼이 Hz(Cycles/Time) 단위입니다.
                        frequencies.append({"mode": mode_no, "hz": hz})
                    elif len(frequencies) > 0:
                        break # 다른 섹션의 테이블 시작됨
        
        return frequencies