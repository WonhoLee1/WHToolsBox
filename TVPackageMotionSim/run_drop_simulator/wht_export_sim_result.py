# -*- coding: utf-8 -*-
"""
[WHTOOLS] Drop Simulation Data Exporter
시뮬레이션 완료 후 생성된 pkl 파일을 읽어들여 CSV 데이터 추출 및 Matplotlib 기반 PNG 그래프 저장을 수행합니다.
"""

import sys
from pathlib import Path

# [WHTOOLS] 단독 실행(main) 시 부모 디렉토리를 sys.path에 등록하여 절대 경로 임포트 보장
curr_file = Path(__file__).resolve()
parent_dir = curr_file.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import os
import csv
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# User Guideline: koreanize-matplotlib (하지만 영어로 기록하라는 요구사항 반영)
try:
    import koreanize_matplotlib
except ImportError:
    pass

from run_drop_simulator.whts_data import DropSimResult

logger = logging.getLogger("WHTS_Exporter")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

class SimulationDataExporter:
    def __init__(self, pkl_path: str):
        """
        초기화 시 pkl 파일을 로드합니다.
        
        Args:
            pkl_path (str): 읽어올 simulation_result.pkl 파일 경로
        """
        self.pkl_path = Path(pkl_path)
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {self.pkl_path}")
            
        self.result: DropSimResult = DropSimResult.load(str(self.pkl_path))
        
        # 출력 디렉토리 설정 (result/data 폴더 생성)
        self.output_dir = self.pkl_path.parent / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 플롯 공통 설정
        plt.rcParams.update({'font.size': 9})
        
    def export_all(self, fig_size: Tuple[float, float] = (8.0, 6.0)) -> None:
        """모든 CSV와 PNG 플롯을 생성합니다."""
        logger.info("Starting data export process...")
        self.export_parts_data(fig_size)
        self.export_engineering_data(fig_size)
        logger.info(f"Data export completed. Files saved in {self.output_dir}")

    def export_parts_data(self, fig_size: Tuple[float, float]) -> None:
        """각 파트별(Chassis, Cushion, Cushion-Rigid, OpenCell) 코너 데이터를 분리하여 저장합니다."""
        # part_corner_hist 구조: { "PartName": { "pos": [...], "vel": [...], "acc": [...] } }
        part_hist = getattr(self.result, "part_corner_hist", None)
        if not part_hist:
            logger.warning("No part_corner_hist found in the result. Skipping parts data export.")
            return

        time_hist = self.result.time_history
        axes = ['x', 'y', 'z']
        
        for part_name, data_types in part_hist.items():
            safe_part_name = part_name.lower().replace("-", "_")
            
            for dtype, history in data_types.items():
                if dtype not in ['pos', 'vel', 'acc']:
                    continue
                    
                if not history:
                    continue
                    
                # history shape check: (num_frames, 8 corners, 3 axes)
                history_np = np.array(history)
                num_frames = history_np.shape[0]
                
                # Check frame mismatch
                if num_frames != len(time_hist):
                    logger.warning(f"Frame mismatch for {part_name}-{dtype}: {num_frames} vs {len(time_hist)}")
                    continue
                
                for ax_idx, ax_name in enumerate(axes):
                    filename_csv = f"{safe_part_name}-{dtype}-{ax_name}.csv"
                    filename_png = f"{safe_part_name}-{dtype}-{ax_name}.png"
                    csv_path = self.output_dir / filename_csv
                    png_path = self.output_dir / filename_png
                    
                    # CSV 저장 (UTF-8)
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        header = ["frame", "time"] + [f"C{i+1}" for i in range(8)]
                        writer.writerow(header)
                        
                        for frame in range(num_frames):
                            row = [frame, f"{time_hist[frame]:.5f}"]
                            for corner in range(8):
                                row.append(f"{history_np[frame, corner, ax_idx]:.6f}")
                            writer.writerow(row)
                    
                    # PNG 플롯 저장
                    fig, ax = plt.subplots(figsize=fig_size)
                    for corner in range(8):
                        ax.plot(time_hist, history_np[:, corner, ax_idx], label=f"C{corner+1}")
                    
                    ax.set_title(f"{part_name} {dtype.capitalize()} ({ax_name.upper()})")
                    ax.set_xlabel("Time (s)")
                    ylabel = "Position (m)" if dtype == "pos" else "Velocity (m/s)" if dtype == "vel" else "Acceleration (m/s²)"
                    ax.set_ylabel(ylabel)
                    ax.grid(True, linestyle="--", alpha=0.7)
                    
                    # 레전드 우측 배치
                    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), prop={'size': 9})
                    fig.tight_layout()
                    fig.savefig(png_path, dpi=300)
                    plt.close(fig)

    def export_engineering_data(self, fig_size: Tuple[float, float]) -> None:
        """통합 공학 데이터(Z Position, Impact, Drag 등)를 CSV 및 PNG로 저장합니다."""
        time_hist = self.result.time_history
        num_frames = len(time_hist)
        
        if num_frames == 0:
            return
            
        csv_path = self.output_dir / "engineering.csv"
        
        # 기록할 데이터 수집
        eng_data: Dict[str, np.ndarray] = {}
        if hasattr(self.result, "z_hist") and self.result.z_hist:
            eng_data["z_position"] = np.array(self.result.z_hist)
        if hasattr(self.result, "ground_impact_hist") and self.result.ground_impact_hist:
            eng_data["ground_impact"] = np.array(self.result.ground_impact_hist)
        if hasattr(self.result, "air_drag_hist") and self.result.air_drag_hist:
            eng_data["air_drag"] = np.array(self.result.air_drag_hist)
        if hasattr(self.result, "air_squeeze_hist") and self.result.air_squeeze_hist:
            eng_data["air_squeeze"] = np.array(self.result.air_squeeze_hist)
            
        if not eng_data:
            logger.info("No engineering data to export.")
            return

        # CSV 저장 (UTF-8)
        keys = list(eng_data.keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["frame", "time"] + keys
            writer.writerow(header)
            
            for frame in range(num_frames):
                row = [frame, f"{time_hist[frame]:.5f}"]
                for k in keys:
                    arr = eng_data[k]
                    if frame < len(arr):
                        row.append(f"{arr[frame]:.6f}")
                    else:
                        row.append("0.0")
                writer.writerow(row)
                
        # PNG 플롯 저장 (각 축별 개별 생성)
        for k in keys:
            arr = eng_data[k]
            png_path = self.output_dir / f"engineering-{k}.png"
            fig, ax = plt.subplots(figsize=fig_size)
            
            # 길이 맞춤 보정
            plot_len = min(num_frames, len(arr))
            ax.plot(time_hist[:plot_len], arr[:plot_len], label=k.replace("_", " ").title())
            
            ax.set_title(f"Engineering Metric: {k.replace('_', ' ').title()}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.7)
            
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), prop={'size': 9})
            fig.tight_layout()
            fig.savefig(png_path, dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    # 단위 테스트/단독 실행 지원
    import sys
    if len(sys.argv) > 1:
        target_pkl = sys.argv[1]
        exporter = SimulationDataExporter(target_pkl)
        exporter.export_all()
    else:
        print("Usage: python wht_export_sim_result.py [path_to_simulation_result.pkl]")
