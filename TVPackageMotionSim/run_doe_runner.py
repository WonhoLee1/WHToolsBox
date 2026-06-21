"""
DOERunner — warm-batch subprocess queue manager for parallel DOE execution.
Spawns n_parallel workers, each processing batch_size cases.
"""
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


class DOERunner:
    def __init__(self,
                 doe_definition=None,
                 n_parallel: int = 4,
                 batch_size: int = 10,
                 output_root: str = None,
                 camera_angle: Optional[Tuple[float, float]] = None):
        self.doe_definition = doe_definition
        self.n_parallel = n_parallel
        self.batch_size = batch_size
        self.output_root = output_root or 'results'
        self.camera_angle = camera_angle

    def run(self, doe_table: pd.DataFrame, config_list: List[dict]) -> str:
        """Execute DOE cases in parallel warm-batch subprocesses. Returns output_dir."""
        timestamp = datetime.now().strftime('D%Y%m%d_%H%M%S')
        run_dir = Path(self.output_root) / f'DOE_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save configs per case
        n_cases = len(config_list)
        for i, cfg in enumerate(config_list):
            case_dir = run_dir / f'case_{i:03d}'
            case_dir.mkdir(exist_ok=True)
            if self.camera_angle is not None:
                cfg = dict(cfg)
                cfg['doe_camera_angle'] = self.camera_angle
            with open(case_dir / 'config.pkl', 'wb') as f:
                pickle.dump(cfg, f)

        # Build batch queue
        batches = []
        for start in range(0, n_cases, self.batch_size):
            batches.append(start)

        slots = [None] * self.n_parallel  # (process, batch_start, retry_count)
        slot_meta = [{'batch_start': None, 'retries': 0} for _ in range(self.n_parallel)]
        pending = list(batches)
        done_log = run_dir / 'done_doe_log.txt'
        run_log = run_dir / 'run_doe_log.txt'
        monitor_dir = str(run_dir / 'monitor')

        print(f"\n[DOERunner] Starting {n_cases} cases in {len(batches)} batches "
              f"({self.n_parallel} parallel slots, batch_size={self.batch_size})")
        print(f"[DOERunner] Output: {run_dir}\n")

        while True:
            # Check completed slots
            for slot_id, proc in enumerate(slots):
                if proc is None:
                    continue
                retcode = proc.poll()
                if retcode is None:
                    continue  # still running
                meta = slot_meta[slot_id]
                batch_start = meta['batch_start']
                elapsed = time.time() - meta.get('start_time', time.time())
                if retcode == 0:
                    # Success
                    with open(done_log, 'a', encoding='utf-8') as f:
                        f.write(f"[DONE] batch_start={batch_start} | PID {proc.pid} | "
                                f"elapsed {elapsed:.1f}s | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    slots[slot_id] = None
                    slot_meta[slot_id] = {'batch_start': None, 'retries': 0}
                else:
                    if meta['retries'] < 1:
                        print(f"[DOERunner] Slot {slot_id} failed (batch_start={batch_start}), retrying...")
                        pending.insert(0, batch_start)
                        meta['retries'] += 1
                    else:
                        print(f"[DOERunner] Slot {slot_id} failed twice for batch_start={batch_start}, skipping")
                        with open(done_log, 'a', encoding='utf-8') as f:
                            f.write(f"[FAILED] batch_start={batch_start} | PID {proc.pid} | "
                                    f"elapsed {elapsed:.1f}s\n")
                    slots[slot_id] = None
                    slot_meta[slot_id] = {'batch_start': None, 'retries': meta['retries']}

            # Fill empty slots
            for slot_id in range(self.n_parallel):
                if slots[slot_id] is not None:
                    continue
                if not pending:
                    continue
                batch_start = pending.pop(0)
                cmd = [
                    sys.executable,
                    '--doe-worker',
                    '--batch-start', str(batch_start),
                    '--batch-size', str(self.batch_size),
                    '--run-dir', str(run_dir),
                    '--slot-id', str(slot_id),
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                slots[slot_id] = proc
                slot_meta[slot_id] = {
                    'batch_start': batch_start,
                    'retries': slot_meta[slot_id].get('retries', 0),
                    'start_time': time.time(),
                }

            # Write run_doe_log.txt
            self._write_run_log(run_log, slots, slot_meta, monitor_dir, n_cases)

            # Check completion
            active = [s for s in slots if s is not None]
            if not active and not pending:
                break

            time.sleep(0.5)

        print(f"\n[DOERunner] Complete. Results in: {run_dir}")
        return str(run_dir)

    def _write_run_log(self, run_log, slots, slot_meta, monitor_dir, n_cases):
        lines = [
            f"=== DOE RUN MONITOR [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===",
            f"SLOT | {'PID':>6} | {'TARGET_T':>8} | {'CURRENT_T':>9} | {'FRAME':>6} | STATUS",
        ]
        for slot_id, proc in enumerate(slots):
            if proc is None:
                lines.append(f"  {slot_id}  | {'---':>6} | {'---':>8} | {'---':>9} | {'---':>6} | idle")
                continue
            mon_path = os.path.join(monitor_dir, f'slot_{slot_id}.json')
            try:
                with open(mon_path, 'r', encoding='utf-8') as f:
                    mon = json.load(f)
                pid = mon.get('pid', proc.pid)
                target = mon.get('target_time', 0)
                current = mon.get('current_time', 0)
                frame = mon.get('frame', 0)
                status = mon.get('status', 'running')
                lines.append(f"  {slot_id}  | {pid:>6} | {target:>7.3f}s | {current:>8.3f}s | {frame:>6} | {status}")
            except Exception:
                lines.append(f"  {slot_id}  | {proc.pid:>6} | {'?':>8} | {'?':>9} | {'?':>6} | running")
        try:
            with open(run_log, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception:
            pass
