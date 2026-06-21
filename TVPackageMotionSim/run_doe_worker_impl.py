"""
DOE Worker Implementation — runs K simulation cases as a batch subprocess.
Entry: doe_worker_main() called from run_drop_simulation_cases_v6.py when --doe-worker in sys.argv
"""
import argparse
import json
import os
import pickle
import sys
import time
import tempfile
import traceback
from pathlib import Path


def _setup_sys_path():
    """Add WHToolsBox root to sys.path — works in both dev and PyInstaller frozen builds."""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        # This file is at TVPackageMotionSim/run_doe_worker_impl.py
        # WHToolsBox root is two levels up
        this_file = os.path.abspath(__file__)
        base = os.path.dirname(os.path.dirname(this_file))
    if base not in sys.path:
        sys.path.insert(0, base)
    tv_dir = os.path.join(base, 'TVPackageMotionSim')
    if tv_dir not in sys.path:
        sys.path.insert(0, tv_dir)


def _write_monitor(monitor_dir: str, slot_id: int, data: dict):
    """Atomically write monitor JSON (temp→rename to avoid partial reads)."""
    os.makedirs(monitor_dir, exist_ok=True)
    slot_path = os.path.join(monitor_dir, f'slot_{slot_id}.json')
    tmp_path = slot_path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp_path, slot_path)
    except Exception:
        pass  # Monitor failure should not crash the simulation


def doe_worker_main():
    _setup_sys_path()

    parser = argparse.ArgumentParser(description='DOE Worker')
    parser.add_argument('--doe-worker', action='store_true')
    parser.add_argument('--batch-start', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--slot-id', type=int, default=0)
    parser.add_argument('--camera-angle', type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    monitor_dir = str(run_dir / 'monitor')
    batch_start = args.batch_start
    batch_size = args.batch_size
    slot_id = args.slot_id

    # Process each case in the batch
    for i in range(batch_start, batch_start + batch_size):
        case_dir = run_dir / f'case_{i:03d}'
        config_path = case_dir / 'config.pkl'

        if not config_path.exists():
            break  # No more cases

        try:
            with open(config_path, 'rb') as f:
                cfg = pickle.load(f)

            # Import here (after sys.path setup) to avoid eager JAX/PySide6 imports
            from run_drop_simulator import DropSimulator

            target_time = cfg.get('sim_duration', 2.0)
            start_time = time.time()

            _write_monitor(monitor_dir, slot_id, {
                'pid': os.getpid(),
                'case_idx': i,
                'target_time': target_time,
                'current_time': 0.0,
                'frame': 0,
                'status': 'running',
            })

            sim = DropSimulator(config=cfg)

            # Monkey-patch step callback for monitor updates
            _orig_step = getattr(sim, '_on_step_complete', None)
            last_update = [0.0]

            def _step_monitor(sim_time, frame):
                now = time.time()
                if now - last_update[0] > 0.5:
                    _write_monitor(monitor_dir, slot_id, {
                        'pid': os.getpid(),
                        'case_idx': i,
                        'target_time': target_time,
                        'current_time': sim_time,
                        'frame': frame,
                        'status': 'running',
                    })
                    last_update[0] = now
                if _orig_step:
                    _orig_step(sim_time, frame)

            sim._on_step_complete = _step_monitor

            sim.simulate()

            # Save result
            result_path = case_dir / 'result.pkl'
            if sim.result is not None:
                with open(result_path, 'wb') as f:
                    pickle.dump(sim.result, f)

            _write_monitor(monitor_dir, slot_id, {
                'pid': os.getpid(),
                'case_idx': i,
                'target_time': target_time,
                'current_time': target_time,
                'frame': -1,
                'status': 'done',
                'elapsed': time.time() - start_time,
            })

        except Exception as e:
            err_path = case_dir / 'error.txt'
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write(f"Case {i} failed:\n{traceback.format_exc()}")
            _write_monitor(monitor_dir, slot_id, {
                'pid': os.getpid(),
                'case_idx': i,
                'status': 'error',
                'error': str(e),
            })
            sys.exit(1)  # Signal failure to DOERunner

    sys.exit(0)
