"""DOE Results Analysis — load, group, and visualize DOE results."""
import glob
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_doe_results(doe_output_dir: str) -> List[Dict[str, Any]]:
    """Load all result pkl files from a DOE output directory."""
    doe_path = Path(doe_output_dir)
    results = []

    # Find all case directories
    case_dirs = sorted(doe_path.glob('case_*'))
    for case_dir in case_dirs:
        # Find result pkl
        pkl_files = list(case_dir.glob('result*.pkl')) + list(case_dir.glob('result.pkl'))
        if not pkl_files:
            continue
        pkl_path = pkl_files[0]
        try:
            with open(pkl_path, 'rb') as f:
                result = pickle.load(f)
            results.append({
                'case_dir': str(case_dir),
                'case_idx': int(case_dir.name.split('_')[-1]),
                'result': result,
                'pkl_path': str(pkl_path),
                'contact_seq': getattr(result, 'contact_seq', None),
            })
        except Exception as e:
            warnings.warn(f"Could not load {pkl_path}: {e}")

    print(f"[run_doe_analysis] Loaded {len(results)} results from {doe_output_dir}")
    return results


def group_by_contact_seq(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by ContactSeqStr."""
    groups: Dict[str, List[Dict]] = {}
    for r in results:
        seq = r.get('contact_seq')
        if seq and isinstance(seq, dict):
            key = seq.get('ContactSeqStr', 'unknown')
        else:
            key = 'no_contact_data'
        groups.setdefault(key, []).append(r)
    return groups


def _get_max_corner_forces(result) -> Dict[str, float]:
    """Extract max contact force per corner from result."""
    raw = getattr(result, 'corner_impact_hist', None)
    if raw is None:
        return {}
    cih = np.asarray(raw)
    forces = {}
    for i in range(min(8, cih.shape[1])):
        forces[f'C{i+1}'] = float(np.max(cih[:, i]))
    return forces


def plot_group_bar(results: List[Dict],
                   output_dir: str = None,
                   show: bool = True) -> None:
    """Bar chart: group (ContactSeqStr) x corner max contact force with 1-sigma error bars."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[run_doe_analysis] matplotlib not available")
        return

    groups = group_by_contact_seq(results)
    corners = [f'C{i+1}' for i in range(8)]
    group_names = sorted(groups.keys())

    fig, ax = plt.subplots(figsize=(max(10, len(group_names) * 2), 6))
    colors = cm.tab10.colors
    x = np.arange(len(group_names))
    width = 0.8 / len(corners)

    for ci, corner in enumerate(corners):
        means, stds = [], []
        for gname in group_names:
            grp = groups[gname]
            forces = [_get_max_corner_forces(r['result']).get(corner, 0.0) for r in grp]
            means.append(np.mean(forces) if forces else 0.0)
            stds.append(np.std(forces) if len(forces) > 1 else 0.0)
        offset = (ci - len(corners) / 2) * width + width / 2
        ax.bar(x + offset, means, width, label=corner,
               color=colors[ci % len(colors)], yerr=stds, capsize=3)

    ax.set_xlabel('Contact Sequence Group')
    ax.set_ylabel('Max Contact Force [N]')
    ax.set_title('DOE Results: Max Corner Contact Force by Posture Group')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=30, ha='right')
    ax.legend(loc='upper right', ncol=4, fontsize='small')
    fig.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'doe_group_bar.png')
        fig.savefig(out_path, dpi=150)
        print(f"[run_doe_analysis] Saved chart to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
