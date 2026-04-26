# TV Package Drop Simulation — Trajectory-Based Parameter Optimization Usage

## File Structure

| File | Role |
|------|------|
| `run_drop_simulation_case_opt_traject.py` | CMA-ES Optimization Execution (Entry Point) |
| `monitor_opt_traject.py` | Real-time Optimization Monitor (PySide6 UI) |
| `resources/profiles.txt` | Measured Drop Trajectory Data |

---

## Quick Start

```bash
python TVPackageMotionSim/run_drop_simulation_case_opt_traject.py
```

If `LAUNCH_MONITOR = True` (default), the monitor window will open automatically when the optimization starts.

---

## Key Settings (Bottom of Entry Point)

```python
SELECTED_CORNERS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
SIGMA0           = 0.25     # Initial search radius (normalized space, smaller = narrower)
POPSIZE          = 10       # Samples per generation (larger = wider search, slower)
MAX_EVALS        = 500      # Maximum number of simulation runs
W_DISP           = 0.50     # Displacement DTW weight
W_VEL            = 0.50     # Velocity DTW weight
N_WORKERS        = 4        # Number of parallel processes (adjust to CPU cores)
SUPPRESS_OUTPUT  = True     # False: Show simulator output in terminal
LAUNCH_MONITOR   = True     # False: Manually launch monitor
```

### Corner-specific Response Weights

```python
RESPONSE_WEIGHTS = {c: {'disp': 1.0, 'vel': 1.0} for c in SELECTED_CORNERS}

# Example: Exclude specific responses (set to 0 to exclude from cost function)
RESPONSE_WEIGHTS['C3']['vel'] = 0.0   # Exclude C3 velocity
RESPONSE_WEIGHTS['C7']['disp'] = 0.0  # Exclude C7 displacement
```

---

## Manual Monitor Execution

```bash
# Method A: Pass the result folder path as an argument
python TVPackageMotionSim/monitor_opt_traject.py opt_traject_results/20260426_223402

# Method B: Run without arguments → Use "Browse..." button in UI
python TVPackageMotionSim/monitor_opt_traject.py
```

> **Tip**: You can use the same UI to check final results by pointing to the result folder even after optimization is finished.

---

## Monitor Tab Descriptions

| Tab | Content |
|----|------|
| **Convergence** | Generation-wise cost scatter + running Best curve, σ(sigma) trend |
| **Obj Evolution** | Evolution of objective function components (f_disp, f_vel) |
| **Param Status** | Initial vs. Best parameter comparison (normalized horizontal bar chart) |
| **Param Evolution** | Parameter trend over generations (small scatter plot array) |
| **Response Dist.** | Corner-wise displacement (blue)/velocity (orange) DTW cost bars — Inactive in gray |
| **Images** | PNG files in result folder (overlay, drop_dir, convergence, etc.) |

The monitor **auto-refreshes every 5 seconds**.

---

## Result Folder Structure

```
opt_traject_results/
└── 20260426_223402/          ← Automatically created with execution timestamp
    ├── opt_meta.json          ← Optimization settings (params, corners, weights, etc.)
    ├── evallog.csv            ← Detailed log per evaluation (cost, params, corner DTW)
    ├── best_params.pkl        ← Optimal parameters (pickle)
    ├── convergence.png        ← Convergence graph image
    ├── overlay_gen0010.png    ← Trajectory comparison image (saved every 10 generations)
    ├── overlay_final.png      ← Final trajectory comparison image
    ├── drop_dir_final.png     | Drop direction (Z-axis) disp/vel comparison image
    └── sim_temp/
        ├── e00001/            ← Temporary simulation folder per evaluation
        ├── e00002/
        │   └── error.log      ← Traceback recording on simulation failure
        └── ...
```

---

## evallog.csv Column Structure

```
gen, eval, cost, f_disp, f_vel, sigma, elapsed_s, diverged,
C1_disp, C1_vel, C2_disp, C2_vel, ..., C8_disp, C8_vel,
friction_gnd_cush, friction_gnd_paper, cush_solref_0, ...
```

- `diverged = True` : Simulation failed (Cost treated as 1e6)
- `C{N}_disp` / `C{N}_vel` : Raw DTW cost per corner for disp/vel (before weights)

---

## Optimization Parameter Definitions (13)

| Parameter | Initial | Range | Description |
|----------|--------|------|------|
| `friction_gnd_cush` | 0.70 | 0.10–2.00 | Ground-Cushion friction coefficient |
| `friction_gnd_paper` | 0.70 | 0.10–2.00 | Ground-Paperbox friction coefficient |
| `cush_solref_0` | 0.001 | 1e-4–5e-2 | Cushion contact timeconstant [s] |
| `cush_solref_1` | 1.0 | 0.1–2.0 | Cushion contact damping ratio |
| `cush_solimp_0` | 0.10 | 0.01–0.50 | Cushion impedance d_min |
| `cush_solimp_1` | 0.95 | 0.70–0.999 | Cushion impedance d_max |
| `cush_solimp_2` | 0.02 | 0.001–0.10 | Cushion impedance transition_width |
| `cog_x` | 0.00 | ±0.05 m | COG X offset |
| `cog_y` | 0.00 | ±0.05 m | COG Y offset |
| `cog_z` | 0.00 | ±0.05 m | COG Z offset |
| `plasticity_ratio` | 0.30 | 0.01–0.80 | Plasticity ratio |
| `cush_yield_pressure` | 1500 | 100–15000 Pa | Cushion yield pressure |
| `plastic_hardening_modulus` | 30000 | 100–300000 Pa | Plastic hardening modulus |

---

## Troubleshooting

| Symptom | Cause / Solution |
|------|------------|
| Early termination after 30 evals | `tolflatfitness` issue → already set to `max_evals` in code |
| Successive failures after `e00005` | MuJoCo `set_mjcb_control` stale callback → fixed with `None` reset in `setup()` / `_wrap_up()` |
| Check sim failure reason | Traceback recorded in `sim_temp/e{N}/error.log` |
| Monitor tabs are empty | Normal if opened before first generation finishes — will refresh in 5s |
