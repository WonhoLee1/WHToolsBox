import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "TVPackageMotionSim"))

from run_drop_simulator.whts_radioss_builder import RadiossModelBuilder
from run_discrete_builder.whtb_config import get_default_config

cfg     = get_default_config()
out_dir = Path("results/rad_test")
R_mat   = np.eye(3)
t_vec   = np.array([0.0, 0.0, cfg.get('drop_height', 0.5)])

builder = RadiossModelBuilder(
    config=cfg,
    output_dir=out_dir,
    R_mat=R_mat,
    t_vec=t_vec,
    transform_mode='parts',
    drop_height_m=cfg.get('drop_height', 0.5),
    model_name="TVDrop_Test",
)
starter = builder.build()
print(f"\n=== Generated: {starter} ===")
