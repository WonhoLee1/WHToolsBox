import numpy as np
import os
import sys
import time
import logging

# Ensure the CADSimplifier class is available. 
# Assuming cad_simplifier.py is in the same directory or python path.
try:
    from cad_simplifier import CADSimplifier
except ImportError:
    # If running from the same directory where cad_simplifier.py resides
    sys.path.append(os.getcwd())
    from cad_simplifier import CADSimplifier

def setup_logging():
    """Sets up logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("cad_simplifier_headless.log", mode='w')
        ]
    )
    return logging.getLogger(__name__)

def run_headless_example():
    """
    Demonstrates the full capabilities of CADSimplifier in a non-UI (headless) environment.
    Includes:
    1. Initialization & Configuration
    2. Loading Data (Sample or File)
    3. Running Simplification (with Auto-tuning option)
    4. Refinement
    5. Result Evaluation
    6. Exporting (CAD, Info, BBox, etc.)
    """
    logger = setup_logging()
    logger.info("=== CAD Simplifier Headless Example Started ===")

    # 1. Initialization
    simplifier = CADSimplifier()
    
    # Configuration (mimicking 'Medium' Preset)
    # Note: These can be adjusted based on needs or loaded from a config file.
    config = {
        'voxel_resolution': 1.0,      # Will be auto-set based on model size later, but good to have default
        'min_volume_ratio': 0.002,
        'max_cutters': 100,
        'tolerance': 0.02,
        'detect_slanted': True,
        'slanted_area_factor': 4.0,
        'slanted_edge_factor': 2.0,
        'min_cutter_size': 3.0,
        'auto_tune': False,           # Set to True to enable auto-resolution scaling
        'target_error': 5.0,
        'max_iterations': 3
    }
    
    logger.info("Configuration set:")
    for k, v in config.items():
        logger.info(f"  - {k}: {v}")

    # 2. Load Model (Using a built-in sample for this example)
    sample_name = 'pipe_connector' # Options: 'basic', 'l_bracket', 'pipe_connector', etc.
    logger.info(f"Loading sample model: {sample_name}...")
    
    if not simplifier.create_sample_shape(sample_name):
        logger.error("Failed to generate sample shape.")
        return

    # Check Model Stats
    min_pt, max_pt = simplifier.bounding_box
    diagonal = np.linalg.norm(max_pt - min_pt)
    logger.info(f"Model Loaded. Bounding Box Diagonal: {diagonal:.2f}mm")
    logger.info(f"  - Min: {min_pt}, Max: {max_pt}")

    # Dynamically adjust parameters based on model size (like the GUI 'Apply Preset' logic)
    # Example: Applying 'Medium' logic manually
    res = diagonal / 100.0
    res = max(0.2, round(res, 2))
    config['voxel_resolution'] = res
    config['min_cutter_size'] = max(1.0, round(res * 3.0, 1)) # approx factor 3.0 for medium
    
    logger.info(f"Auto-adjusted parameters based on scale:")
    logger.info(f"  - Voxel Resolution: {config['voxel_resolution']}mm")
    logger.info(f"  - Min Cutter Size: {config['min_cutter_size']}mm")

    # 3. Processing Loop (Simplification)
    logger.info("Starting Simplification Process...")
    start_time = time.time()

    # Progress callback for logging
    def log_progress(msg):
        logger.info(f"  [Progress] {msg}")

    # Main Cutter Generation Step
    # If auto_tune is True, you would wrap this in a loop similar to the GUI's worker method.
    # Here we demonstrate a single robust pass.
    
    current_max_cutters = config['max_cutters'] # Could increase this in a loop if needed
    
    simplifier.generate_cutters(
        voxel_resolution=config['voxel_resolution'],
        min_volume_ratio=config['min_volume_ratio'],
        max_cutters=current_max_cutters,
        tolerance=config['tolerance'],
        detect_slanted=config['detect_slanted'],
        masks=[], # No exclusion zones
        slanted_area_factor=config['slanted_area_factor'],
        slanted_edge_factor=config['slanted_edge_factor'],
        min_cutter_size=config['min_cutter_size'],
        append=False, # Start fresh
        progress_callback=log_progress
    )

    # 4. Refine Step (Optional but recommended)
    # In the GUI, this is a separate button, but in a pipeline, we might want to do it immediately
    # or just let the generate_cutters' internal refinement handle it. 
    # generate_cutters already calls _refine_cutter internally for each block.
    # If we wanted to "Refine" specifically (re-run refinement on existing cutters with stricter settings):
    logger.info("Refining existing cutters (Pruning inefficient ones)...")
    simplifier.prune_inefficient_cutters(min_volume_ratio=config['min_volume_ratio'])


    # 5. Generate CAD Geometry
    logger.info("Generating Final CAD Geometry (Solid Body)...")
    # You can force 'gmsh' or 'cadquery'. 'gmsh' is preferred for stability in this tool.
    result_shape, cutters_shape = simplifier.generate_cad(use_engine='gmsh')
    
    duration = time.time() - start_time
    logger.info(f"Processing finished in {duration:.2f} seconds.")

    if result_shape is None:
        logger.error("Failed to generate CAD geometry.")
        return

    # 6. Evaluation
    logger.info("Evaluating Accuracy...")
    vol_error = simplifier.calculate_volume_error(result_shape)
    if vol_error is not None:
        logger.info(f"  - Volume Error: {vol_error:.2f}%")
    else:
        logger.warning("  - Could not calculate volume error.")

    # Accessing Cutter Data
    logger.info(f"Total Cutters Generated: {len(simplifier.cutters)}")
    # Example: Inspect first 3 cutters
    for i, c in enumerate(simplifier.cutters[:3]):
        c_type = c.get('type', 'aabb')
        logger.info(f"  - Cutter {i+1}: Type={c_type}")


    # 7. Exports
    output_dir = "headless_output"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Exporting results to '{output_dir}/'...")

    # A. Simplified Model (STEP)
    step_path = os.path.join(output_dir, f"{sample_name}_simplified.step")
    simplifier.export_step(result_shape, step_path)
    logger.info(f"  - Saved Simplified Model: {step_path}")

    # B. Base + Cutters (Assembly)
    assembly_path = os.path.join(output_dir, f"{sample_name}_assembly.step")
    simplifier.export_step("export_base_and_cutters", assembly_path)
    logger.info(f"  - Saved Assembly: {assembly_path}")

    # C. Cutters Only
    cutters_path = os.path.join(output_dir, f"{sample_name}_cutters.step")
    simplifier.export_step("export_cutters_only", cutters_path)
    logger.info(f"  - Saved Cutters Only: {cutters_path}")

    # D. Cutter Info Text
    txt_path = os.path.join(output_dir, f"{sample_name}_info.txt")
    simplifier.export_cutter_info(txt_path)
    logger.info(f"  - Saved Info Text: {txt_path}")

    logger.info("=== Headless Example Completed Successfully ===")

if __name__ == "__main__":
    run_headless_example()
