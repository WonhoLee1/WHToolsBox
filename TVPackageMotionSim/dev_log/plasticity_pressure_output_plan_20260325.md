# Implementation Plan - Plasticity Pressure Output Enhancement
Date: 2026-03-25

## 1. Objective
Enable contact pressure output in terminal logs for both v1 and v2 plasticity algorithms in `run_drop_simulation_v2.py`.

## 2. Changes
### 2.1. Pressure Calculation Utility
- Implement a way to aggregate contact forces and calculate pressure for cushion geoms.
- This will be used in both `apply_plastic_deformation_v1` and `apply_plastic_deformation_v2`.

### 2.2. Update v1 Plasticity (`apply_plastic_deformation_v1`)
- Ensure current pressure is stored and available during the deformation phase.
- Update `log_and_print` for deformation to include the pressure value.

### 2.3. Update v2 Plasticity (`apply_plastic_deformation_v2`)
- Add contact force aggregation to identify pressure during strain-based activation and deformation.
- Update `log_and_print` for both activation and deformation to include the pressure value.

### 2.4. Initial Physics Report
- Add `cush_yield_stress` and `cush_yield_strain` to the "Calculated K & C" section for better visibility of simulation thresholds.

### 2.5. Config Report
- Add description for `cush_yield_stress` in `format_config_report`.

## 3. Verification
- Run simulation with plasticity enabled.
- Verify that terminal output shows pressure (kPa) alongside deformation and strain values.
