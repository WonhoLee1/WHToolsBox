import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize, dual_annealing, differential_evolution
from scipy.interpolate import interp1d
from scipy.stats import qmc
from run_discrete_builder import get_default_config, create_model

# Simulation imports
from run_drop_simulation import run_simulation
from run_stiffness_test import run_stiffness_test, generate_stiffness_test_xml

# Scikit-learn feature flag for Surrogate optimization
try:
    from sklearn.neural_network import MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False

GLOBAL_METRIC_TYPE = "F_TIME" # Default Metric
GLOBAL_INCLUDE_RECOVERY = True # Default Recovery Simulation
GLOBAL_USE_DTW = False # Default DTW Usage
GLOBAL_MSE_WEIGHT = 1000.0 # Default MSE normalization factor (Denominator)
GLOBAL_HISTORY = [] # To store (params, loss) for post-analysis

# [Loss Weighting Configuration]
# 최적화 엔진이 하중의 '크기' vs '형태' vs '부드러움' 중 어디에 더 집중할지 결정하는 가중치들입니다.
# 상황에 따라 아래 수치들을 조절하여 최적화 성향을 바꿀 수 있습니다.
GLOBAL_SHAPE_WEIGHT = 50.0      # R-squared 및 DTW (전체적인 곡선 형태 일치) 중요도
GLOBAL_CURVATURE_WEIGHT = 2.0   # Sharpness (급격하게 꺾이는 곡률 억제) 중요도
GLOBAL_SMOOTHNESS_WEIGHT = 10.0 # Noise (진동 및 톱니바퀴 현상 제거) 중요도

# ==============================================================================
# Helper Functions
# ==============================================================================
def generate_eps_target_curve(
    max_disp=0.8,
    block_size=1.0,
    n_points=100,
    duration=1.0
):
    """
    Generates a synthetic target Force-Displacement and Force-Time curve matching fully non-linear EPS hyperfoam behavior.
    
    Args:
        max_disp (float): Maximum displacement (m) to simulate.
        block_size (float): The side length (m) of the cubic block to calculate area and thickness.
        n_points (int): Number of data points to generate.
        duration (float): The total duration (s) over which the displacement occurs.
        
    Returns:
        tuple (t_arr, d_arr, f_arr): Arrays mapping time to displacement, and displacement to force.
    """
    area = block_size * block_size
    thickness = block_size
    
    d_arr = np.linspace(0, max_disp, n_points)
    t_arr = np.linspace(0, duration, n_points)
    f_arr = np.zeros_like(d_arr)
    
    for i, d in enumerate(d_arr):
        strain = d / thickness
        elastic_stress = 100.0 * (1.0 - np.exp(-strain / 0.015)) 
        plateau_stress = 20.0 * strain
        densification_strain = max(0.0, strain - 0.55)
        densification_stress = 8000.0 * (densification_strain ** 3)
        stress_pa = elastic_stress + plateau_stress + densification_stress
        f_arr[i] = stress_pa * area
        
    return t_arr, d_arr, f_arr

def generate_eps_target_curve_with_recovery(
    max_disp=0.8,
    block_size=1.0,
    n_points=100,
    duration=2.0,
    recovery_disp_zero=0.4,
    recovery_shape=2.0
):
    """
    Generates a synthetic target curve including both loading (compression) and unloading (recovery) phases.
    Uses a smooth decay logic representing low-density foams, where force gently lands to 0 at `recovery_disp_zero`.
    
    Args:
        max_disp (float): Maximum displacement (m) to simulate.
        block_size (float): The side length (m) of the cubic block.
        n_points (int): Number of total data points to generate.
        duration (float): The total duration (s). Peak compression is at duration/2.
        recovery_disp_zero (float): The displacement (m) at which the restorative force becomes completely 0.
        recovery_shape (float): Polynomial power dictating the curvature of the unloading curve (>1 is concave up).
        
    Returns:
        tuple (t_arr, d_arr, f_arr): Arrays mapping time to displacement, and time to force.
    """
    area = block_size * block_size
    thickness = block_size
    
    t_arr = np.linspace(0, duration, n_points)
    d_arr = np.zeros_like(t_arr)
    f_arr = np.zeros_like(t_arr)
    
    half_idx = int(n_points / 2)
    
    # 1. Loading Phase (0 to half_idx)
    for i in range(half_idx):
        d = max_disp * (t_arr[i] / (duration / 2.0))
        d_arr[i] = d
        strain = d / thickness
        elastic_stress = 10.0 * (1.0 - np.exp(-strain / 0.015)) 
        plateau_stress = 20.0 * strain
        densification_strain = max(0.0, strain - 0.55)
        densification_stress = 8000.0 * (densification_strain ** 3)
        stress_pa = elastic_stress + plateau_stress + densification_stress
        f_arr[i] = stress_pa * area
        
    # 2. Unloading Phase (half_idx to end)
    max_d = d_arr[half_idx - 1]
    max_f = f_arr[half_idx - 1]
    
    for i in range(half_idx, n_points):
        d = max_disp * (1.0 - (t_arr[i] - duration/2.0) / (duration / 2.0))
        d_arr[i] = d
        
        if d <= recovery_disp_zero:
            # Completely plasticized/disconnected area
            f_arr[i] = 0.0
        else:
            # Smoothly transition from max_f down to 0 as displacement retreats from max_d down to recovery_disp_zero
            d_norm = (d - recovery_disp_zero) / (max_d - recovery_disp_zero)
            f_arr[i] = max_f * (d_norm ** recovery_shape)
            
    return t_arr, d_arr, f_arr

_opt_iter_cnt = 0
def reset_opt_counter():
    global _opt_iter_cnt
    _opt_iter_cnt = 0

# ==============================================================================
# Objective Function (Used by all optimization methods)
# ==============================================================================
def objective_function(
    x,
    target_time_hist=None,
    target_z_hist=None,
    target_max_def=None,
    config=None,
    xml_path="temp_opt.xml",
    model_type="FULL",
    target_fd=None
):
    """
    Core evaluation function for the optimization process to assess a set of material parameters.
    Runs a MuJoCo simulation using the proposed parameters and computes the error (loss) against targets.
    
    Args:
        x (list|ndarray): Parameter vector being proposed by the optimizer (stiffness, damping, solimp bounds/params).
        target_time_hist (ndarray): Target time history array (used in FULL mode).
        target_z_hist (ndarray): Target Z-displacement history array (used in FULL mode).
        target_max_def (float): Target maximum deformation target (used in FULL mode).
        config (dict): The baseline MuJoCo build configuration dictionary.
        xml_path (str): Temporary file path for the generated XML scene.
        model_type (str): Simulation mode ("FULL" or "UNIT_TEST").
        target_fd (tuple): Target tuples generated for testing (Target_Time, Target_Displacement, Target_Force).
        
    Returns:
        float: Calculated loss value to be minimized by the optimization algorithm.
    """
    global _opt_iter_cnt
    _opt_iter_cnt += 1
    
    # 1. Unpack variables and clamp to safe physics bounds
    sr_stiff = np.clip(x[0], 0.00001, 10.0)   
    sr_damp = np.clip(x[1], 0.0001, 10.0)     
    si_dmin = np.clip(x[2], 0.0001, 0.999)
    si_dmax = np.clip(x[3], 0.0001, 0.9999)  
    
    if si_dmin > si_dmax:
        si_dmin, si_dmax = si_dmax, si_dmin
        
    if (si_dmax - si_dmin) < 0.005:
        si_dmax = min(0.9999, si_dmin + 0.01)
        
    si_width, si_mid, si_pow = 0.001, 0.5, 2.0
    cush_density = 20.0
    if len(x) >= 7:
        si_width = np.clip(x[4], 0.0001, 0.5)
        si_mid = np.clip(x[5], 0.01, 0.99)
        si_pow = np.clip(x[6], 0.1, 10.0)
    if len(x) >= 8:
        cush_density = np.clip(x[7], 1.0, 500.0)
    
    # Update config with current proposal
    cfg = get_default_config(config)
    cfg["cush_solref_timec"] = sr_stiff
    cfg["cush_solref_dampr"] = sr_damp
    cfg["cush_solimp_dmin"] = si_dmin
    cfg["cush_solimp_dmax"] = si_dmax
    if len(x) >= 7:
        cfg["cush_solimp_width"] = si_width
        cfg["cush_solimp_mid"] = si_mid
        cfg["cush_solimp_power"] = si_pow
    if len(x) >= 8:
        cfg["cush_density"] = cush_density
    
    loss = 0.0
    
    # FULL DROP SIMULATION MODE
    if model_type == "FULL":
        xml_str = create_model(xml_path, config=cfg)
        try:
            time_hist, z_hist, vel_hist, acc_hist, corner_acc, max_g, metrics = run_simulation(cfg, sim_duration=cfg.get("sim_duration", 0.5))
        except Exception as e:
            sys.stderr.write(f"\n[Simulation Error] {e}\n")
            return 1e6
            
        if target_time_hist is not None and target_z_hist is not None:
            try:
                sim_z_interp = interp1d(time_hist, z_hist, kind='linear', bounds_error=False, fill_value="extrapolate")
                sim_z_at_target_times = sim_z_interp(target_time_hist)
                mse = np.mean((sim_z_at_target_times - target_z_hist) ** 2)
                loss += mse * 10000.0
            except Exception as e:
                loss += 1e6
                
        if target_max_def is not None:
            sim_max_def = 0.0
            for comp, j_data in metrics.items():
                if comp == "BPackagingBox": continue
                for j, hist in j_data.items():
                    if hist.get('bending'): sim_max_def = max(sim_max_def, max(hist['bending']))
                    if hist.get('twist'): sim_max_def = max(sim_max_def, max(hist['twist']))
            error_def = (sim_max_def - target_max_def) / (target_max_def + 1e-6)
            loss += (error_def ** 2) * 50.0
            
        min_zg = min(z_hist) if z_hist else 0.0
        peak_metric = max_g
        
        # Plot Progress
        if target_time_hist is not None and target_z_hist is not None:
            try:
                plt.figure(figsize=(8, 5))
                plt.plot(target_time_hist, target_z_hist, 'tab:orange', linestyle='--', linewidth=2, label='Target Z')
                if time_hist is not None and z_hist is not None:
                    plt.plot(time_hist, z_hist, 'tab:blue', linewidth=2, label='Current Z')
                plt.title(f'FULL Drop Optimization (Loss: {loss:.4f})')
                plt.xlabel('Time [s]')
                plt.ylabel('Z Displacement [m]')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig('opt_progress_FULL.png', dpi=100)
                plt.close()
            except Exception as e:
                pass
                
    # UNIT TEST MODE
    elif model_type == "UNIT_TEST":
        cfg["unit_size"] = [1.0, 1.0, 1.0]
        cfg["unit_div"] = [3, 3, 3] 
        xml_test = generate_stiffness_test_xml("BUnitBlock", test_type="COMPRESSION", cfg=cfg)
        
        try:
            # target_fd = (target_t, target_d, target_f)
            max_disp = np.max(target_fd[1])
            target_duration = target_fd[0][-1] if len(target_fd) >= 3 else 1.0
            
            t, d, f, k = run_stiffness_test(xml_test, test_type="COMPRESSION", target_disp=-max_disp, duration=target_duration, plot_results=False, include_recovery=GLOBAL_INCLUDE_RECOVERY)
            
            if len(d) < 2 or len(f) < 2 or (max(d) - min(d)) < 1e-6:
                raise ValueError("Simulation unstable or crashed.")
                
            if GLOBAL_METRIC_TYPE == "F_TIME":
                sim_f_interp = interp1d(t, f, kind='linear', bounds_error=False, fill_value="extrapolate")
                sim_f_at_target = sim_f_interp(target_fd[0])
            else:
                sim_f_interp = interp1d(d, f, kind='linear', bounds_error=False, fill_value="extrapolate")
                sim_f_at_target = sim_f_interp(target_fd[1])
            
            # 여기서 MSE Loss는 선택된 기준(Time 또는 Displacement)으로 계산됩니다.
            error = sim_f_at_target - target_fd[2]
            mse = np.mean(error**2)
            if np.isnan(mse) or np.isinf(mse):
                return 1e8
                
            # [DTW (Dynamic Time Warping) Shape Penalty]
            # DTW는 시간축(X축)이나 스케일이 조금 엇나가더라도 파고(Peak)나 오르락내리락하는 
            # '기하학적 기저 형태(Shape)' 자체가 얼마나 똑같이 생겼는지를 고무줄처럼 늘려가며 매칭한 최단 거리를 반환합니다.
            dtw_penalty = 0.0
            if HAS_FASTDTW and GLOBAL_USE_DTW:
                try:
                    # 1D array 형태로 (거리 계산용)
                    sim_f_reshaped = sim_f_at_target.reshape(-1, 1)
                    tgt_f_reshaped = target_fd[2].reshape(-1, 1)
                    # fastdtw 반환값: (거리, 매칭 경로). 거리가 짧을 수록 모양이 유사함.
                    dtw_distance, _ = fastdtw(sim_f_reshaped, tgt_f_reshaped, dist=euclidean)
                    # 거리 스케일을 MSE 스케일에 어느 정도 맞추기 위해 정규화 및 가중치 부여
                    mean_tgt_mag = np.mean(np.abs(target_fd[2])) + 1e-6
                    normalized_dtw = dtw_distance / (len(sim_f_at_target) * mean_tgt_mag)
                    dtw_penalty = normalized_dtw * 50.0 # 하이퍼파라미터 (페널티 위력 곱)
                except Exception as e:
                    pass
            
            # [R-squared Shape Penalty & Target Adherence (보완재)]
            tgt_mean = np.mean(target_fd[2])
            ss_tot = np.sum((target_fd[2] - tgt_mean)**2) + 1e-8
            ss_res = np.sum((target_fd[2] - sim_f_at_target)**2)
            r_squared = 1.0 - (ss_res / ss_tot)
            r2_penalty = max(0.0, 1.0 - r_squared) * GLOBAL_SHAPE_WEIGHT
            
            # MSE 기본 Loss 
            # GLOBAL_MSE_WEIGHT(기본 1000)으로 나누어 다른 Shape Penalty(DTW, R^2)와 체급을 맞춤
            loss += mse / GLOBAL_MSE_WEIGHT
            
            # 각 Penalty 항목 병합
            loss += dtw_penalty
            loss += r2_penalty
            
            # [Sharpness/Curvature Penalty (곡률/뾰족함 페널티)]
            # 시뮬레이션 하중 곡선이 타겟 곡선처럼 부드럽게 변하지 않고, 특정 구간(예: 정점)에서 
            # '뾰족'하게 꺾이는(Sharp corner) 현상을 억제하기 위한 물리적 제약 조건입니다.
            # 2차 미분(Second Derivative)은 '변화율의 변화율'을 의미하며, 이 값이 클수록 곡률이 급격함을 뜻합니다.
            if len(f) > 3:
                # np.diff(f, n=2)를 통해 하중 곡선의 가속도 성분(곡률)을 추출합니다.
                d2f = np.diff(f, n=2)
                # 하중 스케일을 고려하여 2차 미분의 최댓값에 가중치를 부여합니다.
                # 이를 통해 수치적으로는 맞더라도 '물리적으로 부자연스러운 꺾임'이 있는 모델은 후순위로 밀려나게 됩니다.
                sharpness_penalty = np.max(np.abs(d2f)) * GLOBAL_CURVATURE_WEIGHT 
                loss += sharpness_penalty
                
            # [Smoothness/Noise Penalty (진동/노이즈 페널티)]
            # 곡선이 매끄럽지 못하고 톱니바퀴처럼 진동하거나 노이즈가 섞여 발생하는 시뮬레이션 불안정성을 배제합니다.
            df = np.diff(f)
            # 하중의 증감 방향(Sign)이 너무 자주 바뀌면(5회 초과) 불필요한 시뮬레이션 노이즈로 간주합니다.
            sign_changes = np.sum(np.diff(np.sign(df)) != 0)
            if sign_changes > 5:
                # 정상적인 압축 거동은 변곡점이 적어야 하므로, 진동 횟수에 비례하여 강력한 페널티를 누적합니다.
                loss += (sign_changes * GLOBAL_SMOOTHNESS_WEIGHT)
        except Exception as e:
            return 1e9
            
        min_zg = d[-1] if len(d) > 0 else 0.0
        peak_metric = f[-1] if len(f) > 0 else 0.0

        if target_fd is not None and len(target_fd) >= 3:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot 1: Force-Displacement
                ax1.plot(target_fd[1], target_fd[2], 'tab:orange', linestyle='--', linewidth=2, label='Target F-D')
                if len(d) > 0 and len(f) > 0:
                    d_clean, f_clean = np.array(d), np.array(f)
                    valid_idx = np.isfinite(d_clean) & np.isfinite(f_clean)
                    if np.any(valid_idx):
                        ax1.plot(d_clean[valid_idx], f_clean[valid_idx], 'tab:blue', linewidth=2, label='Current F-D')
                ax1.set_title(f'UNIT_TEST Optimization (Loss: {loss:.4f})')
                ax1.set_xlabel('Displacement [m]')
                ax1.set_ylabel('Compressive Force [N]')
                ax1.grid(True)
                ax1.legend()
                
                # Plot 2: Force-Time
                ax2.plot(target_fd[0], target_fd[2], 'tab:orange', linestyle='--', linewidth=2, label='Target F-Time')
                if len(t) > 0 and len(f) > 0:
                    t_clean, f_clean2 = np.array(t), np.array(f)
                    valid_idx2 = np.isfinite(t_clean) & np.isfinite(f_clean2)
                    if np.any(valid_idx2):
                        ax2.plot(t_clean[valid_idx2], f_clean2[valid_idx2], 'tab:green', linewidth=2, label='Current F-Time')
                ax2.set_title('Force vs Time')
                ax2.set_xlabel('Time [s]')
                ax2.set_ylabel('Compressive Force [N]')
                ax2.grid(True)
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig('opt_progress_UNIT_TEST.png', dpi=100)
                plt.close()
            except Exception as e:
                pass

    debug_str = f"[{model_type} Opt: {_opt_iter_cnt}] solref=({sr_stiff:.4f},{sr_damp:.4f}) imp=({si_dmin:.4f},{si_dmax:.4f}"
    if len(x) >= 7:
        debug_str += f",{si_width:.4f},{si_mid:.4f},{si_pow:.4f}"
    if len(x) >= 8:
        debug_str += f", dens:{cush_density:.1f}"
    debug_str += ")"
        
    print(f"{debug_str} -> Peak: {peak_metric:.1f} | Min D: {min_zg:.3f} | Loss: {loss:.4f}")
    
    # Store in history for later bounds analysis
    GLOBAL_HISTORY.append((x.copy(), loss))
    
    return loss


# ==============================================================================
# Traditional Solvers (Nelder-Mead, Dual Annealing)
# ==============================================================================
def run_traditional_tuning(
    target_time_hist=None,
    target_z_hist=None,
    target_max_def=None,
    initial_guess=[0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0, 20.0],
    method='Nelder-Mead',
    model_type="FULL",
    target_fd=None,
    max_iterations=200
):
    """
    Executes traditional optimization methods like Nelder-Mead (local), Dual-Annealing (global), or DE directly on the simulator.
    
    Args:
        target_time_hist (ndarray): Array of simulation target time points.
        target_z_hist (ndarray): Array of simulation target Z-displacements.
        target_max_def (float): Float target value for maximal deformation.
        initial_guess (list): The starting variables to explore from.
        method (str): Name of scipy.optimize method being used ('Nelder-Mead', 'dual_annealing', 'differential_evolution').
        model_type (str): Simulation mode ("FULL" or "UNIT_TEST").
        target_fd (tuple): Input targets for Force-Displacement validation.
        max_iterations (int): Maximum simulation calls allowed.
        
    Returns:
        OptimizeResult: Scipy's optimization result object containing best coordinates and loss.
    """
    cfg = get_default_config({
        "drop_mode": "F", 
        "drop_height": 0.5,
        "sim_duration": 0.45, 
        "plot_results": False 
    })
    
    # Encode and Decode the initial guess to ensure it's snapped to the grid and range
    initial_guess_norm = encode_norm(initial_guess)
    initial_guess_safe = decode_norm(initial_guess_norm)
    
    xml_temp = "opt_temp.xml"
    args = (target_time_hist, target_z_hist, target_max_def, cfg, xml_temp, model_type, target_fd)
    
    if method == 'dual_annealing':
        res = dual_annealing(objective_function, bounds=GLOBAL_BOUNDS, args=args, maxiter=max_iterations, x0=initial_guess_safe)
    elif method == 'differential_evolution':
        res = differential_evolution(objective_function, bounds=GLOBAL_BOUNDS, args=args, maxiter=max_iterations//15) # popsize multiplier adjusting
    else: # Nelder-Mead
        res = minimize(objective_function, initial_guess_safe, args=args, method='Nelder-Mead', bounds=GLOBAL_BOUNDS, options={'maxiter': max_iterations, 'disp': True, 'xatol': 1e-3, 'fatol': 1e-3})
    
    return res

# ==============================================================================
# State-of-the-Art Solvers (CMA-ES, Optuna/TPE Bayesian)
# ==============================================================================
class DummyResult:
    pass

def run_cmaes_tuning(
    target_time_hist=None,
    target_z_hist=None,
    target_max_def=None,
    model_type="FULL",
    target_fd=None,
    max_iterations=100
):
    """
    Executes the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm, highly recommended for rigid physical simulations.
    
    Args:
        target_time_hist (ndarray): Array of simulation target time points.
        target_z_hist (ndarray): Array of simulation target Z-displacements.
        target_max_def (float): Float target value for maximal deformation.
        model_type (str): Simulation mode ("FULL" or "UNIT_TEST").
        target_fd (tuple): Input targets for Force-Displacement correlation.
        max_iterations (int): Maximum simulation constraints for termination.
        
    Returns:
        DummyResult: Standardized format mimicking an OptimizeResult comprising success stat, best coordinate, and final loss.
    """
    if not HAS_CMA:
        print("\n[오류] cma 패키지가 설치되지 않았습니다. 터미널에서 'pip install cma' 를 실행해주세요.")
        return None
    
    cfg = get_default_config({"drop_mode": "F", "drop_height": 0.5, "sim_duration": 0.45, "plot_results": False})
    
    def obj_wrapper(x_norm):
        return objective_function(decode_norm(x_norm), target_time_hist, target_z_hist, target_max_def, cfg, "opt_temp.xml", model_type, target_fd)
    
    reset_opt_counter()
    print(f"\n 🧬 [CMA-ES] 진화 전략 기반 공분산 행렬 업데이트 시작 (최대 {max_iterations}번 시뮬레이션 런 제한)")
    
    x0 = [0.5] * len(GLOBAL_BOUNDS)
    sigma0 = 0.1 # 탐색 반경을 10%로 줄여 극단적으로 흩어지지 않고 스무스하게 수렴하도록 유도
    
    res = cma.fmin(obj_wrapper, x0, sigma0, options={'bounds': [0, 1], 'maxfevals': max_iterations, 'verbose': -9})
    
    ret = DummyResult()
    ret.x = decode_norm(res[0])
    ret.fun = res[1]
    ret.success = True
    return ret

def run_optuna_tuning(
    target_time_hist=None,
    target_z_hist=None,
    target_max_def=None,
    model_type="FULL",
    target_fd=None,
    max_iterations=100
):
    """
    Executes Tree-structured Parzen Estimator (TPE) algorithm using Optuna for smart, sample-efficient Bayesian searching.
    
    Args:
        target_time_hist (ndarray): Target timeline array.
        target_z_hist (ndarray): Target vertical displacement array.
        target_max_def (float): Float representing maximal target deformation.
        model_type (str): Simulation mode ("FULL" or "UNIT_TEST").
        target_fd (tuple): Tuple with target physical properties (T, D, F).
        max_iterations (int): Budget for number of simulation trials.
        
    Returns:
        DummyResult: Standardized result object holding the ultimate configuration parameters and derived cost.
    """
    if not HAS_OPTUNA:
        print("\n[오류] optuna 패키지가 설치되지 않았습니다. 터미널에서 'pip install optuna' 를 실행해주세요.")
        return None
        
    cfg = get_default_config({"drop_mode": "F", "drop_height": 0.5, "sim_duration": 0.45, "plot_results": False})
    
    def obj_wrapper(trial):
        x_norm = [trial.suggest_float(f'x{i}', 0.0, 1.0) for i in range(len(GLOBAL_BOUNDS))]
        return objective_function(decode_norm(x_norm), target_time_hist, target_z_hist, target_max_def, cfg, "opt_temp.xml", model_type, target_fd)
        
    reset_opt_counter()
    print(f"\n 🧠 [Bayesian/Optuna TPE] 확률 추론 기반 스마트 탐색 시작 (총 {max_iterations}회 시뮬레이션 시도)")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(obj_wrapper, n_trials=max_iterations)
    
    ret = DummyResult()
    best_norm = [study.best_params[f'x{i}'] for i in range(len(GLOBAL_BOUNDS))]
    ret.x = decode_norm(best_norm)
    ret.fun = study.best_value
    ret.success = True
    return ret

# ==============================================================================
# DOE + ML Surrogate + Local Optimization (Hybrid)
# ==============================================================================
def run_surrogate_hybrid_tuning(
    target_time_hist=None,
    target_z_hist=None,
    target_max_def=None,
    model_type="FULL",
    target_fd=None,
    n_samples=100,
    local_method='Nelder-Mead',
    max_iterations=200
):
    """
    Two-staged hybrid approach utilizing Space-filling Design of Experiments (LHS), fitting a Neural Network Surrogate Model,
    then randomly sampling the Surrogate model to select the best start points for a polishing run (Nelder/DE/etc.).
    
    Args:
        target_time_hist (ndarray): Timestamps target array.
        target_z_hist (ndarray): Z-displacement target matching timestamps.
        target_max_def (float): Bending/twisting allowable target metric.
        model_type (str): Either "FULL" body or 1x1x1 "UNIT_TEST" mode.
        target_fd (tuple): Formatted test reference target inputs for metrics.
        n_samples (int): Count of iterations dedicated to Neural Network training samples.
        local_method (str): Secondary traditional optimizer used for finishing ('Nelder-Mead', 'differential_evolution', etc.).
        max_iterations (int): Dedicated iterations allotted for the secondary algorithm.
        
    Returns:
        OptimizeResult: Scipy wrapper result with optimal values.
    """
    if not HAS_SKLEARN:
        print("[오류] 대리모델 생성을 위해 scikit-learn 패키지가 필요합니다. (pip install scikit-learn)")
        return None

    cfg = get_default_config({
        "drop_mode": "F", 
        "drop_height": 0.5,
        "sim_duration": 0.45, 
        "plot_results": False 
    })

    def obj_wrapper(x_norm, t_time, t_z, t_max_def, cfg_inner, x_path, m_type, t_fd):
        return objective_function(decode_norm(x_norm), target_time_hist=t_time, target_z_hist=t_z, target_max_def=t_max_def, config=cfg_inner, xml_path=x_path, model_type=m_type, target_fd=t_fd)

    # 1. DOE Sampling
    print(f"\n ⚙️ [1/4] LHS DOE 샘플 수집 중 ({n_samples}회)...")
    sampler = qmc.LatinHypercube(d=len(GLOBAL_BOUNDS))
    X_train_norm = sampler.random(n=n_samples)
    
    y_train = []
    reset_opt_counter()
    for i, x_n in enumerate(X_train_norm):
        loss = obj_wrapper(x_n, target_time_hist, target_z_hist, target_max_def, cfg, "opt_temp_doe.xml", model_type, target_fd)
        y_train.append(loss)
    
    # 2. Train Surrogate Model
    print(f"\n 🧠 [2/4] 신경망(MLP) 대리모델 학습 중...")
    surrogate_model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=42)
    safe_y_train = np.log(np.clip(y_train, 1e-6, 1e9))
    surrogate_model.fit(X_train_norm, safe_y_train)
    
    # 3. Global Random Search on Surrogate
    print(f"\n 🚀 [3/4] 대리 모델 상에서 100,000개 탐색 및 최적 초기값 스크리닝 중...")
    X_test_norm = sampler.random(n=100000)
    predicted_log_loss = surrogate_model.predict(X_test_norm)
    best_idx = np.argmin(predicted_log_loss)
    best_initial_norm = X_test_norm[best_idx]
    
    print("\n 🎯 [대리모델 최고 초기값]")
    print(f"   파라미터: {np.round(decode_norm(best_initial_norm), 4)}")
    
    # 4. Local Polishing
    print(f"\n 🛠️ [4/4] 국소 최적화 탑재 (알고리즘: {local_method}, 노이즈 방어 특화)")
    bnds_norm = [(0.0, 1.0) for _ in range(len(GLOBAL_BOUNDS))]
    args = (target_time_hist, target_z_hist, target_max_def, cfg, "opt_temp.xml", model_type, target_fd)
    
    reset_opt_counter()
    if local_method == 'dual_annealing':
        res = dual_annealing(
            obj_wrapper,
            bounds=bnds_norm,
            args=args,
            maxiter=max_iterations,
            x0=best_initial_norm
        )
    elif local_method == 'differential_evolution':
        # DE는 population 기반이라 x0(초기값) 주입을 'init' 인자로 전달할 수 있으나 스케일이 다를 수 있으므로
        res = differential_evolution(
            obj_wrapper,
            bounds=bnds_norm,
            args=args,
            maxiter=max_iterations//15,
            x0=best_initial_norm
        )
    else:
        res = minimize(
            obj_wrapper, 
            best_initial_norm, 
            args=args,
            method=local_method,
            bounds=bnds_norm,
            options={'maxiter': max_iterations, 'disp': True}
        )
    
    # Restore real values
    res.x = decode_norm(res.x)
    return res

# ==============================================================================
# Global Design Variables Configuration
# ==============================================================================
GLOBAL_BOUNDS = [
    [ 0.001,  5.0],    # x[0]: cush_solref_timec (timeconst) - 강성 (작을수록 딱딱함)
    [ 0.001,  5.0],    # x[1]: cush_solref_dampr (dampratio) - 감쇠 (1.0 근처가 안정적)
    [ 0.001,  0.3],    # x[2]: cush_solimp_dmin - 최소 임피던스
    [   0.2,  0.9],    # x[3]: cush_solimp_dmax - 최대 임피던스
    [ 0.001,  0.5],    # x[4]: cush_solimp_width - 전이 구간 폭
    [ 0.001,  0.9],    # x[5]: cush_solimp_mid - 전이 중심점
    [  0.01, 10.0],    # x[6]: cush_solimp_power - 거듭제곱 지수
    [   5.0, 40.0]     # x[7]: cush_density (kg/m^3) - 밀도 (EPS 포장재 표준 범위)
]

# Log 스케일 적용 여부 (범위가 여러 자릿수(Scale)를 가로지를 때 탐색 효율을 극대화합니다)
GLOBAL_IS_LOG = [True, True, True, False, True, False, False, False]

def decode_norm(x_norm):
    """
    Decodes randomly normalized [0~1] ranges used by optimizers to actual physics bounds mapped in GLOBAL_BOUNDS,
    managing linear/log scaling logic automatically alongside safety constrains ensuring parameter validities.
    
    Args:
        x_norm (ndarray/list): Evaluated array bounded [0.0, 1.0] from optimizers.
        
    Returns:
        ndarray: Scaled variables directly applicable inside objective function parameters.
    """
    x_real = np.zeros_like(x_norm)
    for i in range(len(GLOBAL_BOUNDS)):
        low, high = GLOBAL_BOUNDS[i]
        x_n = np.clip(x_norm[i], 0.0, 1.0)
        if GLOBAL_IS_LOG[i]: x_real[i] = 10 ** (np.log10(low) + x_n * (np.log10(high) - np.log10(low)))
        else: x_real[i] = low + x_n * (high - low)
    if x_real[2] >= x_real[3]: x_real[2], x_real[3] = x_real[3], x_real[2]
    if (x_real[3] - x_real[2]) < 0.005: x_real[3] = min(GLOBAL_BOUNDS[3][1], x_real[2] + 0.01)
    return x_real

def encode_norm(x_real):
    """
    Encodes actual physics variables into normalized [0.0, 1.0] range based on GLOBAL_BOUNDS.
    Opposite of decode_norm.
    """
    x_norm = np.zeros_like(x_real)
    for i in range(len(GLOBAL_BOUNDS)):
        low, high = GLOBAL_BOUNDS[i]
        val = np.clip(x_real[i], low, high)
        if GLOBAL_IS_LOG[i]:
            x_norm[i] = (np.log10(val) - np.log10(low)) / (np.log10(high) - np.log10(low))
        else:
            x_norm[i] = (val - low) / (high - low)
    return np.clip(x_norm, 0.0, 1.0)

# ==============================================================================
# MAIN CLI
# ==============================================================================
if __name__ == "__main__":
    print("="*75)
    print(" 🎯 Choose Simulation Target Mode:")
    print("   [1] FULL (Current Built-In Cushion Drop Simulation)")
    print("   [2] UNIT_TEST (1x1x1 Unit Block Compression matching EPS Target Curve)")
    print("="*75)
    mode_choice = input("Enter choice (1 or 2, default 2): ").strip()
    if not mode_choice: mode_choice = "2"
    
    print("\n" + "="*75)
    print(" ⚙️  Choose Optimization Algorithm (🔥 Recommended Priority Rank):")
    print("   [1] 🥇 CMA-ES (추천 1순위: 물리/로봇 최적화의 제왕. 평가가 거칠어도 완벽 적응)")
    print("   [2] 🥈 Optuna / Bayesian TPE (추천 2순위: 가장 적은 횟수로 현명하게 찔러봄)")
    print("   [3] 🥉 Surrogate (MLP) + Nelder-Mead (노이즈 방어 특화 하이브리드)")
    print("   [4] Surrogate (MLP) + Differential Evolution (전역 탐색 하이브리드)")
    print("   [5] Differential Evolution 단독 (군집 돌연변이 전역 탐색)")
    print("   [6] Surrogate + Dual Annealing (대리모델 점프 후 전역 탐색)")
    print("   [7] Nelder-Mead 단독 (가장 고전적인 로컬 탐색)")
    print("   [8] Dual Annealing 단독 (무작위 전역 탐색)")
    print("="*75)
    algo_choice = input("Enter choice (1~8, default 1): ").strip()
    if not algo_choice: algo_choice = "1"
    
    print("\n" + "="*75)
    print(" 🔁 Enter Maximum Iterations:")
    print("   (이 값이 높을 수록 더 오래 탐색하며 최적점을 정밀하게 찾습니다. 기본값: 100)")
    print("="*75)
    max_iter_input = input("Max Iterations (default 100): ").strip()
    try:
        max_iters = int(max_iter_input) if max_iter_input else 100
    except ValueError:
        max_iters = 100
    
    print("\n" + "="*75)
    print(" 📏 Enter Number of Comparison Points (데이터 추출 및 비교 평가 점 개수):")
    print("   (이 정밀도가 높을 수록 촘촘하게 비교하지만 보간이 미세하게 느려집니다. 기본값: 100)")
    print("="*75)
    n_pts_input = input("n_points (default 100): ").strip()
    try:
        n_points = int(n_pts_input) if n_pts_input else 100
    except ValueError:
        n_points = 100
        
    print("\n" + "="*75)
    print(" 🎯 Choose Loss Comparison Metric:")
    print("   [1] F-Time (Force-Time, Recommended for Recovery/Plasticity)")
    print("   [2] F-D (Force-Displacement, Classic)")
    print("="*75)
    metric_choice = input("Enter choice (1 or 2, default 1): ").strip()
    GLOBAL_METRIC_TYPE = "F_D" if metric_choice == "2" else "F_TIME"
    
    print("\n" + "="*75)
    print(" 🔄 Include Recovery (Plasticity) Phase in UNIT_TEST?")
    print("   [Y] Yes, Simulate compression and recovery (requires F-Time metric)")
    print("   [N] No, compression only")
    print("="*75)
    recovery_choice = input("Enter choice (Y or N, default Y): ").strip().upper()
    GLOBAL_INCLUDE_RECOVERY = False if recovery_choice == "N" else True
    
    print("\n" + "="*75)
    print(" 📏 Evaluate Shape Similarity using DTW (Dynamic Time Warping)?")
    print("   [Y] Yes, DTW penalty 적용 (형상의 기하학적 유사도/시프트 중점 평가, 느림)")
    print("   [N] No, 기존 MSE + R^2(분산 예측) 만으로 평가 (빠름)")
    print("="*75)
    dtw_choice = input("Enter choice (Y or N, default Y): ").strip().upper()
    GLOBAL_USE_DTW = False if dtw_choice == "N" else True

    print("\n" + "="*75)
    print(" ⚖️  Set MSE Loss Weight (Magnitude Importance):")
    print("   [1] High Sensitivity (Div by 100) - 하중 수치 일치에 매우 강력하게 집착")
    print("   [2] Balanced (Div by 1000) - 수치와 형태(Shape)를 균형있게 고려 (추천)")
    print("   [3] Low Sensitivity (Div by 10000) - 수치보다는 형태의 흐름에 더 집중")
    print("="*75)
    weight_choice = input("Enter choice (1, 2, 3, default 2): ").strip()
    if weight_choice == "1": GLOBAL_MSE_WEIGHT = 100.0
    elif weight_choice == "3": GLOBAL_MSE_WEIGHT = 10000.0
    else: GLOBAL_MSE_WEIGHT = 1000.0
    
    model_type = "UNIT_TEST" if mode_choice == "2" else "FULL"
    target_fd, target_time_hist, target_z_hist, target_max_def = None, None, None, None
    
    # 1. Setup Target Data
    if model_type == "UNIT_TEST":
        print(f"\n▶ Selected UNIT_TEST Mode. Generating EPS target curve... (points: {n_points})")
        block_thickness = 1.0 # 1x1x1 단위 블록
        
        if GLOBAL_INCLUDE_RECOVERY:
            # 타겟 리커버리 곡선의 초기 기울기가 로딩(압축) 곡선의 마지막 기울기보다 커야 곡선이 겹치지 않습니다.
            # recovery_shape을 2.0에서 4.0으로 높여 더 급격한 초기 하중 감소를 유도합니다.
            t_t, t_d, t_f = generate_eps_target_curve_with_recovery(max_disp=0.8, block_size=block_thickness, n_points=n_points, duration=2.0, recovery_disp_zero=0.4, recovery_shape=4.0)
        else:
            t_t, t_d, t_f = generate_eps_target_curve(max_disp=0.8, block_size=block_thickness, n_points=n_points, duration=1.0)
            
        target_fd = (t_t, t_d, t_f)
        
        # width의 최대값을 시료 두께의 90%로 제한
        GLOBAL_BOUNDS[4][1] = block_thickness * 0.90
        print(f"   [Constraint] 'cush_solimp_width' 상한이 블록 두께의 90% ({GLOBAL_BOUNDS[4][1]}m) 로 자동 설정되었습니다.")
        
    else:
        print(f"\n▶ Selected FULL Drop Simulation Mode. Generating dummy physical test target... (points: {n_points})")
        dummy_time = np.linspace(0, 0.45, n_points)
        dummy_z = np.clip(0.5 - 0.5 * 9.8 * dummy_time**2, 0.1, 0.5) 
        bounce_idx = dummy_time > 0.28
        dummy_z[bounce_idx] = 0.1 + (dummy_time[bounce_idx] - 0.28) * 0.5
        target_time_hist, target_z_hist = dummy_time, dummy_z
        target_max_def = 15.0
        
        # FULL Mode 역시 필요하다면 패키징 폼 두께에 맞게 상한치 조정 가능 (여기서는 임의로 가정)
        cushion_thickness = 0.1 # 예: 10cm 두께 쿠션
        GLOBAL_BOUNDS[4][1] = cushion_thickness * 0.90
        print(f"   [Constraint] 'cush_solimp_width' 상한이 예상 쿠션 두께의 90% ({GLOBAL_BOUNDS[4][1]}m) 로 자동 설정되었습니다.")

    # 2. Run Optimization
    if algo_choice == "1":
        res = run_cmaes_tuning(
            target_time_hist=target_time_hist, target_z_hist=target_z_hist, target_max_def=target_max_def,
            model_type=model_type, target_fd=target_fd, max_iterations=max_iters
        )
    elif algo_choice == "2":
        res = run_optuna_tuning(
            target_time_hist=target_time_hist, target_z_hist=target_z_hist, target_max_def=target_max_def,
            model_type=model_type, target_fd=target_fd, max_iterations=max_iters
        )
    elif algo_choice == "3":
        res = run_surrogate_hybrid_tuning(
            target_time_hist=target_time_hist, target_z_hist=target_z_hist, target_max_def=target_max_def,
            model_type=model_type, target_fd=target_fd,
            n_samples=max_iters//2, local_method='Nelder-Mead', max_iterations=max_iters//2 # split budget
        )
    elif algo_choice == "4":
        res = run_surrogate_hybrid_tuning(
            target_time_hist=target_time_hist, target_z_hist=target_z_hist, target_max_def=target_max_def,
            model_type=model_type, target_fd=target_fd,
            n_samples=max_iters//2, local_method='differential_evolution', max_iterations=max_iters//2
        )
    elif algo_choice == "5":
        res = run_traditional_tuning(
            target_time_hist=target_time_hist, target_z_hist=target_z_hist, target_max_def=target_max_def,
            method='differential_evolution', model_type=model_type, target_fd=target_fd, max_iterations=max_iters
        )
    elif algo_choice == "6":
        res = run_surrogate_hybrid_tuning(
            target_time_hist=target_time_hist, target_z_hist=target_z_hist, target_max_def=target_max_def,
            model_type=model_type, target_fd=target_fd,
            n_samples=max_iters//2, local_method='dual_annealing', max_iterations=max_iters//2
        )
    elif algo_choice == "7":
        res = run_traditional_tuning(
            target_time_hist=target_time_hist, target_z_hist=target_z_hist, target_max_def=target_max_def,
            method='Nelder-Mead', model_type=model_type, target_fd=target_fd, max_iterations=max_iters
        )
    else:
        res = run_traditional_tuning(
            target_time_hist=target_time_hist, target_z_hist=target_z_hist, target_max_def=target_max_def,
            method='dual_annealing', model_type=model_type, target_fd=target_fd, max_iterations=max_iters
        )

    # 3. Print Results
    if res is not None:
        print("\n" + "="*75)
        print(" ✅ Optimization Complete")
        print(f" Success: {res.success}")
        print(f" Final Loss: {res.fun:.4f}")
        
        best = res.x
        print("\n[Optimal Cushion Parameters Found]")
        print(f"  'cush_solref_timec': {best[0]:.5f}")
        print(f"  'cush_solref_dampr': {best[1]:.5f}")
        print(f"  'cush_solimp_dmin': {best[2]:.5f}")
        print(f"  'cush_solimp_dmax': {best[3]:.5f}")
        print(f"  'cush_solimp_width': {best[4]:.5f}")
        print(f"  'cush_solimp_mid': {best[5]:.5f}")
        print(f"  'cush_solimp_power': {best[6]:.5f}")
        if len(best) >= 8:
            print(f"  'cush_density': {best[7]:.2f} kg/m^3")
        print("="*75)
        
        print("\n 📊 Generating final optimization plot with the best parameters...")
        # 전역/랜덤 탐색의 마지막 평가가 최고 평가가 아니므로(마지막 탐색은 랜덤한 점일 수 있음), 
        # 찾아낸 '최종 최적값'을 기준으로 그래프를 새로 찍어 진짜 결과를 opt_progress_*.png 로 덮어씁니다.
        cfg_final = get_default_config({
            "drop_mode": "F", 
            "drop_height": 0.5,
            "sim_duration": 0.45, 
            "plot_results": False   # 시뮬레이션 직후 자동 분석 그래프(PNG)를 생성, 최적화 시에는 False로 설정
        })
        objective_function(best, target_time_hist, target_z_hist, target_max_def, cfg_final, "opt_temp.xml", model_type, target_fd)
        if model_type == "UNIT_TEST":
            print("   ✅ [Final Plot Saved] Please check 'opt_progress_FULL.png'")
            
        # --- [NEW] Search Space Optimization Suggestion ---
        if len(GLOBAL_HISTORY) > 10:
            print("\n" + "🔍" + " "*2 + "[Search Space Analysis & Suggestion]")
            print("-" * 75)
            # Sort by loss and pick top 10%
            sorted_history = sorted(GLOBAL_HISTORY, key=lambda x: x[1])
            top_n = max(5, len(sorted_history) // 10)
            top_performers = np.array([h[0] for h in sorted_history[:top_n]])
            
            print(f"  Analyzed top {top_n} performers out of {len(sorted_history)} iterations.")
            print("  Suggested New GLOBAL_BOUNDS (Copy-paste to replace current settings):")
            print("\nGLOBAL_BOUNDS = [")
            
            param_names = [
                "sr_stiff", "sr_damp", "si_dmin", "si_dmax", 
                "si_width", "si_mid", "si_pow", "density "
            ]
            
            for i in range(len(GLOBAL_BOUNDS)):
                p_min = np.min(top_performers[:, i])
                p_max = np.max(top_performers[:, i])
                # Add 20% margin
                margin = (p_max - p_min) * 0.2 + 1e-6
                suggested_low = max(GLOBAL_BOUNDS[i][0], p_min - margin)
                suggested_high = min(GLOBAL_BOUNDS[i][1], p_max + margin)
                
                comma = "," if i < len(GLOBAL_BOUNDS)-1 else ""
                print(f"    [{suggested_low:10.5f}, {suggested_high:10.5f}]{comma}  # x[{i}]: {param_names[i]}")
            
            print("]")
            print("-" * 75)
            
        # Cleanup
        if os.path.exists("opt_temp.xml"): os.remove("opt_temp.xml")
        if os.path.exists("opt_temp_doe.xml"): os.remove("opt_temp_doe.xml")
