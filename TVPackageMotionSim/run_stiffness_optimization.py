import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, dual_annealing, differential_evolution
from scipy.interpolate import interp1d

from run_discrete_builder import get_default_config
from run_stiffness_test import generate_stiffness_test_xml, run_stiffness_test

# ==============================================================================
# Optional API Dependencies
# ==============================================================================
try:
    import cma
    HAS_CMAES = True
except ImportError:
    HAS_CMAES = False

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

# ==============================================================================
# Global Settings & Loss Weights
# ==============================================================================
GLOBAL_USE_DTW = False
GLOBAL_MSE_WEIGHT = 1000.0

# [Loss Weighting Configuration]
GLOBAL_SHAPE_WEIGHT = 50.0      # R-squared 및 DTW 중요도
GLOBAL_CURVATURE_WEIGHT = 2.0   # Sharpness 억제
GLOBAL_SMOOTHNESS_WEIGHT = 10.0 # Noise 진동 억제
GLOBAL_HISTORY = []             # Bounds 분석용 이력 기록

# ==============================================================================
# Global Design Variables Configuration (Soft Weld)
# ==============================================================================
GLOBAL_BOUNDS = [
    [0.001, 5.0],    # x[0]: solref_timec (timeconst) - 강성
    [0.01, 5.0],     # x[1]: solref_dampr (dampratio) - 감쇠
    [0.01, 0.95],    # x[2]: solimp_dmin - 최소 임피던스
    [0.5, 0.999],    # x[3]: solimp_dmax - 최대 임피던스
    [0.0001, 0.5],   # x[4]: solimp_width - 전이 구간 폭
    [0.01, 0.99],    # x[5]: solimp_mid - 전이 중심점
    [0.1, 10.0]      # x[6]: solimp_power - 거듭제곱 지수
]

GLOBAL_IS_LOG = [True, True, False, False, True, False, False]

def decode_norm(x_norm):
    x_real = np.zeros_like(x_norm)
    for i in range(len(x_norm)):
        low, high = GLOBAL_BOUNDS[i]
        x_n = np.clip(x_norm[i], 0.0, 1.0)
        if GLOBAL_IS_LOG[i]: 
            x_real[i] = 10 ** (np.log10(low) + x_n * (np.log10(high) - np.log10(low)))
        else: 
            x_real[i] = low + x_n * (high - low)
            
    # 안전 제약 조건 부여
    if len(x_real) > 3:
        if x_real[2] >= x_real[3]: 
            x_real[2], x_real[3] = x_real[3], x_real[2]
        if (x_real[3] - x_real[2]) < 0.005: 
            x_real[3] = min(GLOBAL_BOUNDS[3][1], x_real[2] + 0.01)
    return x_real

def encode_norm(x_real):
    x_norm = np.zeros_like(x_real)
    for i in range(len(x_real)):
        low, high = GLOBAL_BOUNDS[i]
        val = np.clip(x_real[i], low, high)
        if GLOBAL_IS_LOG[i]:
            x_norm[i] = (np.log10(val) - np.log10(low)) / (np.log10(high) - np.log10(low))
        else:
            x_norm[i] = (val - low) / (high - low)
    return np.clip(x_norm, 0.0, 1.0)

_opt_iter_cnt = 0
def reset_opt_counter():
    global _opt_iter_cnt
    _opt_iter_cnt = 0

def estimate_initial_weld_parameters(test_configs, full_weld=False):
    baseline_timeconst = 0.002
    baseline_kbend = 223597.0
    baseline_ktwist = 6.0
    
    estimated_timeconsts = []
    
    for t_cfg in test_configs:
        t_type = t_cfg["type"]
        t_stiff = t_cfg.get("target_stiffness")
        
        if t_stiff is None and t_cfg.get("disp_hist") is not None and t_cfg.get("force_hist") is not None:
            max_d = np.max(np.abs(t_cfg["disp_hist"]))
            if max_d > 1e-4:
                idx = np.argmax(np.abs(t_cfg["disp_hist"]))
                t_stiff = abs(t_cfg["force_hist"][idx] / max_d)
                if t_stiff < 1e-6: t_stiff = 1e-6
                
        if t_stiff is not None and t_stiff > 1e-6:
            if t_type == "BENDING" or t_type == "COMPRESSION":
                tau = baseline_timeconst * np.sqrt(baseline_kbend / t_stiff)
                estimated_timeconsts.append(tau)
            elif t_type == "TWIST":
                tau = baseline_timeconst * np.sqrt(baseline_ktwist / t_stiff)
                estimated_timeconsts.append(tau)
                
    est_width, est_mid, est_pow = 0.001, 0.5, 2.0
    
    if full_weld:
        for t_cfg in test_configs:
            if t_cfg.get("disp_hist") is not None and t_cfg.get("force_hist") is not None:
                d_hist, f_hist = np.abs(t_cfg["disp_hist"]), np.abs(t_cfg["force_hist"])
                if len(d_hist) > 5 and np.max(d_hist) > 1e-4:
                    valid = (d_hist > 1e-5) & (f_hist > 1e-5)
                    if np.sum(valid) > 2:
                        p_slope, _ = np.polyfit(np.log(d_hist[valid]), np.log(f_hist[valid]), 1)
                        est_pow = np.clip(p_slope, 0.5, 5.0)
                    
                    f_diff2 = np.diff(f_hist, n=2)
                    if len(f_diff2) > 0:
                        max_curvature_idx = np.argmax(np.abs(f_diff2)) + 1
                        est_mid = np.clip(d_hist[max_curvature_idx] / np.max(d_hist), 0.1, 0.9)
                        
                    est_width = np.clip(np.std(f_hist)/np.mean(f_hist) * 0.05, 0.0001, 0.1)
                break
            
    optimal_tau = 0.002
    if estimated_timeconsts:
        optimal_tau = np.clip(np.mean(estimated_timeconsts), 0.0001, 0.5)
        
    if full_weld:
        print(f" 💡 [이론적 초기값 설정] timeconst={optimal_tau:.5f}, width={est_width:.4f}, mid={est_mid:.2f}, pow={est_pow:.2f}")
        return [optimal_tau, 1.0, 0.9, 0.95, est_width, est_mid, est_pow]
    else:
        print(f" 💡 [이론적 초기값 설정] Plate Theory & Constraint Dynamics 역산: timeconst = {optimal_tau:.5f}")
        return [optimal_tau, 1.0, 0.9, 0.95]

def objective_function(x_norm, comp_name, test_configs, full_weld, base_config=None, xml_path="temp_stiff_opt.xml"):
    global _opt_iter_cnt
    _opt_iter_cnt += 1
    
    # 디코딩 (0~1.0 -> 실제 물리 단위)
    x_real = decode_norm(x_norm)
    
    sr_timec, sr_dampr, si_dmin, si_dmax = x_real[0], x_real[1], x_real[2], x_real[3]
    si_width, si_mid, si_pow = 0.001, 0.5, 2.0
    if full_weld:
        si_width, si_mid, si_pow = x_real[4], x_real[5], x_real[6]
        
    cfg = get_default_config(base_config)
    
    # 타겟 컴포넌트 파라미터 맵핑
    if comp_name == "BCushion":
        cfg["cush_solref_timec"], cfg["cush_solref_damprr"] = sr_timec, sr_dampr
        cfg["cush_solimp_dmin"], cfg["cush_solimp_dmax"] = si_dmin, si_dmax
        if full_weld:
            cfg["cush_solimp_width"], cfg["cush_solimp_mid"], cfg["cush_solimp_power"] = si_width, si_mid, si_pow
    elif comp_name in ["BChassis", "BOpenCell"]:
        cfg["tv_solref_timec"], cfg["tv_solref_damprr"] = sr_timec, sr_dampr
        cfg["tv_solimp_dmin"], cfg["tv_solimp_dmax"] = si_dmin, si_dmax
        if full_weld:
            cfg["tv_solimp_width"], cfg["tv_solimp_mid"], cfg["tv_solimp_power"] = si_width, si_mid, si_pow
    elif comp_name == "BOpenCellCohesive":
        cfg["tape_solref_timec"], cfg["tape_solref_damprr"] = sr_timec, sr_dampr
        cfg["tape_solimp_dmin"], cfg["tape_solimp_dmax"] = si_dmin, si_dmax
        if full_weld:
            cfg["tape_solimp_width"], cfg["tape_solimp_mid"], cfg["tape_solimp_power"] = si_width, si_mid, si_pow
            
    total_loss = 0.0
    debug_str = f"[{comp_name} Opt: {_opt_iter_cnt}] solref=({sr_timec:.4f},{sr_dampr:.4f}) imp=({si_dmin:.4f},{si_dmax:.4f}"
    if full_weld: debug_str += f",{si_width:.4f},{si_mid:.4f},{si_pow:.4f})"
    else: debug_str += ")"
        
    for t_cfg in test_configs:
        t_type = t_cfg["type"]
        t_disp = t_cfg["disp"]
        t_dur  = t_cfg.get("duration", 3.0)
        t_stiff = t_cfg.get("target_stiffness", None)
        t_d_hist = t_cfg.get("disp_hist", None)
        t_f_hist = t_cfg.get("force_hist", None)
        
        cfg["plot_results"] = False
        xml_file = generate_stiffness_test_xml(comp_name, t_type, cfg)
        if os.path.exists(xml_file):
            os.replace(xml_file, xml_path)
        else: return 1e6
            
        try:
            t_hist, disp_hist, force_hist, sim_stiffness = run_stiffness_test(
                xml_path, t_type, target_disp=t_disp, duration=t_dur, plot_results=False
            )
        except Exception:
            return 1e6
            
        loss = 0.0
        
        # 1. 단일 강성 평가 (에러율 비율)
        if t_stiff is not None:
            error_ratio = abs(sim_stiffness - t_stiff) / (t_stiff + 1e-6)
            loss += error_ratio * 100.0
            
        # 2. 곡선 형태 기반 복합 평가 (cushion_optimization에서 이식)
        if t_d_hist is not None and t_f_hist is not None:
            try:
                sim_force_interp = interp1d(disp_hist, force_hist, kind='linear', bounds_error=False, fill_value="extrapolate")
                sim_force_at_target_disp = sim_force_interp(t_d_hist)
                
                # A. 기본적인 절대 수치 오차 (Magnitude)
                mse = np.mean((sim_force_at_target_disp - t_f_hist)**2)
                loss += mse / GLOBAL_MSE_WEIGHT
                
                # B. DTW Penalty (위상 및 비선형 형상 정렬)
                if HAS_FASTDTW and GLOBAL_USE_DTW:
                    try:
                        dtw_dist, _ = fastdtw(sim_force_at_target_disp.reshape(-1,1), t_f_hist.reshape(-1,1), dist=euclidean)
                        mean_tgt_mag = np.mean(np.abs(t_f_hist)) + 1e-6
                        loss += (dtw_dist / (len(sim_force_at_target_disp) * mean_tgt_mag)) * GLOBAL_SHAPE_WEIGHT
                    except: pass
                
                # C. R-squared Penalty (분산-형태 일치)
                ss_tot = np.sum((t_f_hist - np.mean(t_f_hist))**2) + 1e-8
                ss_res = np.sum((t_f_hist - sim_force_at_target_disp)**2)
                r_squared = 1.0 - (ss_res / ss_tot)
                loss += max(0.0, 1.0 - r_squared) * GLOBAL_SHAPE_WEIGHT
                
                # D. Sharpness/Curvature Penalty (부자연스러운 뾰족함 제어)
                if len(force_hist) > 3:
                    d2f = np.diff(force_hist, n=2)
                    loss += np.max(np.abs(d2f)) * GLOBAL_CURVATURE_WEIGHT
                    
                # E. Smoothness/Noise Penalty (시뮬레이션 진동 노이즈 페널티)
                sign_changes = np.sum(np.diff(np.sign(np.diff(force_hist))) != 0)
                if sign_changes > 5:
                    loss += (sign_changes * GLOBAL_SMOOTHNESS_WEIGHT)
                    
            except Exception:
                loss += 1e6
                
        # [그래프 진행 상황 시각화] (모든 모드 지원)
        try:
            plt.figure(figsize=(7, 5))
            
            # 1. Target 곡선/선분 그리기
            if t_d_hist is not None and t_f_hist is not None:
                plt.plot(t_d_hist, t_f_hist, 'tab:orange', linestyle='--', linewidth=2, label='Target F-D Curve')
            elif t_stiff is not None:
                # 단순 강성값만 있을 경우, 0점과 최대 변위 지점을 잇는 타겟 선형 궤적을 그림
                plt.plot([0, t_disp], [0, t_disp * t_stiff * (1.0 if t_disp > 0 else -1.0) if t_type == "TWIST" else t_disp * t_stiff], 
                         'tab:orange', linestyle='--', linewidth=2, label=f'Target Stiffness ({t_stiff:,.0f})')
                
            # 2. 실시간 시뮬레이션 곡선 그리기
            if len(disp_hist) > 0 and len(force_hist) > 0:
                d_clean, f_clean = np.array(disp_hist), np.array(force_hist)
                valid_idx = np.isfinite(d_clean) & np.isfinite(f_clean)
                if np.any(valid_idx):
                    plt.plot(d_clean[valid_idx], f_clean[valid_idx], 'tab:blue', linewidth=2, label=f'Current F-D Curve (k={sim_stiffness:,.0f})')
                    
            plt.title(f'{comp_name} Optimization (Loss: {total_loss + loss:.4f})')
            plt.xlabel('Displacement (m) or Angle (deg)')
            plt.ylabel('Force (N) or Torque (Nm)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'opt_progress_{comp_name}_{t_type}.png', dpi=100)
            plt.close()
        except Exception as e:
            pass
                
        total_loss += loss
        debug_str += f" | {t_type[:3]}_k:{sim_stiffness:,.0f}"
        
    print(f"{debug_str} | Total Loss: {total_loss:.4f}")
    GLOBAL_HISTORY.append((x_real.copy(), total_loss))
    return total_loss

# ==============================================================================
# Tuning Runners 
# ==============================================================================
def run_cmaes_tuning(comp_name, test_configs, full_weld, max_iterations=100, initial_guess=None, global_custom_config=None):
    if not HAS_CMAES:
        return run_traditional_tuning(comp_name, test_configs, full_weld, 'Nelder-Mead', max_iterations, initial_guess, global_custom_config)
        
    print(f"\n🚀 Starting CMA-ES Optimization (max_iter ~ {max_iterations})")
    reset_opt_counter()
    n_params = 7 if full_weld else 4
    x0_safe = [0.5] * n_params if initial_guess is None else encode_norm(initial_guess)[:n_params]
        
    def obj_wrapper(x):
        return objective_function(x, comp_name, test_configs, full_weld, base_config=global_custom_config)
        
    sigma0 = 0.25
    res = cma.fmin(obj_wrapper, x0_safe, sigma0, options={
        'bounds': [0.0, 1.0], 'maxfevals': max_iterations, 'verbose': -9,
        'popsize': max(4, int(4 + 3 * np.log(n_params)))
    })
            
    class OptResult:
        def __init__(self, x, fun, success): self.x, self.fun, self.success = x, fun, success
    return OptResult(decode_norm(res[0]), res[1], True)

def run_optuna_tuning(comp_name, test_configs, full_weld, max_iterations=100, global_custom_config=None):
    if not HAS_OPTUNA:
        return run_traditional_tuning(comp_name, test_configs, full_weld, 'Nelder-Mead', max_iterations, None, global_custom_config)
        
    print(f"\n🚀 Starting Optuna (TPE) Optimization (max_iter = {max_iterations})")
    reset_opt_counter()
    n_params = 7 if full_weld else 4
    
    def optuna_objective(trial):
        x_norm = [trial.suggest_float(f'x{i}', 0.0, 1.0) for i in range(n_params)]
        return objective_function(x_norm, comp_name, test_configs, full_weld, base_config=global_custom_config)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=max_iterations)
    
    class OptResult:
        def __init__(self, x, fun, success): self.x, self.fun, self.success = x, fun, success
    return OptResult(decode_norm([study.best_params[f'x{i}'] for i in range(n_params)]), study.best_value, True)

def run_traditional_tuning(comp_name, test_configs, full_weld, method='Nelder-Mead', max_iterations=100, initial_guess=None, global_custom_config=None):
    reset_opt_counter()
    n_params = 7 if full_weld else 4
    x0_norm = [0.5] * n_params if initial_guess is None else encode_norm(initial_guess)[:n_params]
    bnds_norm = [(0.0, 1.0) for _ in range(n_params)]
    args = (comp_name, test_configs, full_weld, global_custom_config)
    
    if method == 'dual_annealing':
        res = dual_annealing(objective_function, bounds=bnds_norm, args=args, maxiter=max_iterations, x0=x0_norm)
    elif method == 'differential_evolution':
        res = differential_evolution(objective_function, bounds=bnds_norm, args=args, maxiter=max_iterations//15)
    else:
        res = minimize(objective_function, x0_norm, args=args, method='Nelder-Mead', bounds=bnds_norm, options={'maxiter': max_iterations, 'disp': True})
    res.x = decode_norm(res.x)
    return res

if __name__ == "__main__":
    print("="*75)
    print(" 🛠️  Multi-Objective Soft Weld Stiffness Optimizer ")
    print("="*75)
    print(" 🎯 Choose Target Component:")
    print("   [1] BChassis (TV Framework - Bending/Twist Target)")
    print("   [2] BOpenCell (TV Display Panel)")
    print("   [3] BCushion (Packaging Foam - Non-linear Curve)")
    print("   [4] BOpenCellCohesive (Tape/Adhesive logic)")
    print("="*75)
    comp_choice = input("Enter choice (1~4, default 1): ").strip()
    
    comp_map = {"1": "BChassis", "2": "BOpenCell", "3": "BCushion", "4": "BOpenCellCohesive"}
    comp_name = comp_map.get(comp_choice, "BChassis")
    
    print("\n" + "="*75)
    print(" ⚙️  Choose Optimization Algorithm:")
    print("   [1] 🥇 CMA-ES (Recommended)")
    print("   [2] 🥈 Optuna / Bayesian TPE")
    print("   [3] 🥉 Nelder-Mead (Local)")
    print("   [4] Differential Evolution (Global)")
    print("   [5] Dual Annealing (Global)")
    print("="*75)
    algo_choice = input("Enter choice (1~5, default 1): ").strip()
    algo_map = {"1": "CMA", "2": "OPTUNA", "3": "NM", "4": "DE", "5": "DA"}
    algo_key = algo_map.get(algo_choice, "CMA")
    
    max_iter_input = input("\n 🔁 Enter Maximum Iterations (default 100): ").strip()
    max_iters = int(max_iter_input) if max_iter_input else 100
    
    print("\n" + "="*75)
    print(" 🔧 Choose Weld Complexity:")
    print("   [1] Base Weld (4 Params: solref, solimp_dmin, solimp_dmax)")
    print("   [2] Full Weld (7 Params: + width, mid, power for highly non-linear)")
    print("="*75)
    weld_choice = input("Enter choice (1 or 2, default: 2 if BCushion else 1): ").strip()
    full_weld = True if weld_choice == "2" or (not weld_choice and comp_name == "BCushion") else False
        
    print("\n" + "="*75)
    print(" ⚖️  Set MSE Loss Weight (Magnitude Importance):")
    print("   [1] High Sensitivity (Div by 100)")
    print("   [2] Balanced (Div by 1000) (Default)")
    print("   [3] Low Sensitivity (Div by 10000)")
    print("="*75)
    weight_choice = input("Enter choice (1, 2, 3, default 2): ").strip()
    if weight_choice == "1": GLOBAL_MSE_WEIGHT = 100.0
    elif weight_choice == "3": GLOBAL_MSE_WEIGHT = 10000.0

    print("\n" + "="*75)
    print(" 📏 Evaluate Shape Similarity using DTW?")
    print("   [Y] Yes (Slower, shape focused)")
    print("   [N] No (Faster, default)")
    print("="*75)
    dtw_choice = input("Enter choice (Y or N, default N): ").strip().upper()
    if dtw_choice == "Y": GLOBAL_USE_DTW = True

    print("\n" + "="*75)
    print(f" 🎯 Formulate Target Test Configurations for [{comp_name}]")
    print("="*75)
    
    test_configs = []
    
    # 1. 컴포넌트 크기 (차원) 설정
    print("\n [Step 1] Geometry Configuration")
    print("  -> (설정된 크기에 따라 xml이 자동 생성되며, 강성 역산에도 영향을 줍니다)")
    box_w_input = input("  Enter 'box_w' (Width, default 1.5): ").strip()
    box_w = float(box_w_input) if box_w_input else 1.5
    
    box_h_input = input("  Enter 'box_h' (Height, default 0.9): ").strip()
    box_h = float(box_h_input) if box_h_input else 0.9
    
    box_thick_input = input("  Enter 'box_thick' (Thickness, default 0.05): ").strip()
    box_thick = float(box_thick_input) if box_thick_input else 0.05
    
    # 전역 base_config를 덮어씌워 objective_function에 전달해야 함 (단, 함수의 인자가 제한적이므로 임시 저장소 역할)
    global_custom_config = {
        "box_w": box_w,
        "box_h": box_h,
        "box_thick": box_thick
    }
    
    # 2. 부품별 지원 평가 모드 설정
    print("\n [Step 2] Evaluation Modes & Targets")
    if comp_name == "BCushion":
        print("  -> BCushion supports [COMPRESSION] primarily.")
        target_disp_input = input("  Enter Max Compression Displacement (m) (default -0.05): ").strip()
        t_disp = float(target_disp_input) if target_disp_input else -0.05
        
        # 임시 비선형 타겟 데이터 생성 (차후 외부 데이터 임포트 가능)
        dummy_disp = np.linspace(0, abs(t_disp), 50) 
        dummy_force = 100000.0 * dummy_disp**3.5 
        test_configs.append({"type": "COMPRESSION", "disp": t_disp, "disp_hist": -dummy_disp, "force_hist": dummy_force, "duration": 3.0})
        
    else:
        # BChassis, BOpenCell 등은 굽힘/비틀림 복합 검증이 기본
        print("  -> Supports [BENDING] and/or [TWIST]")
        use_bend = input("  Include BENDING test? (Y/N, default Y): ").strip().upper() != "N"
        if use_bend:
            trg_stiff = input("    Target BENDING Stiffness (N/m, default 5000.0): ").strip()
            test_configs.append({
                "type": "BENDING", 
                "disp": -0.05, 
                "target_stiffness": float(trg_stiff) if trg_stiff else 5000.0, 
                "duration": 3.0
            })
            
        use_twist = input("  Include TWIST test? (Y/N, default Y): ").strip().upper() != "N"
        if use_twist:
            trg_stiff = input("    Target TWIST Stiffness (Nm/deg, default 15.0): ").strip()
            test_configs.append({
                "type": "TWIST", 
                "disp": 20.0, 
                "target_stiffness": float(trg_stiff) if trg_stiff else 15.0, 
                "duration": 3.0
            })
            
    if not test_configs:
        print("  ❌ No tests selected. Exiting.")
        sys.exit(0)
        
    initial_guess = estimate_initial_weld_parameters(test_configs, full_weld)
    
    if algo_key == "CMA": res = run_cmaes_tuning(comp_name, test_configs, full_weld, max_iters, initial_guess, global_custom_config)
    elif algo_key == "OPTUNA": res = run_optuna_tuning(comp_name, test_configs, full_weld, max_iters, global_custom_config)
    elif algo_key == "NM": res = run_traditional_tuning(comp_name, test_configs, full_weld, 'Nelder-Mead', max_iters, initial_guess, global_custom_config)
    elif algo_key == "DE": res = run_traditional_tuning(comp_name, test_configs, full_weld, 'differential_evolution', max_iters, None, global_custom_config)
    elif algo_key == "DA": res = run_traditional_tuning(comp_name, test_configs, full_weld, 'dual_annealing', max_iters, initial_guess, global_custom_config)

    if res is not None:
        print("\n" + "="*75)
        print(" ✅ Optimization Complete")
        print(f" Success: {res.success}")
        print(f" Final Loss: {res.fun:.4f}")
        
        best = res.x
        print(f"\n[Optimal {comp_name} Parameters Found]")
        print(f"  'solref_timec': {best[0]:.5f}")
        print(f"  'solref_dampr': {best[1]:.5f}")
        print(f"  'solimp_dmin': {best[2]:.5f}")
        print(f"  'solimp_dmax': {best[3]:.5f}")
        if full_weld:
            print(f"  'solimp_width': {best[4]:.5f}")
            print(f"  'solimp_mid': {best[5]:.5f}")
            print(f"  'solimp_power': {best[6]:.5f}")
        print("="*75)
        
        # --- Bounds Suggestion System (생략 유지) ---
        if len(GLOBAL_HISTORY) > 10:
            print("\n🔍  [Search Space Analysis & Suggestion]")
            print("-" * 75)
            sorted_history = sorted(GLOBAL_HISTORY, key=lambda x: x[1])
            top_n = max(5, len(sorted_history) // 10)
            top_performers = np.array([h[0] for h in sorted_history[:top_n]])
            
            print(f"  Analyzed top {top_n} performers out of {len(sorted_history)} iterations.")
            print("  Suggested New GLOBAL_BOUNDS (Copy-paste to replace current settings):")
            print("\nGLOBAL_BOUNDS = [")
            
            param_names = ["solref_timec", "solref_dampr", "solimp_dmin", "solimp_dmax", "solimp_width", "solimp_mid", "solimp_power"]
            n_p = 7 if full_weld else 4
            
            for i in range(n_p):
                p_min = np.min(top_performers[:, i])
                p_max = np.max(top_performers[:, i])
                margin = (p_max - p_min) * 0.2 + 1e-6
                suggested_low = max(GLOBAL_BOUNDS[i][0], p_min - margin)
                suggested_high = min(GLOBAL_BOUNDS[i][1], p_max + margin)
                comma = "," if i < n_p-1 else ""
                print(f"    [{suggested_low:10.5f}, {suggested_high:10.5f}]{comma}  # x[{i}]: {param_names[i]}")
            print("]")
            print("-" * 75)
            
        print("\n 📊 Generating final optimization plot with the best parameters...")
        # 최고 성능 파라미터로 최종 시뮬레이션을 돌려 그래프를 마지막으로 갱신합니다.
        best_norm = encode_norm(best)
        objective_function(best_norm, comp_name, test_configs, full_weld, base_config=global_custom_config, xml_path="temp_stiff_opt.xml")
        
        for t_cfg in test_configs:
            png_name = f"opt_progress_{comp_name}_{t_cfg['type']}.png"
            print(f"   ✅ [Final Plot Saved] Please check '{png_name}'")
            
        xml_temp = "temp_stiff_opt.xml"
        if os.path.exists(xml_temp): os.remove(xml_temp)
