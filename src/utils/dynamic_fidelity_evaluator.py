# -*- coding: utf-8 -*-
"""
dynamic_fidelity_evaluator.py
=============================
Specialized evaluation and calibration module for 1D/3D dynamic time-series
data (displacement & velocity).
"""

import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from typing import Callable, Tuple, List, Optional, Dict, Any


class ComprehensiveDynamicEvaluator:
    """
    ========================================================================================================================
    [1D & 3D 동역학 시계열 정합도 평가 지표 및 최적화 연동 체계]
    ========================================================================================================================
    | 차원 (Dim)  | 모드 (Mode) | 주 평가지표 (Headline 100%) | 기준 척도 (0% Base)      | 보조 및 세부 진단 지표 (Detailed Metrics)         | 최적화 손실 (calc_loss)           |
    |-------------|-------------|-----------------------------|--------------------------|----------------------------------------------------|-----------------------------------|
    | 1D Scalar   | 변위 (disp) | Global NRMSE Score          | Peak-to-Peak 폭 (Δx)     | • 통계: MSE, RMSE, R² (NSE), 피어슨 상관계수 (Corr) | 0.7 * L2(오차/Δx)²                |
    |             |             | max(0, 1 - RMSE/Δx) * 100   | (xmax - xmin)            | • 경로: DTW 거리/점수, 최대 피크 변위, 최종 잔류 변위 | + 0.3 * (피크오차/Δx)²            |
    |-------------|-------------|-----------------------------|--------------------------|----------------------------------------------------|-----------------------------------|
    | 1D Scalar   | 속도 (vel)  | Sprague & Geers Score       | 1D 적분 에너지 (∫v² dt)  | • S&G 분해: M(크기 오차), P(위상/타이밍 오차), C   | 0.4 * L2(오차) / ∫v² dt           |
    |             |             | max(0, 1 - C) * 100         |                          | • 경량 CORA 3축: Kp(지연ms), Kv(형상), Kg(면적/피크)| + 0.3 * (피크오차비)² + 0.3 * 면적비² |
    |-------------|-------------|-----------------------------|--------------------------|----------------------------------------------------|-----------------------------------|
    | 3D Vector   | 변위 (disp) | Spatial ATE Score           | Bounding Box 대각선      | • ATE 공간 RMSE: sqrt(mean(||r_exp - r_sim||²))    | 0.7 * L2(공간오차/L_diag)²         |
    |             |             | max(0, 1 - RMSE_3D/L_diag)  | L_diag = norm([dx,dy,dz])| • 기하 형상: Hausdorff (최악 이탈점), Fréchet 거리   | + 0.3 * (피크오차/L_diag)²        |
    |             |             |                             |                          | • 강체 정렬: Procrustes Disparity (순수 형상 일치) |                                   |
    |-------------|-------------|-----------------------------|--------------------------|----------------------------------------------------|-----------------------------------|
    | 3D Vector   | 속도 (vel)  | Vector Sprague & Geers      | 3D 총 운동에너지          | • M_3D: 3축 벡터 합성 크기/에너지 오차              | 0.4 * L2(3D오차) / ∫||v||² dt     |
    |             |             | max(0, 1 - C_3D) * 100      | ∫ (vx² + vy² + vz²) dt   | • P_3D: 시간 지연 + 3차원 진행 방향(Orientation) 오차| + 0.3 * (피크오차비)² + 0.3 * 면적비² |
    ========================================================================================================================

    매개변수:
    ----------
    time : np.ndarray
        (N,) 형태의 시간 배열 (단위: s)
    y_exp : np.ndarray
        (N,) 형태의 1D 신호 또는 (N, 3) 형태의 3D 공간 벡터 신호 (시험/기준 데이터)
    mode : str ("vel" 또는 "disp")
        신호 유형 ("vel": 속도/충격량 중심, "disp": 위치/궤적 중심)
    ref_scale : float (선택)
        사용자 정의 기준 스케일 (L_ref, V_ref). 미지정 시 데이터에서 자동 산출.
    tau_allow : float (선택)
        CORA 시간 지연 평가 시 0%로 간주할 허용 최대 지연 시간 (초, 기본값: 총 시간의 10%)
    """

    def __init__(
        self,
        time: np.ndarray,
        y_exp: np.ndarray,
        mode: str = "vel",
        ref_scale: Optional[float] = None,
        tau_allow: Optional[float] = None,
    ):
        self.t = np.asarray(time, dtype=np.float64)
        self.y_exp = np.asarray(y_exp, dtype=np.float64)
        self.mode = mode.lower()
        self.dt = np.gradient(self.t)
        self.N = len(self.t)

        if self.mode not in ["vel", "disp"]:
            raise ValueError("mode는 'vel' 또는 'disp'이어야 합니다.")

        # 1. 1D vs 3D 차원 판별
        if self.y_exp.ndim == 1 or (self.y_exp.ndim == 2 and self.y_exp.shape[1] == 1):
            self.is_3d = False
            self.y_exp = self.y_exp.flatten()
        elif self.y_exp.ndim == 2 and self.y_exp.shape[1] == 3:
            self.is_3d = True
        else:
            raise ValueError("y_exp는 (N,) 형태의 1D 신호이거나 (N, 3) 형태의 3D 신호여야 합니다.")

        # 2. 물리적 기준 척도(Global Scale) 자동 산출
        if ref_scale is not None:
            self.ref_scale = float(ref_scale)
            self.ptp_per_axis = np.ptp(self.y_exp, axis=0) if self.is_3d else np.ptp(self.y_exp)
        else:
            if not self.is_3d:
                self.ptp_per_axis = float(np.ptp(self.y_exp))
                self.ref_scale = max(self.ptp_per_axis, 1e-9)
            else:
                self.ptp_per_axis = np.ptp(self.y_exp, axis=0)  # [dx, dy, dz]
                self.ref_scale = float(np.linalg.norm(self.ptp_per_axis))
                if self.ref_scale < 1e-9:
                    self.ref_scale = 1.0  # 정적 상태 방어 기본값

        # 3. 시간 지연 허용치 설정
        total_time = self.t[-1] - self.t[0]
        self.tau_allow = tau_allow if tau_allow is not None else max(0.02, total_time * 0.1)

    # =========================================================================
    # [1D 계열 세부 평가 함수군]
    # =========================================================================
    def calc_1d_statistical(self, y_sim: np.ndarray) -> Dict[str, float]:
        """MSE, RMSE, R2 (NSE), Pearson Correlation 계산"""
        y_sim = y_sim.flatten()
        mse = float(np.mean((self.y_exp - y_sim) ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = np.sum((self.y_exp - y_sim) ** 2)
        ss_tot = np.sum((self.y_exp - np.mean(self.y_exp)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

        std_e, std_s = np.std(self.y_exp), np.std(y_sim)
        corr = float(np.corrcoef(self.y_exp, y_sim)[0, 1]) if (std_e > 1e-9 and std_s > 1e-9) else 0.0

        return {"MSE": mse, "RMSE": rmse, "R2": r2, "Pearson_Corr": corr}

    def calc_1d_dtw(self, y_sim: np.ndarray) -> Dict[str, float]:
        """1D Dynamic Time Warping (DTW) 점별 평균 누적 거리"""
        s1, s2 = self.y_exp, y_sim.flatten()
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )
        dtw_dist = float(dtw_matrix[n, m] / n)
        dtw_score = max(0.0, (1.0 - dtw_dist / self.ref_scale) * 100.0)
        return {"DTW_Distance": dtw_dist, "DTW_Score(%)": round(dtw_score, 1)}

    def calc_1d_cora_light(self, y_sim: np.ndarray) -> Dict[str, float]:
        """경량 CORA 3축 분해 평가: 시간 딜레이(Kp), 형상(Kv), 크기/면적(Kg)"""
        y_sim = y_sim.flatten()
        dt = float(np.mean(self.dt))

        # 1) 시간 지연 (Phase shift, Kp)
        corr = signal.correlate(y_sim - np.mean(y_sim), self.y_exp - np.mean(self.y_exp), mode="full")
        lags = signal.correlation_lags(self.N, self.N, mode="full") * dt
        best_lag = float(lags[np.argmax(corr)])
        k_p = max(0.0, 1.0 - abs(best_lag) / self.tau_allow)

        # 2) 위상 보정 후 순수 형상 (Shape, Kv)
        shift_steps = int(round(best_lag / dt))
        if shift_steps > 0:
            y_sim_shifted = np.pad(y_sim, (0, shift_steps), mode="edge")[shift_steps:]
        elif shift_steps < 0:
            y_sim_shifted = np.pad(y_sim, (-shift_steps, 0), mode="edge")[:shift_steps]
        else:
            y_sim_shifted = y_sim.copy()

        std_e, std_s = np.std(self.y_exp), np.std(y_sim_shifted)
        k_v = max(0.0, float(np.corrcoef(self.y_exp, y_sim_shifted)[0, 1])) if (std_e > 1e-9 and std_s > 1e-9) else 0.0

        # 3) 크기 및 에너지 (Size/Area, Kg)
        peak_e, peak_s = float(np.max(np.abs(self.y_exp))), float(np.max(np.abs(y_sim)))
        s_peak = max(0.0, 1.0 - abs(peak_e - peak_s) / max(peak_e, peak_s, 1e-9))
        area_e = float(np.sum(np.abs(self.y_exp) * self.dt)) + 1e-9
        area_s = float(np.sum(np.abs(y_sim) * self.dt)) + 1e-9
        s_area = min(area_s / area_e, area_e / area_s)
        k_g = 0.5 * s_peak + 0.5 * s_area

        cora_score = (k_p * k_v * k_g) * 100.0
        return {
            "CORA_Score(%)": round(cora_score, 1),
            "Delay_Score_Kp": round(k_p, 4),
            "Time_Delay_ms": round(best_lag * 1000.0, 2),
            "Shape_Score_Kv": round(k_v, 4),
            "Magnitude_Score_Kg": round(k_g, 4),
        }

    # =========================================================================
    # [3D 계열 세부 궤적 평가 함수군]
    # =========================================================================
    def calc_3d_frechet(self, r_sim: np.ndarray) -> Dict[str, float]:
        """3차원 이산 프레셰 거리 (Discrete Fréchet Distance)"""
        p, q = self.y_exp, np.asarray(r_sim, dtype=np.float64)
        n, m = len(p), len(q)
        ca = np.full((n, m), -1.0)

        def c_dist_recur(i, j):
            if ca[i, j] > -0.5:
                return ca[i, j]
            d = np.linalg.norm(p[i] - q[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i > 0 and j == 0:
                ca[i, j] = max(c_dist_recur(i - 1, 0), d)
            elif i == 0 and j > 0:
                ca[i, j] = max(c_dist_recur(0, j - 1), d)
            else:
                ca[i, j] = max(
                    min(
                        c_dist_recur(i - 1, j),
                        c_dist_recur(i - 1, j - 1),
                        c_dist_recur(i, j - 1),
                    ),
                    d,
                )
            return ca[i, j]

        stride = max(1, n // 100)
        p_sub, q_sub = p[::stride], q[::stride]
        d_F = float(cdist(p_sub, q_sub).max() if len(p_sub) > 200 else c_dist_recur(len(p_sub) - 1, len(q_sub) - 1))
        score_frechet = max(0.0, (1.0 - d_F / self.ref_scale) * 100.0)
        return {"Frechet_Dist(m)": round(d_F, 5), "Frechet_Score(%)": round(score_frechet, 1)}

    def calc_3d_procrustes(self, r_sim: np.ndarray) -> Dict[str, float]:
        """SVD 기반 강체 정렬(회전/이동/스케일 최적화) 순수 형상 잔차 분석"""
        X = self.y_exp - np.mean(self.y_exp, axis=0)
        Y = np.asarray(r_sim, dtype=np.float64) - np.mean(r_sim, axis=0)

        norm_X = np.sqrt(np.sum(X**2))
        norm_Y = np.sqrt(np.sum(Y**2))
        if norm_X < 1e-12 or norm_Y < 1e-12:
            return {"Procrustes_Disparity": 1.0, "Procrustes_Score(%)": 0.0}

        X /= norm_X
        Y /= norm_Y

        H = np.dot(Y.T, X)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(U, Vt)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            S[-1] *= -1

        disparity = float(max(0.0, 1.0 - (np.sum(S)) ** 2))
        score_proc = max(0.0, (1.0 - np.sqrt(disparity)) * 100.0)
        return {"Procrustes_Disparity": round(disparity, 5), "Procrustes_Score(%)": round(score_proc, 1)}

    def calc_3d_hausdorff(self, r_sim: np.ndarray) -> Dict[str, float]:
        """3차원 하우스도르프 최악 이탈 거리 (Worst-case Peak Error)"""
        dists = cdist(self.y_exp, np.asarray(r_sim, dtype=np.float64))
        d_H = float(max(np.max(np.min(dists, axis=1)), np.max(np.min(dists, axis=0))))
        score_h = max(0.0, (1.0 - d_H / self.ref_scale) * 100.0)
        return {"Hausdorff_Dist(m)": round(d_H, 5), "Hausdorff_Score(%)": round(score_h, 1)}

    # =========================================================================
    # [1D/3D 공통: Sprague & Geers 코어]
    # =========================================================================
    def calc_sprague_geers(self, y_sim: np.ndarray) -> Dict[str, float]:
        """1D 스칼라 및 3D 공간 벡터 통합 Sprague & Geers 계산"""
        y_sim = np.asarray(y_sim, dtype=np.float64)
        if not self.is_3d:
            I_ee = np.sum(self.y_exp**2 * self.dt)
            I_ss = np.sum(y_sim.flatten() ** 2 * self.dt)
            I_es = np.sum(self.y_exp * y_sim.flatten() * self.dt)
        else:
            I_ee = np.sum(np.sum(self.y_exp * self.y_exp, axis=1) * self.dt)
            I_ss = np.sum(np.sum(y_sim * y_sim, axis=1) * self.dt)
            I_es = np.sum(np.sum(self.y_exp * y_sim, axis=1) * self.dt)

        eps = 1e-12
        I_ee = max(I_ee, eps)
        I_ss = max(I_ss, eps)

        M = float(np.sqrt(I_ss / I_ee) - 1.0)
        cos_val = np.clip(I_es / np.sqrt(I_ee * I_ss), -1.0, 1.0)
        P = float((1.0 / np.pi) * np.arccos(cos_val))
        C = float(np.sqrt(M**2 + P**2))
        score = float(max(0.0, (1.0 - C) * 100.0))
        return {"M": M, "P": P, "C": C, "Score(%)": score}

    # =========================================================================
    # [종합 보고서 출력: evaluate]
    # =========================================================================
    def evaluate(self, y_sim: np.ndarray) -> Dict[str, Any]:
        """입력 차원 및 모드에 따른 100% 만점 체계의 정합도 보고서 딕셔너리 생성"""
        y_sim = np.asarray(y_sim, dtype=np.float64)
        report: Dict[str, Any] = {
            "Dimension": "3D Vector" if self.is_3d else "1D Scalar",
            "Mode": "Velocity" if self.mode == "vel" else "Displacement",
            "Reference_Scale": round(self.ref_scale, 4),
        }

        # -------------------------------------------------------------
        # 1D 모드 보고서
        # -------------------------------------------------------------
        if not self.is_3d:
            stats = self.calc_1d_statistical(y_sim)
            dtw = self.calc_1d_dtw(y_sim)
            cora = self.calc_1d_cora_light(y_sim)
            sg = self.calc_sprague_geers(y_sim)

            nrmse_score = max(0.0, (1.0 - stats["RMSE"] / self.ref_scale) * 100.0)

            if self.mode == "vel":
                report["Main_Headline_Accuracy(%)"] = round(sg["Score(%)"], 1)
                report["Sprague_Geers"] = {"M": round(sg["M"], 4), "P": round(sg["P"], 4), "C": round(sg["C"], 4)}
                report["CORA_Light_Decomposition"] = cora
            else:
                report["Main_Headline_Accuracy(%)"] = round(nrmse_score, 1)
                report["Global_NRMSE_Score(%)"] = round(nrmse_score, 1)
                report["Residual_Disp_Error(m)"] = round(
                    abs(
                        np.mean(self.y_exp[-int(self.N * 0.1) :])
                        - np.mean(y_sim.flatten()[-int(self.N * 0.1) :])
                    ),
                    5,
                )

            report["Statistical_Metrics"] = {
                "RMSE": round(stats["RMSE"], 5),
                "R2": round(stats["R2"], 4),
                "Pearson_Corr": round(stats["Pearson_Corr"], 4),
            }
            report["DTW"] = dtw

        # -------------------------------------------------------------
        # 3D 모드 보고서
        # -------------------------------------------------------------
        else:
            sg_3d = self.calc_sprague_geers(y_sim)
            frechet = self.calc_3d_frechet(y_sim)
            procrustes = self.calc_3d_procrustes(y_sim)
            hausdorff = self.calc_3d_hausdorff(y_sim)

            diff_r = self.y_exp - y_sim
            spatial_rmse = float(np.sqrt(np.mean(np.sum(diff_r**2, axis=1))))
            ate_score = max(0.0, (1.0 - spatial_rmse / self.ref_scale) * 100.0)

            if self.mode == "vel":
                report["Main_Headline_Accuracy(%)"] = round(sg_3d["Score(%)"], 1)
                report["Vector_Sprague_Geers"] = {
                    "M_Energy": round(sg_3d["M"], 4),
                    "P_Phase_Orientation": round(sg_3d["P"], 4),
                    "C_Combined": round(sg_3d["C"], 4),
                }
            else:
                report["Main_Headline_Accuracy(%)"] = round(ate_score, 1)
                report["Spatial_ATE_Score(%)"] = round(ate_score, 1)

            report["3D_Trajectory_Metrics"] = {
                "Spatial_RMSE_ATE(m)": round(spatial_rmse, 5),
                "Bounding_Box_Diag_Lref(m)": round(self.ref_scale, 4),
                "Hausdorff": hausdorff,
                "Frechet": frechet,
                "Procrustes_Pure_Shape": procrustes,
            }

        return report

    # =========================================================================
    # [최적화 엔진용 손실 함수: calc_loss]
    # =========================================================================
    def calc_loss(self, y_sim: np.ndarray) -> float:
        """
        SciPy, JAX, Optuna 등 구배 기반 탐색기가 원활히 수렴하도록
        불연속점을 배제하고 설계된 전구간 연속 미분 가능 무차원 복합 손실값 반환
        """
        y_sim = np.asarray(y_sim, dtype=np.float64)

        if not self.is_3d:
            y_sim = y_sim.flatten()
            if self.mode == "vel":
                p_exp, p_sim = float(np.max(np.abs(self.y_exp))), float(np.max(np.abs(y_sim)))
                loss_peak = ((p_exp - p_sim) / (p_exp + 1e-6)) ** 2
                a_exp = float(np.sum(np.abs(self.y_exp) * self.dt))
                a_sim = float(np.sum(np.abs(y_sim) * self.dt))
                loss_area = ((a_exp - a_sim) / (a_exp + 1e-6)) ** 2
                loss_l2 = float(np.sum((self.y_exp - y_sim) ** 2) / (np.sum(self.y_exp**2) + 1e-6))
                return float(0.3 * loss_peak + 0.3 * loss_area + 0.4 * loss_l2)
            else:
                loss_l2 = float(np.mean((self.y_exp - y_sim) ** 2) / (self.ref_scale**2 + 1e-6))
                loss_peak = float(((np.max(self.y_exp) - np.max(y_sim)) / self.ref_scale) ** 2)
                return float(0.7 * loss_l2 + 0.3 * loss_peak)

        else:
            if self.mode == "vel":
                p_exp = float(np.max(np.linalg.norm(self.y_exp, axis=1)))
                p_sim = float(np.max(np.linalg.norm(y_sim, axis=1)))
                loss_peak = ((p_exp - p_sim) / (p_exp + 1e-6)) ** 2
                a_exp = float(np.sum(np.linalg.norm(self.y_exp, axis=1) * self.dt))
                a_sim = float(np.sum(np.linalg.norm(y_sim, axis=1) * self.dt))
                loss_area = ((a_exp - a_sim) / (a_exp + 1e-6)) ** 2
                loss_l2 = float(np.sum((self.y_exp - y_sim) ** 2) / (np.sum(self.y_exp**2) + 1e-6))
                return float(0.3 * loss_peak + 0.3 * loss_area + 0.4 * loss_l2)
            else:
                loss_l2 = float(np.mean(np.sum((self.y_exp - y_sim) ** 2, axis=1)) / (self.ref_scale**2 + 1e-6))
                p_exp = float(np.max(np.linalg.norm(self.y_exp, axis=1)))
                p_sim = float(np.max(np.linalg.norm(y_sim, axis=1)))
                loss_peak = ((p_exp - p_sim) / (self.ref_scale + 1e-6)) ** 2
                return float(0.7 * loss_l2 + 0.3 * loss_peak)


def optimize_dynamic_parameters(
    sim_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    init_params: np.ndarray,
    time: np.ndarray,
    y_exp: np.ndarray,
    mode: str = "vel",
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = "L-BFGS-B",
    ref_scale: Optional[float] = None,
    options: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    ComprehensiveDynamicEvaluator를 목적함수로 삼아 시뮬레이션 파라미터를
    자동 보정(Calibration)하고 보정 전/후 정합도 리포트를 산출하는 함수
    """
    evaluator = ComprehensiveDynamicEvaluator(
        time=time,
        y_exp=y_exp,
        mode=mode,
        ref_scale=ref_scale,
    )

    init_p = np.asarray(init_params, dtype=np.float64)
    history = {"iter": 0, "loss": []}

    def objective_wrapper(params: np.ndarray) -> float:
        try:
            y_sim = sim_func(params, time)
            y_sim = np.asarray(y_sim, dtype=np.float64)

            if y_sim.shape != y_exp.shape and y_sim.flatten().shape != y_exp.flatten().shape:
                return 1e6

            loss = evaluator.calc_loss(y_sim)

            if not np.isfinite(loss):
                return 1e6
        except Exception:
            return 1e6

        history["iter"] += 1
        history["loss"].append(loss)
        return float(loss)

    y_sim_init = sim_func(init_p, time)
    init_report = evaluator.evaluate(y_sim_init)
    init_loss = objective_wrapper(init_p)

    if verbose:
        print("==================================================")
        print(f"[동역학 파라미터 최적화 시작] Mode: {mode.upper()} | Dim: {'3D' if evaluator.is_3d else '1D'}")
        print(f"  초기 파라미터: {init_p}")
        print(f"  초기 손실값(Loss): {init_loss:.6e}")
        print(f"  초기 Headline 정합도: {init_report['Main_Headline_Accuracy(%)']:.1f}%")
        print("--------------------------------------------------")

    opt_options = options if options is not None else {"maxiter": 150}
    res = minimize(
        fun=objective_wrapper,
        x0=init_p,
        method=method,
        bounds=bounds,
        options=opt_options,
    )

    opt_params = res.x
    y_sim_opt = sim_func(opt_params, time)
    final_report = evaluator.evaluate(y_sim_opt)

    init_score = init_report["Main_Headline_Accuracy(%)"]
    final_score = final_report["Main_Headline_Accuracy(%)"]

    if verbose:
        print(f"[최적화 완료] 수렴 성공 여부: {res.success} ({res.message})")
        print(f"  반복 횟수: {res.nit}회 (목적함수 호출 {res.nfev}회)")
        print(f"  최적 파라미터: {np.round(opt_params, 5)}")
        print(f"  최종 손실값(Loss): {res.fun:.6e} (감소율: {(1.0 - res.fun / max(init_loss, 1e-12)) * 100:.1f}%)")
        print(f"  >> 정합도 개선: {init_score:.1f}% ──> {final_score:.1f}% (+{final_score - init_score:.1f}%p)")
        print("==================================================")

    return {
        "optimal_params": opt_params,
        "success": res.success,
        "message": res.message,
        "iterations": res.nit,
        "initial_loss": init_loss,
        "final_loss": float(res.fun),
        "initial_accuracy(%)": init_score,
        "calibrated_accuracy(%)": final_score,
        "initial_report": init_report,
        "calibrated_report": final_report,
        "y_sim_calibrated": y_sim_opt,
        "evaluator": evaluator,
        "scipy_result": res,
    }
