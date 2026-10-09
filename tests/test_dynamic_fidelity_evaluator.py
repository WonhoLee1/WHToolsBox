# -*- coding: utf-8 -*-
"""
test_dynamic_fidelity_evaluator.py
==================================
Unit tests for ComprehensiveDynamicEvaluator and optimize_dynamic_parameters.
"""

import pytest
import numpy as np
from src.utils.dynamic_fidelity_evaluator import (
    ComprehensiveDynamicEvaluator,
    optimize_dynamic_parameters,
)


def test_1d_vel_perfect_match():
    t = np.linspace(0, 1.0, 200)
    v_exp = -9.81 * t

    evaluator = ComprehensiveDynamicEvaluator(t, v_exp, mode="vel")
    report = evaluator.evaluate(v_exp)

    assert report["Dimension"] == "1D Scalar"
    assert report["Mode"] == "Velocity"
    assert report["Main_Headline_Accuracy(%)"] == 100.0
    assert report["Sprague_Geers"]["C"] == 0.0
    assert report["CORA_Light_Decomposition"]["CORA_Score(%)"] == 100.0
    assert evaluator.calc_loss(v_exp) < 1e-12


def test_1d_disp_perfect_match():
    t = np.linspace(0, 1.0, 200)
    x_exp = 1.0 - 0.5 * 9.81 * (t**2)

    evaluator = ComprehensiveDynamicEvaluator(t, x_exp, mode="disp")
    report = evaluator.evaluate(x_exp)

    assert report["Dimension"] == "1D Scalar"
    assert report["Mode"] == "Displacement"
    assert report["Main_Headline_Accuracy(%)"] == 100.0
    assert report["Global_NRMSE_Score(%)"] == 100.0
    assert evaluator.calc_loss(x_exp) < 1e-12


def test_1d_vel_degraded():
    t = np.linspace(0, 1.0, 200)
    v_exp = -9.81 * np.clip(t, 0, 0.45)
    v_sim = -9.81 * 0.8 * np.clip(t - 0.03, 0, 0.45)

    evaluator = ComprehensiveDynamicEvaluator(t, v_exp, mode="vel")
    report = evaluator.evaluate(v_sim)

    assert report["Main_Headline_Accuracy(%)"] < 100.0
    assert report["Sprague_Geers"]["M"] != 0.0
    assert report["Sprague_Geers"]["P"] > 0.0
    assert evaluator.calc_loss(v_sim) > evaluator.calc_loss(v_exp)


def test_3d_disp_perfect_match():
    t = np.linspace(0, 1.0, 100)
    r_exp = np.stack([2.0 * t, np.sin(5 * t), 10.0 - 4.9 * t**2], axis=1)

    evaluator = ComprehensiveDynamicEvaluator(t, r_exp, mode="disp")
    report = evaluator.evaluate(r_exp)

    assert report["Dimension"] == "3D Vector"
    assert report["Mode"] == "Displacement"
    assert report["Main_Headline_Accuracy(%)"] == 100.0
    assert report["Spatial_ATE_Score(%)"] == 100.0
    assert report["3D_Trajectory_Metrics"]["Procrustes_Pure_Shape"]["Procrustes_Score(%)"] == 100.0
    assert report["3D_Trajectory_Metrics"]["Frechet"]["Frechet_Score(%)"] == 100.0
    assert report["3D_Trajectory_Metrics"]["Hausdorff"]["Hausdorff_Score(%)"] == 100.0


def test_3d_vel_perfect_match():
    t = np.linspace(0, 1.0, 100)
    v_exp = np.stack([2.0 * np.ones_like(t), 5 * np.cos(5 * t), -9.8 * t], axis=1)

    evaluator = ComprehensiveDynamicEvaluator(t, v_exp, mode="vel")
    report = evaluator.evaluate(v_exp)

    assert report["Dimension"] == "3D Vector"
    assert report["Mode"] == "Velocity"
    assert report["Main_Headline_Accuracy(%)"] == 100.0
    assert report["Vector_Sprague_Geers"]["C_Combined"] == 0.0


def test_optimize_dynamic_parameters():
    t = np.linspace(0, 0.5, 200)

    def sim_model(params, time):
        scale, delay = params
        t_adj = np.maximum(0.0, time - delay)
        return -9.81 * scale * t_adj

    true_params = np.array([1.2, 0.01])
    v_target = sim_model(true_params, t)

    init_params = np.array([0.8, 0.03])

    opt_res = optimize_dynamic_parameters(
        sim_func=sim_model,
        init_params=init_params,
        time=t,
        y_exp=v_target,
        mode="vel",
        bounds=[(0.5, 2.0), (0.0, 0.05)],
        verbose=False,
    )

    assert opt_res["success"] is True
    assert opt_res["calibrated_accuracy(%)"] > opt_res["initial_accuracy(%)"]
    assert np.allclose(opt_res["optimal_params"], true_params, atol=1e-2)
