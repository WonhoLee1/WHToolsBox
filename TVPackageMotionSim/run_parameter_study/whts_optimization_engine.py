# -*- coding: utf-8 -*-
"""
[WHTOOLS] Optimization & DOE Engine (v1.0)
GooeyParser 입력에 대응하여 Latin Hypercube, Random, Full Factorial 방식의
DOE 테이블을 생성하고, MuJoCo 낙하 시뮬레이션을 배치 실행하며, 최적안을 추출하는 핵심 엔진입니다.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# UTF-8 인코딩 강제
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

# [WHTOOLS] 경로 설정 보완 (run_parameter_study용)
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)
parent_dir = os.path.dirname(curr_dir)  # TVPackageMotionSim
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
grandparent_dir = os.path.dirname(parent_dir)  # WHToolsBox
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

from run_drop_simulator import DropSimulator
from run_discrete_builder.whtb_config import load_config, save_config
from run_drop_simulator.whts_multipostprocessor_engine import ShellDeformationAnalyzer, scale_result_to_mm


class DOEEngine:
    """
    [WHTOOLS] DOE(실험계획법) 샘플링 및 테이블 생성을 전담하는 클래스입니다.
    """

    def __init__(self, variables: List[Dict[str, Any]], sampling_method: str = "LHS", sample_count: int = 10):
        """
        Parameters
        ----------
        variables : List[Dict[str, Any]]
            각 변수의 정의 목록.
            예: [{"name": "cush_friction", "min": 0.2, "max": 1.0, "type": "Continuous"},
                 {"name": "box_w", "min": 1.8, "max": 2.2, "type": "Discrete", "step": 0.05}]
        sampling_method : str
            샘플링 기법 ("LHS", "Random", "FullFact")
        sample_count : int
            LHS 또는 Random 사용 시 생성할 샘플 수 (FullFact에서는 무시됨)
        """
        self.variables = variables
        self.sampling_method = sampling_method
        self.sample_count = sample_count

    def generate_doe_table(self) -> List[Dict[str, float]]:
        """
        설정된 기법에 따라 DOE 테이블을 생성하여 반환합니다.

        Returns
        -------
        List[Dict[str, float]]
            각 케이스별 파라미터 맵의 리스트
        """
        n_vars = len(self.variables)
        if n_vars == 0:
            return []

        if self.sampling_method == "FullFact":
            return self._generate_full_factorial()
        elif self.sampling_method == "Random":
            return self._generate_random(n_vars)
        else: # Default: LHS
            return self._generate_lhs(n_vars)

    def _generate_lhs(self, n_vars: int) -> List[Dict[str, float]]:
        """NumPy 기반의 Latin Hypercube Sampling 구현"""
        n_samples = self.sample_count
        samples = np.zeros((n_samples, n_vars))

        for d in range(n_vars):
            grid = np.linspace(0, 1, n_samples + 1)
            lower = grid[:-1]
            upper = grid[1:]
            points = np.random.uniform(lower, upper, size=n_samples)
            np.random.shuffle(points)
            samples[:, d] = points

        return self._map_samples_to_bounds(samples)

    def _generate_random(self, n_vars: int) -> List[Dict[str, float]]:
        """몬테카를로 무작위 샘플링 구현"""
        n_samples = self.sample_count
        samples = np.random.uniform(0.0, 1.0, size=(n_samples, n_vars))
        return self._map_samples_to_bounds(samples)

    def _generate_full_factorial(self) -> List[Dict[str, float]]:
        """Full Factorial 격자 샘플링 구현"""
        grids = []
        for var in self.variables:
            v_min, v_max = var["min"], var["max"]
            v_type = var.get("type", "Continuous")
            
            if v_type == "Discrete":
                step = var.get("step", (v_max - v_min) / 4.0)
                if step <= 0:
                    step = 0.1
                values = np.arange(v_min, v_max + step * 0.1, step)
            else:
                # 연속형 변수는 기본 5분할 격자 생성
                values = np.linspace(v_min, v_max, 5)
            grids.append(values)

        mesh = np.meshgrid(*grids, indexing='ij')
        flat_coords = [m.flatten() for m in mesh]
        n_samples = len(flat_coords[0])

        doe_table = []
        for idx in range(n_samples):
            case = {}
            for d_idx, var in enumerate(self.variables):
                val = float(flat_coords[d_idx][idx])
                # 이산형인 경우 다시 바운딩 처리
                if var.get("type") == "Discrete":
                    val = self._clamp_to_step(val, var["min"], var["max"], var["step"])
                case[var["name"]] = val
            doe_table.append(case)

        return doe_table

    def _map_samples_to_bounds(self, samples: np.ndarray) -> List[Dict[str, float]]:
        """[0, 1] 구간의 샘플 행렬을 실제 변수의 물리적 범위로 매핑합니다."""
        doe_table = []
        n_samples, n_vars = samples.shape

        for i in range(n_samples):
            case = {}
            for d in range(n_vars):
                var = self.variables[d]
                v_min, v_max = var["min"], var["max"]
                val = float(v_min + samples[i, d] * (v_max - v_min))

                if var.get("type") == "Discrete":
                    val = self._clamp_to_step(val, v_min, v_max, var["step"])
                case[var["name"]] = val
            doe_table.append(case)

        return doe_table

    @staticmethod
    def _clamp_to_step(val: float, v_min: float, v_max: float, step: float) -> float:
        """이산형 값의 범위 스텝화 및 한계치 보정"""
        if step <= 0:
            return val
        n_steps = round((val - v_min) / step)
        clamped = v_min + n_steps * step
        return float(np.clip(clamped, v_min, v_max))


class DOEBatchRunner:
    """
    [WHTOOLS] 각 DOE Case에 대해 시뮬레이션을 수행하고 결과를 정밀 관리하는 배치 프로세서 클래스입니다.
    """

    def __init__(self, base_config_path: str, output_base_dir: str = "doe_results"):
        """
        Parameters
        ----------
        base_config_path : str
            기준이 되는 원본 JSON 설정 파일 경로
        output_base_dir : str
            결과가 보존될 루트 디렉토리
        """
        self.base_config_path = base_config_path
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def run_doe_batch(self, doe_table: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        전체 DOE 테이블에 대해 순차적 해석을 실행하고, 핵심 요약 지표를 수집합니다.
        Gooey Progress Bar 파싱용 로그를 터미널로 실시간 스트리밍합니다.

        Parameters
        ----------
        doe_table : List[Dict[str, float]]
            DOEEngine을 통해 생성된 설계 파라미터 조합 목록

        Returns
        -------
        List[Dict[str, Any]]
            각 케이스별 설계값과 해석 응답의 통합 결과 요약 리스트
        """
        total_cases = len(doe_table)
        print(f"\n[WHTOOLS] Starting DOE Batch Run. Total Cases: {total_cases}", flush=True)

        # 원본 설정 로드
        try:
            base_config = load_config(self.base_config_path)
        except Exception as e:
            print(f"❌ Failed to load base config: {e}", flush=True)
            return []

        summary_results = []
        # Pre-allocate summary results with 'Pending' status so UI can see predefined parameters
        for idx, case_params in enumerate(doe_table):
            summary = {
                "case_id": idx + 1,
                "parameters": case_params,
                "max_ground_force": 0.0,
                "max_stress_mpa": 0.0,
                "max_displacement_mm": 0.0,
                "status": "Pending",
                "time_history": [],
                "force_history": [],
                "z_displacement_history": []
            }
            summary_results.append(summary)

        # Save initial pending state
        with open(self.output_base_dir / "doe_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=4, ensure_ascii=False)

        for idx, case_params in enumerate(doe_table):
            case_id = idx + 1
            print(f"\n==================================================", flush=True)
            print(f"🏃 Running DOE Case {case_id} / {total_cases}", flush=True)
            print(f"🔧 Parameters: {case_params}", flush=True)
            print(f"==================================================", flush=True)

            # 개별 설정 조합 생성
            case_config = base_config.copy()
            
            # target_cog 변수 조합 매핑 및 밸런싱 동기화
            cog_x = case_params.get("target_cog_x")
            cog_y = case_params.get("target_cog_y")
            cog_z = case_params.get("target_cog_z")
            if cog_x is not None or cog_y is not None or cog_z is not None:
                cb = case_config.get("components_balance", {}).copy()
                tc = list(cb.get("target_cog", [0.0, 0.0, 0.0]))
                if cog_x is not None: tc[0] = cog_x
                if cog_y is not None: tc[1] = cog_y
                if cog_z is not None: tc[2] = cog_z
                cb["target_cog"] = tc
                case_config["components_balance"] = cb

            case_config.update(case_params)
            
            # auto-balancing 물리 보정값 자동 갱신
            from run_discrete_builder.whtb_physics import analyze_and_balance_components
            case_config = analyze_and_balance_components(case_config, verbose=False)

            # UI 비활성화 설정 강제화 (Batch 안정성)
            case_config["use_viewer"] = False
            case_config["use_postprocess_ui"] = False
            case_config["batch_run_save_figures"] = False
            case_config["batch_run_show_figures"] = False

            # 결과 폴더 개별 격리 지정
            case_dir = self.output_base_dir / f"case_{case_id:04d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            case_config["result_base_dir"] = str(case_dir)

            try:
                # 시뮬레이터 인스턴스화 및 실행
                sim = DropSimulator(config=case_config)
                sim.simulate()

                # 시뮬레이션 결과 구조 분석 및 후처리 연계
                time_hist = np.array(sim.time_history)
                ground_force = np.array(sim.ground_impact_hist)
                
                # 평판 어셈블리 정밀 구조 후처리 분석 (최대 VM 응력 추출 목적)
                max_vm_stress = 0.0
                max_displacement = 0.0
                
                if hasattr(sim, 'result') and sim.result is not None:
                    # mm 스케일링 사전 적용
                    scaled_result = scale_result_to_mm(sim.result)
                    
                    # MPP 엔진에 적합한 데이터 수집
                    from run_drop_simulator.whts_mapping import get_assembly_data_from_sim
                    assembly_data = get_assembly_data_from_sim(scaled_result)
                    
                    if assembly_data:
                        from run_drop_simulator.whts_multipostprocessor_engine import PlateAssemblyManager, ShellDeformationAnalyzer
                        times_mm = np.array(sim.time_history) * 1000.0 # Time axis matching mm-scale analysis
                        manager = PlateAssemblyManager(times=times_mm, sim_data=scaled_result)
                        
                        # 컴포넌트별 분석기 추가
                        for part_name, m_hist in assembly_data.items():
                            ana = ShellDeformationAnalyzer(name=part_name)
                            ana.m_raw = m_hist
                            manager.add_analyzer(ana)
                            
                        # 수치해석 실행
                        manager.run_all()
                        
                        # 최대 응력 및 변위 계산
                        for a in manager.analyzers:
                            if a.results:
                                max_vm_stress = max(max_vm_stress, float(np.max(a.results['Von-Mises [MPa]'])))
                                max_displacement = max(max_displacement, float(np.max(np.abs(a.results['Displacement [mm]']))))

                # 시계열 원본 데이터 수집
                time_s = time_hist.tolist()
                force_n = ground_force.tolist()
                
                # Z 방향 변위 추출 (Box Center 기준 또는 Corner Center 기준)
                z_disp_m = []
                if len(sim.corner_pos_hist) > 0:
                    c_pos = np.array(sim.corner_pos_hist) # (N, 8, 3)
                    z_disp_m = (c_pos[:, :, 2].mean(axis=1) - c_pos[0, :, 2].mean()).tolist()
                else:
                    z_disp_m = [0.0] * len(time_s)

                # 요약 정보 취합
                summary_results[idx].update({
                    "max_ground_force": float(np.max(ground_force)) if len(ground_force) > 0 else 0.0,
                    "max_stress_mpa": max_vm_stress,
                    "max_displacement_mm": max_displacement,
                    "status": "Success",
                    "time_history": time_s,
                    "force_history": force_n,
                    "z_displacement_history": z_disp_m
                })

                # 저장 파일 출력
                self._save_case_data(case_dir, case_config, summary_results[idx])

            except Exception as e:
                print(f"❌ Error in DOE Case {case_id}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                
                summary_results[idx].update({
                    "status": f"Failed: {str(e)}"
                })
                # 에러 발생 시 기본 설정과 상태 저장
                self._save_case_data(case_dir, case_config, summary_results[idx])

            # 매 케이스 완료 시점마다 통합 요약본 갱신 (Incremental Save)
            with open(self.output_base_dir / "doe_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_results, f, indent=4, ensure_ascii=False)

            # [Gooey Progress Parser 연동] 터미널 진행 상태 출력 (표준 규격)
            progress_percent = (case_id / total_cases) * 100.0
            print(f"\nprogress: {progress_percent:.1f}%", flush=True)

        print(f"\n🎉 DOE Batch Execution Finished. Results saved in: {self.output_base_dir}", flush=True)
        return summary_results

    def _save_case_data(self, case_dir: Path, config: Dict[str, Any], summary: Dict[str, Any]):
        """개별 케이스 결과 데이터 디스크 쓰기"""
        # config.json 저장
        save_config(config, case_dir / "case_config.json")
        
        # summary 및 궤적 데이터 저장 (JSON/Pickle 이중화 지원으로 유연성 유지)
        with open(case_dir / "case_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            
        with open(case_dir / "sim_result.pkl", "wb") as f:
            pickle.dump(summary, f)


class OptimizationEvaluator:
    """
    [WHTOOLS] DOE 데이터베이스를 조회하여 제약 조건 필터링 및 목적 함수 기준 최적안을 추출하는 최적화 모듈입니다.
    """

    def __init__(self, summary_data: List[Dict[str, Any]]):
        self.summary_data = summary_data

    def evaluate_best_case(self, constraints: Dict[str, float], objective: str = "min_force") -> Optional[Dict[str, Any]]:
        """
        주어진 설계 제약 및 목적 사항에 맞춰 최적의 케이스를 추출합니다.

        Parameters
        ----------
        constraints : Dict[str, float]
            제약 조건 맵.
            예: {"max_displacement_mm": 15.0, "max_stress_mpa": 200.0}
        objective : str
            목적 함수 모드 ("min_force", "min_stress", "min_disp")

        Returns
        -------
        Optional[Dict[str, Any]]
            최적 Case의 요약 정보 Dictionary (없으면 None)
        """
        valid_cases = []

        for case in self.summary_data:
            if case.get("status") != "Success":
                continue

            # 제약 조건 검증
            violated = False
            for c_key, c_val in constraints.items():
                if c_key in case:
                    # Max 값 판정
                    if case[c_key] > c_val:
                        violated = True
                        break

            if not violated:
                valid_cases.append(case)

        if not valid_cases:
            return None

        # 목적에 따른 정렬
        if objective == "min_force":
            valid_cases.sort(key=lambda x: x.get("max_ground_force", float('inf')))
        elif objective == "min_stress":
            valid_cases.sort(key=lambda x: x.get("max_stress_mpa", float('inf')))
        elif objective == "min_disp":
            valid_cases.sort(key=lambda x: x.get("max_displacement_mm", float('inf')))

        return valid_cases[0]
