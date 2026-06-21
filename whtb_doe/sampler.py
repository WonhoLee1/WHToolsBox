"""DOE Sampler — LHS, FullFact, MonteCarlo using scipy.stats.qmc"""
import copy
import itertools
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from whtb_doe.dsl_parser import VarSpec


def _get_levels(spec: VarSpec) -> List[Any]:
    """Get discrete levels for a variable spec."""
    if spec.var_type in ('discrete_list', 'string_list'):
        return spec.values
    elif spec.var_type == 'linspace':
        v = spec.values
        return list(np.linspace(v['start'], v['end'], v['count']))
    elif spec.var_type == 'continuous':
        v = spec.values
        return list(np.linspace(v['min'], v['max'], 5))
    elif spec.var_type == 'normal':
        v = spec.values
        n = v.get('n') or 10
        return list(np.linspace(v['mean'] - 3*v['std'], v['mean'] + 3*v['std'], n))
    return []


def _scale_unit_sample(unit: float, spec: VarSpec) -> Any:
    """Scale a unit [0,1] sample to the variable's domain."""
    if spec.var_type in ('discrete_list', 'string_list'):
        levels = spec.values
        idx = min(int(unit * len(levels)), len(levels) - 1)
        return levels[idx]
    elif spec.var_type == 'linspace':
        v = spec.values
        return v['start'] + unit * (v['end'] - v['start'])
    elif spec.var_type == 'continuous':
        v = spec.values
        return v['min'] + unit * (v['max'] - v['min'])
    elif spec.var_type == 'normal':
        from scipy.stats import norm
        v = spec.values
        u = np.clip(unit, 1e-6, 1 - 1e-6)
        return float(norm.ppf(u, loc=v['mean'], scale=v['std']))
    return unit


class DOESampler:
    def sample(self,
               var_specs: List[VarSpec],
               method: str = 'lhs',
               n_samples: int = 100,
               seed: int = 42) -> pd.DataFrame:

        if not var_specs:
            return pd.DataFrame()

        method = method.lower()
        if method == 'fullfact':
            return self._fullfact(var_specs)
        elif method == 'montecarlo':
            return self._montecarlo(var_specs, n_samples, seed)
        else:  # lhs default
            return self._lhs(var_specs, n_samples, seed)

    def _lhs(self, var_specs: List[VarSpec], n: int, seed: int) -> pd.DataFrame:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=len(var_specs), seed=seed)
        unit_samples = sampler.random(n)  # shape (n, d)
        rows = []
        for i, units in enumerate(unit_samples):
            row = {'case_number': i}
            for j, spec in enumerate(var_specs):
                row[spec.var_name] = _scale_unit_sample(units[j], spec)
            rows.append(row)
        return pd.DataFrame(rows)

    def _montecarlo(self, var_specs: List[VarSpec], n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        unit_samples = rng.uniform(0, 1, size=(n, len(var_specs)))
        rows = []
        for i, units in enumerate(unit_samples):
            row = {'case_number': i}
            for j, spec in enumerate(var_specs):
                row[spec.var_name] = _scale_unit_sample(units[j], spec)
            rows.append(row)
        return pd.DataFrame(rows)

    def _fullfact(self, var_specs: List[VarSpec]) -> pd.DataFrame:
        level_lists = [_get_levels(spec) for spec in var_specs]
        rows = []
        for i, combo in enumerate(itertools.product(*level_lists)):
            row = {'case_number': i}
            for spec, val in zip(var_specs, combo):
                row[spec.var_name] = val
            rows.append(row)
        return pd.DataFrame(rows)
