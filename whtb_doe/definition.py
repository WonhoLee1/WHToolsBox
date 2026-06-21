"""DOEDefinition — main API class for whtb_doe."""
import copy
from typing import Dict, List, Optional, Tuple

import pandas as pd

from whtb_doe.dsl_parser import VarSpec, parse_dsl, validate_dsl
from whtb_doe.sampler import DOESampler


def _set_nested(d: dict, key_path: List[str], value) -> dict:
    cur = d
    for k in key_path[:-1]:
        if isinstance(cur, list) and k.isdigit():
            cur = cur[int(k)]
        else:
            cur = cur[k]
            
    last_k = key_path[-1]
    if isinstance(cur, list) and last_k.isdigit():
        cur[int(last_k)] = value
    else:
        cur[last_k] = value
    return d


class DOEDefinition:
    def __init__(self,
                 dsl_string: str,
                 base_config: dict,
                 output_dir: str = None):
        self.dsl_string = dsl_string
        self.base_config = base_config
        self.output_dir = output_dir
        self._var_specs: Optional[List[VarSpec]] = None

    def _get_specs(self) -> List[VarSpec]:
        if self._var_specs is None:
            self._var_specs = parse_dsl(self.dsl_string, self.base_config)
        return self._var_specs

    def validate(self) -> Tuple[bool, str]:
        return validate_dsl(self.dsl_string, self.base_config)

    def generate(self,
                 method: str = 'lhs',
                 n_samples: int = 100,
                 seed: int = 42) -> Tuple[pd.DataFrame, List[dict]]:
        ok, msg = self.validate()
        if not ok:
            raise ValueError(f"DSL validation failed: {msg}")

        specs = self._get_specs()
        sampler = DOESampler()
        doe_table = sampler.sample(specs, method=method, n_samples=n_samples, seed=seed)

        config_list = self._table_to_configs(doe_table, specs)
        return doe_table, config_list

    def regenerate(self,
                   modified_doe_table: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
        """Use a modified DOE table directly (no re-sampling)."""
        specs = self._get_specs()
        config_list = self._table_to_configs(modified_doe_table, specs)
        return modified_doe_table, config_list

    def _table_to_configs(self,
                          doe_table: pd.DataFrame,
                          specs: List[VarSpec]) -> List[dict]:
        configs = []
        for _, row in doe_table.iterrows():
            cfg = copy.deepcopy(self.base_config)
            for spec in specs:
                val = row[spec.var_name]
                _set_nested(cfg, spec.key_path, val)
            configs.append(cfg)
        return configs

    def set_output_dir(self, output_dir: str):
        self.output_dir = output_dir
