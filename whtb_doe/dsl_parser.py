"""
whtb_doe DSL Parser — Composite approach:
- LHS (key path): ast.parse() for Subscript chain extraction
- RHS (value): ast.literal_eval() first, then colon-regex fallback
"""
import ast
import re
import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class VarSpec:
    key_path: List[str]      # ['a', 'b']
    var_name: str            # 'a.b' (DOE table column name)
    var_type: str            # 'discrete_list','linspace','continuous','normal','string_list'
    values: Any              # type-specific params
    original_value: Any      # value in base_config


def _extract_key_path(lhs: str) -> Optional[List[str]]:
    """Extract key path from cfg['a']['b'] using AST."""
    try:
        tree = ast.parse(lhs.strip() + " = None", mode='exec')
        assign = tree.body[0]
        target = assign.targets[0]
        path = []
        node = target
        while isinstance(node, ast.Subscript):
            # Get the key
            if isinstance(node.slice, ast.Constant):
                path.append(str(node.slice.value))
            elif isinstance(node.slice, ast.Index):  # Python 3.8 compat
                val = node.slice.value
                if isinstance(val, ast.Constant):
                    path.append(str(val.value))
            node = node.value
        path.reverse()
        return path if path else None
    except Exception:
        return None


def _get_nested(d: dict, key_path: List[str]) -> Any:
    cur = d
    for k in key_path:
        if isinstance(cur, list) and k.isdigit():
            idx = int(k)
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        elif isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _parse_rhs(rhs: str) -> Tuple[str, Any]:
    """Parse the right-hand side of a DSL assignment.
    Returns (var_type, values).
    """
    rhs = rhs.strip()

    # 1. Try ast.literal_eval first (handles [10,20,30], ['A','B'], numbers)
    try:
        val = ast.literal_eval(rhs)
        if isinstance(val, list):
            if all(isinstance(v, str) for v in val):
                return 'string_list', val
            return 'discrete_list', val
        if isinstance(val, (int, float)):
            return 'continuous', {'min': val, 'max': val, 'init': val}
        return 'discrete_list', [val]
    except Exception:
        pass

    # 2. [start:end:count] — linspace with brackets
    m = re.match(r'^\[\s*(-?[\d.]+)\s*:\s*(-?[\d.]+)\s*:\s*(\d+)\s*\]$', rhs)
    if m:
        return 'linspace', {
            'start': float(m.group(1)),
            'end':   float(m.group(2)),
            'count': int(m.group(3)),
        }

    # 3. start:end:count or min:max:init (3 colon-separated numbers)
    m = re.match(r'^(-?[\d.]+)\s*:\s*(-?[\d.]+)\s*:\s*(-?[\d.]+)$', rhs)
    if m:
        a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return 'continuous', {'min': a, 'max': b, 'init': c}

    # 4. norm:mean:std[:n]
    m = re.match(r'^norm\s*:\s*(-?[\d.]+)\s*:\s*(-?[\d.]+)(?:\s*:\s*(\d+))?$', rhs, re.IGNORECASE)
    if m:
        return 'normal', {
            'mean': float(m.group(1)),
            'std':  float(m.group(2)),
            'n':    int(m.group(3)) if m.group(3) else None,
        }

    raise ValueError(f"Cannot parse RHS: {rhs!r}")


def parse_dsl(dsl_string: str, base_config: dict) -> List[VarSpec]:
    """Parse a DSL string and return list of VarSpec."""
    specs = []
    for lineno, line in enumerate(dsl_string.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise ValueError(f"Line {lineno}: no '=' found: {line!r}")
        eq_idx = line.index('=')
        lhs = line[:eq_idx].strip()
        rhs = line[eq_idx+1:].strip()

        key_path = _extract_key_path(lhs)
        if key_path is None:
            raise ValueError(f"Line {lineno}: cannot parse key path from: {lhs!r}")

        var_name = '.'.join(key_path)
        original = _get_nested(base_config, key_path)

        var_type, values = _parse_rhs(rhs)

        specs.append(VarSpec(
            key_path=key_path,
            var_name=var_name,
            var_type=var_type,
            values=values,
            original_value=original,
        ))
    return specs


def validate_dsl(dsl_string: str, base_config: dict) -> Tuple[bool, str]:
    """Validate DSL string. Returns (success, message)."""
    try:
        specs = parse_dsl(dsl_string, base_config)
    except ValueError as e:
        return False, str(e)

    errors = []
    total_fullfact = 1
    for spec in specs:
        orig = _get_nested(base_config, spec.key_path)
        if orig is None:
            errors.append(f"Key path {spec.var_name!r} not found in base_config")

        if spec.var_type == 'discrete_list':
            total_fullfact *= len(spec.values)
        elif spec.var_type == 'string_list':
            total_fullfact *= len(spec.values)
        elif spec.var_type == 'linspace':
            total_fullfact *= spec.values['count']
        elif spec.var_type == 'continuous':
            total_fullfact *= 5  # default 5 levels
        elif spec.var_type == 'normal':
            n = spec.values.get('n') or 10
            total_fullfact *= n

    if errors:
        return False, '\n'.join(errors)

    if total_fullfact > 1000:
        return True, f"Warning: FullFact would generate {total_fullfact} cases (>1000). Proceed with LHS/MonteCarlo."

    return True, f"OK — {len(specs)} variable(s) defined. FullFact estimate: {total_fullfact} cases."
