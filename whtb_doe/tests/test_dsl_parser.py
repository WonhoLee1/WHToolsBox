import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from whtb_doe.dsl_parser import parse_dsl, validate_dsl, VarSpec

BASE_CFG = {'a': {'b': 10, 'f': 50}, 'd': 100, 's': 'foo'}

def test_discrete_list():
    specs = parse_dsl("cfg['a']['b'] = [10, 20, 30]", BASE_CFG)
    assert len(specs) == 1
    assert specs[0].var_type == 'discrete_list'
    assert specs[0].values == [10, 20, 30]
    assert specs[0].var_name == 'a.b'

def test_linspace_bracket():
    specs = parse_dsl("cfg['a']['f'] = [10:199:15]", BASE_CFG)
    assert specs[0].var_type == 'linspace'
    assert specs[0].values['count'] == 15

def test_continuous():
    specs = parse_dsl("cfg['d'] = 10:200:100", BASE_CFG)
    assert specs[0].var_type == 'continuous'
    assert specs[0].values['min'] == 10
    assert specs[0].values['max'] == 200

def test_normal():
    specs = parse_dsl("cfg['d'] = norm:10.0:2.0:50", BASE_CFG)
    assert specs[0].var_type == 'normal'
    assert specs[0].values['mean'] == 10.0
    assert specs[0].values['n'] == 50

def test_string_list():
    specs = parse_dsl("cfg['s'] = ['A', 'B', 'C']", BASE_CFG)
    assert specs[0].var_type == 'string_list'
    assert specs[0].values == ['A', 'B', 'C']

def test_validate_ok():
    ok, msg = validate_dsl("cfg['a']['b'] = [10, 20, 30]", BASE_CFG)
    assert ok, msg

def test_validate_bad_key():
    ok, msg = validate_dsl("cfg['x']['missing'] = [1, 2]", BASE_CFG)
    assert not ok

if __name__ == '__main__':
    test_discrete_list()
    test_linspace_bracket()
    test_continuous()
    test_normal()
    test_string_list()
    test_validate_ok()
    test_validate_bad_key()
    print("All tests passed.")
