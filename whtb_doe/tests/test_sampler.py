import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from whtb_doe import DOEDefinition

BASE_CFG = {'a': {'b': 10, 'f': 50}, 'd': 100}

DSL = """
cfg['a']['b'] = [10, 20, 30]
cfg['d'] = 10:200:100
"""

def test_lhs():
    doe = DOEDefinition(DSL, BASE_CFG)
    ok, msg = doe.validate()
    assert ok, msg
    table, configs = doe.generate(method='lhs', n_samples=5, seed=42)
    assert len(table) == 5
    assert len(configs) == 5
    assert 'a.b' in table.columns
    assert 'd' in table.columns

def test_fullfact():
    dsl = "cfg['a']['b'] = [10, 20, 30]\ncfg['a']['f'] = [10:50:3]"
    doe = DOEDefinition(dsl, BASE_CFG)
    table, configs = doe.generate(method='fullfact')
    assert len(table) == 9  # 3 * 3
    assert len(configs) == 9

def test_regenerate():
    doe = DOEDefinition(DSL, BASE_CFG)
    table, configs = doe.generate(method='lhs', n_samples=3, seed=0)
    table.loc[0, 'a.b'] = 99
    _, new_configs = doe.regenerate(table)
    assert new_configs[0]['a']['b'] == 99

if __name__ == '__main__':
    test_lhs()
    test_fullfact()
    test_regenerate()
    print("All sampler tests passed.")
