from model_registry.registry import PARAM_SETS

def test_five_candidate_parameter_sets():
    assert len(PARAM_SETS) == 5
    assert len({tuple(sorted(p.items())) for p in PARAM_SETS}) == 5
