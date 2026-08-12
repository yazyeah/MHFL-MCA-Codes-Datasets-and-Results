from mhfl_review.specs import branch_semantics, manuscript_spec, parameter_count_m


def test_uo_parameter_count_and_order():
    spec = manuscript_spec("uo")
    assert abs(parameter_count_m(spec) - 5.111759) < 1e-6
    assert branch_semantics(spec) == ("vibration_query", "other_query")


def test_kaist_candidate_and_order():
    spec = manuscript_spec("kaist")
    assert abs(parameter_count_m(spec) - 7.380173) < 1e-6
    assert branch_semantics(spec) == ("vibration_source", "other_source")


def test_direct_softmax_has_fewer_parameters():
    base = manuscript_spec("kaist")
    direct = base.updated(gate_mode="direct_softmax", expected_params_m=None)
    assert parameter_count_m(direct) < parameter_count_m(base)
