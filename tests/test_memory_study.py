import numpy as np

from analysis.memory_study import crossed_interval, exact_sign_flip_pvalue, paired_deal_diffs
from analysis.robustness_study import attack_edge


def test_paired_deal_diffs_average_both_seatings():
    match = {
        "records": [
            {"deal": 0, "score_a": 10, "score_b": 4},
            {"deal": 0, "score_a": 3, "score_b": 5},
            {"deal": 1, "score_a": 8, "score_b": 8},
            {"deal": 1, "score_a": 7, "score_b": 3},
        ]
    }
    assert np.array_equal(paired_deal_diffs(match), np.array([2.0, 2.0]))


def test_exact_sign_flip_pvalue_uses_training_seed_effects():
    # With three equal positive effects, two of eight sign assignments are
    # at least as extreme as the observed absolute mean.
    assert exact_sign_flip_pvalue(np.ones(3)) == 0.25


def test_crossed_interval_resamples_shared_deal_axis():
    interval = crossed_interval([np.ones(4), np.ones(4)], n_boot=100)
    assert interval == (1.0, 1.0)


def test_attack_edge_uses_unrounded_mirrored_records():
    result = {
        "records": [
            {"deal": 0, "score_a": 4, "score_b": 1},
            {"deal": 0, "score_a": 2, "score_b": 3},
            {"deal": 1, "score_a": 5, "score_b": 1},
            {"deal": 1, "score_a": 0, "score_b": 2},
        ]
    }
    assert attack_edge(result) == 1.0
