from visible_curvature.aggregate import consensus_label


def test_consensus_label_rejects_positive_negative_seed_conflict():
    assert consensus_label(["positive", "negative"], minimum_seed_count=2) == (
        "inconclusive",
        "seed_sign_conflict",
    )


def test_consensus_label_requires_minimum_seed_count():
    assert consensus_label(["positive", "positive"], minimum_seed_count=3) == (
        "inconclusive",
        "insufficient_seed_count",
    )


def test_consensus_label_accepts_unanimous_signed_seeds():
    assert consensus_label(["positive"] * 3, minimum_seed_count=3) == ("positive", "")
    assert consensus_label(["negative"] * 3, minimum_seed_count=3) == ("negative", "")
