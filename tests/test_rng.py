import numpy as np

from vessel_state_estimation.rng import STREAMS, make_rngs


def test_same_seed_same_draws():
    a, b = make_rngs(7), make_rngs(7)
    for name in STREAMS:
        assert np.array_equal(a[name].normal(size=10), b[name].normal(size=10))


def test_streams_are_independent():
    r = make_rngs(7)
    assert not np.array_equal(r["imu"].normal(size=10), r["gps"].normal(size=10))
