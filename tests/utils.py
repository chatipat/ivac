import numpy as np


def allclose_sign(a, b):
    """True is all values are close, up to the same sign for all values."""
    return np.allclose(a, b) or np.allclose(a, -b)


def allclose_trajs(trajs1, trajs2):
    """Check that all values of the trajectories are close."""
    trajs1 = np.concatenate(trajs1, axis=0)
    trajs2 = np.concatenate(trajs2, axis=0)
    return np.allclose(trajs1, trajs2)


def allclose_trajs_sign(trajs1, trajs2):
    """Check that all values of the trajectories are close up to sign."""
    trajs1 = np.concatenate(trajs1, axis=0)
    trajs2 = np.concatenate(trajs2, axis=0)
    assert trajs1.shape == trajs2.shape
    return np.all(
        np.logical_or(
            np.all(np.isclose(trajs1, trajs2), axis=0),
            np.all(np.isclose(trajs1, -trajs2), axis=0),
        )
    )
