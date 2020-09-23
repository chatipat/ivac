import numba as nb
import numpy as np
import pytest


@nb.njit
def ou_process(n, dt):
    """Simulate an OU process.

    Simulates the OU process

    .. math::

        dX_t = -X_t dt + \sqrt{2} dW_t

    Parameters
    ----------
    n : int
        Length of trajectory.
    dt : float
        Sampling time.

    Returns
    -------
    (n,) ndarray
        Trajectory.

    """
    a = np.exp(-dt)
    b = np.sqrt(1.0 - np.exp(-2.0 * dt))
    traj = np.empty(n)
    traj[0] = np.random.normal()
    for i in range(1, n):
        traj[i] = np.random.normal(a * traj[i - 1], b)
    return traj


def indicator_basis(traj, bins):
    """Compute an indicator basis on a 1d trajectory.

    Parameters
    ----------
    traj : (n,) array-like
        Trajectory.
    bins : int or (nbins-1,) array-like
        If an integer is passed, constructs this number of uniform bins.
        If an array is passed, these are treated as the bin edges.

    Returns
    -------
    (n, nbins) array-like
        Featurized trajectory.

    """
    traj = np.asarray(traj)
    bins = np.asarray(bins)
    if bins.ndim == 0:
        nbins = bins
        bins = np.linspace(np.min(traj), np.max(traj), bins + 1)[1:-1]
        assert len(bins) + 1 == nbins
    nbins = len(bins) + 1
    ix = np.searchsorted(bins, traj)
    basis = np.zeros((len(traj), nbins))
    basis[np.arange(len(traj)), ix] = 1.0

    # sanity checks
    assert ix.shape == traj.shape
    assert np.all(ix >= 0) and np.all(ix < nbins)
    assert np.all(np.sum(basis, axis=-1) == 1.0)

    return basis
