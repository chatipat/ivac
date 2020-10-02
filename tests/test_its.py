"""Test calculation of implied timescales."""


import numpy as np
from ivac.linear import _vac_its as vac_its
from ivac.linear import _ivac_its as ivac_its


def vac_eval(sigmas, lag):
    return np.exp(-sigmas * lag)


def ivac_eval(sigmas, minlag, maxlag, lagstep):
    assert (maxlag - minlag) % lagstep == 0
    lags = np.arange(minlag, maxlag + 1, lagstep)
    return np.sum(np.exp(-np.outer(sigmas, lags)), axis=-1)


def test_vac_its():
    lags = np.unique(np.rint(np.logspace(0, 4, 100)).astype(int))
    sigmas = np.logspace(-4, 1, 100)
    ref_its = 1.0 / sigmas
    assert np.all(ref_its > 0.0)
    assert not np.any(np.isnan(ref_its))
    for lag in lags:
        evals = vac_eval(sigmas, lag)
        assert np.all(evals >= 0.0)
        assert np.all(evals <= 1.0)
        test_its = vac_its(evals, lag)
        mask = np.logical_not(np.isnan(test_its))
        assert np.allclose(test_its[mask], ref_its[mask])
        assert np.all(np.isnan(test_its[np.logical_not(mask)]))
        length = len(test_its[mask])
        assert np.all(np.logical_not(np.isnan(test_its[:length])))
        assert np.all(np.isnan(test_its[length:]))


def test_ivac_its():
    sigmas = np.logspace(-4, 1, 100)
    ref_its = 1.0 / sigmas
    minlags = np.unique(np.rint(np.logspace(0, 4, 100)).astype(int))
    lagsteps = np.unique(np.rint(np.logspace(0, 3, 100)).astype(int))
    nlags = np.arange(100)
    assert np.all(ref_its > 0.0)
    assert not np.any(np.isnan(ref_its))
    for _ in range(100):
        minlag = np.random.choice(minlags)
        lagstep = np.random.choice(lagsteps)
        maxlag = minlag + lagstep * np.random.choice(nlags)
        numlags = (maxlag - minlag) % lagstep + 1
        evals = ivac_eval(sigmas, minlag, maxlag, lagstep)
        assert np.all(evals >= 0.0)
        assert np.all(evals <= ((maxlag - minlag) // lagstep + 1))
        test_its = ivac_its(evals, minlag, maxlag, lagstep)
        mask = np.logical_not(np.isnan(test_its))
        assert np.allclose(test_its[mask], ref_its[mask], rtol=1e-3)
        assert np.all(np.isnan(test_its[np.logical_not(mask)]))
        length = len(test_its[mask])
        assert np.all(np.logical_not(np.isnan(test_its[:length])))
        assert np.all(np.isnan(test_its[length:]))
