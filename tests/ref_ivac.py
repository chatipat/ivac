"""Reference implementations for linear VAC and IVAC."""

import numpy as np
import ref_corr
from scipy import linalg


def vac_all(trajs, lag):
    ct = ref_corr.ct_all(trajs, lag)
    c0 = ref_corr.c0_all(trajs)
    return symeig(ct, c0)


def vac_all_adj(trajs, lag):
    ct = ref_corr.ct_all(trajs, lag)
    c0 = ref_corr.c0_all_adj_ct(trajs, lag)
    return symeig(ct, c0)


def ivac_all(trajs, lags):
    ic = ref_corr.ic_all(trajs, lags)
    c0 = ref_corr.c0_all(trajs)
    return symeig(ic, c0)


def ivac_all_adj(trajs, lags):
    ic = ref_corr.ic_all(trajs, lags)
    c0 = ref_corr.c0_all_adj_ic(trajs, lags)
    return symeig(ic, c0)


def vac_weights(trajs, lag, cutlag):
    ct = ref_corr.ct_trunc(trajs, lag, cutlag)
    c0 = ref_corr.c0_trunc(trajs, cutlag)
    coeffs = solve_stationary(ct, c0)
    assert np.allclose(coeffs @ ct, coeffs @ c0)
    return build_weights(trajs, coeffs, cutlag)


def ivac_weights(trajs, lags, cutlag):
    ic = ref_corr.ic_trunc(trajs, lags, cutlag) / len(lags)
    c0 = ref_corr.c0_trunc(trajs, cutlag)
    coeffs = solve_stationary(ic, c0)
    assert np.allclose(coeffs @ ic, coeffs @ c0)
    return build_weights(trajs, coeffs, cutlag)


def vac_rt(trajs, lag, cutlag, weights):
    ct = ref_corr.ct_rt(trajs, lag, cutlag, weights)
    c0 = ref_corr.c0_rt(trajs, cutlag, weights)
    return symeig(ct, c0)


def vac_rt_adj(trajs, lag, cutlag, weights):
    ct = ref_corr.ct_rt(trajs, lag, cutlag, weights)
    c0 = ref_corr.c0_rt_adj_ct(trajs, lag, cutlag, weights)
    return symeig(ct, c0)


def ivac_rt(trajs, lags, cutlag, weights):
    ic = ref_corr.ic_rt(trajs, lags, cutlag, weights)
    c0 = ref_corr.c0_rt(trajs, cutlag, weights)
    return symeig(ic, c0)


def ivac_rt_adj(trajs, lags, cutlag, weights):
    ic = ref_corr.ic_rt(trajs, lags, cutlag, weights)
    c0 = ref_corr.c0_rt_adj_ic(trajs, lags, cutlag, weights)
    return symeig(ic, c0)


def symeig(a, b):
    a = 0.5 * (a + a.T)
    b = 0.5 * (b + b.T)
    evals, evecs = linalg.eigh(a, b)
    return evals[::-1], evecs[:, ::-1]


def solve_stationary(a, b):
    evals, evecs = linalg.eig(a.T, b)
    order = np.argsort(np.abs(evals))[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    assert np.isclose(evals[0], 1.0)
    assert not np.any(np.isclose(evals[1:], 1.0))
    assert np.all(np.abs(evals[1:]) < 1.0)
    coeffs = np.real_if_close(evecs[:, 0])
    assert np.isrealobj(coeffs)
    return coeffs


def build_weights(trajs, coeffs, cutlag):
    weights = []
    for traj in trajs:
        weight = traj @ coeffs
        weight[len(weight) - cutlag :] = 0.0
        weights.append(weight)
    return weights
