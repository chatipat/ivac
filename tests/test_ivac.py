"""Test linear IVAC solver against a reference implementation."""

import data
import numpy as np
import ref_ivac as ref
import utils

from ivac.linear import _ivac_weights as ivac_weights
from ivac.linear import _solve_ivac as solve_ivac


def ref_is_close(test_output, ref_output):
    test_evals, test_evecs = test_output
    ref_evals, ref_evecs = ref_output
    assert np.allclose(test_evals, ref_evals)
    assert utils.allclose_sign(ref_evecs, ref_evecs)


def result_is_close(test_output, ref_output):
    test_c0, test_evals, test_evecs = test_output
    ref_evals, ref_evecs = ref_output
    assert np.allclose(test_evals, ref_evals)
    assert utils.allclose_sign(ref_evecs, ref_evecs)


def test_solve():
    trajs = [
        data.indicator_basis(data.ou_process(n, 0.1), 10)
        for n in [1000, 1500, 2000]
    ]
    params = [
        np.arange(1, 10),
        np.arange(5, 10),
        np.arange(1, 20),
        np.arange(1, 20, 2),
        np.arange(2, 20, 3),
    ]
    check(trajs, params)


def check(trajs, params):

    vac_params = np.unique(np.concatenate(params))
    ivac_params = params

    # equilibrium

    for method in ["direct", "fft"]:
        for lag in vac_params:
            result_is_close(
                solve_ivac(trajs, lag, adjust=False, method=method),
                ref.vac_all(trajs, lag),
            )
            result_is_close(
                solve_ivac(trajs, lag, adjust=True, method=method),
                ref.vac_all_adj(trajs, lag),
            )
        for lags in ivac_params:
            result_is_close(
                solve_ivac(trajs, lags, adjust=False, method=method),
                ref.ivac_all(trajs, lags),
            )
            result_is_close(
                solve_ivac(trajs, lags, adjust=True, method=method),
                ref.ivac_all_adj(trajs, lags),
            )

    # nonequilibrium

    for method in ["direct", "fft"]:
        for lag in vac_params:
            cutlag = lag
            ref_weights = ref.vac_weights(trajs, lag, cutlag)
            test_weights = ivac_weights(
                trajs, lag, weights=cutlag, method=method
            )
            check_weights(test_weights, ref_weights)
            for weights in [ref_weights, test_weights]:
                result_is_close(
                    solve_ivac(
                        trajs,
                        lag,
                        weights=weights,
                        adjust=False,
                        method=method,
                    ),
                    ref.vac_rt(trajs, lag, cutlag, weights),
                )
                result_is_close(
                    solve_ivac(
                        trajs,
                        lag,
                        weights=weights,
                        adjust=True,
                        method=method,
                    ),
                    ref.vac_rt_adj(trajs, lag, cutlag, weights),
                )
            ref_is_close(
                ref.vac_rt(trajs, lag, cutlag, test_weights),
                ref.vac_rt(trajs, lag, cutlag, ref_weights),
            )
            ref_is_close(
                ref.vac_rt_adj(trajs, lag, cutlag, test_weights),
                ref.vac_rt_adj(trajs, lag, cutlag, ref_weights),
            )

        for lags in ivac_params:
            cutlag = np.max(lags)
            ref_weights = ref.ivac_weights(trajs, lags, cutlag)
            test_weights = ivac_weights(
                trajs, lags, weights=cutlag, method=method
            )
            check_weights(test_weights, ref_weights)
            for weight in [ref_weights, test_weights]:
                result_is_close(
                    solve_ivac(
                        trajs,
                        lags,
                        weights=weights,
                        adjust=False,
                        method=method,
                    ),
                    ref.ivac_rt(trajs, lags, cutlag, weights),
                )
                result_is_close(
                    solve_ivac(
                        trajs,
                        lags,
                        weights=weights,
                        adjust=True,
                        method=method,
                    ),
                    ref.ivac_rt_adj(trajs, lags, cutlag, weights),
                )
            ref_is_close(
                ref.ivac_rt(trajs, lags, cutlag, test_weights),
                ref.ivac_rt(trajs, lags, cutlag, ref_weights),
            )
            ref_is_close(
                ref.ivac_rt_adj(trajs, lags, cutlag, test_weights),
                ref.ivac_rt_adj(trajs, lags, cutlag, ref_weights),
            )


def check_weights(wtest, wref):
    assert len(wtest) == len(wref)
    for wt, wr in zip(wtest, wref):
        assert wt.shape == wr.shape
        assert wt.ndim == 1
        wt = wt / np.sum(wt)
        wr = wr / np.sum(wr)
        assert utils.allclose_sign(wt, wr)
