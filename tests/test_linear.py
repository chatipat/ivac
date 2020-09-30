import numpy as np
from data import indicator_basis, ou_process
from utils import allclose_sign, allclose_trajs, allclose_trajs_sign
import pytest

import ref_corr
import ivac


def make_data(ns, dt, nbins):
    """Generate featurized data for testing."""
    return [indicator_basis(ou_process(n, dt), nbins) for n in ns]


def sym(mat):
    """Symmetrize matrix."""
    return 0.5 * (mat + mat.T)


def fit_is_close(test, ref, nevecs=None):
    """Check that fit results are similar."""

    assert np.allclose(test.evals[:nevecs], ref.evals[:nevecs])
    assert allclose_sign(test.evecs[:, :nevecs], ref.evecs[:, :nevecs])
    assert np.allclose(test.its[:nevecs], ref.its[:nevecs])

    # ignore c0 if not saved
    if test.cov is not None and ref.cov is not None:
        assert np.allclose(test.cov, ref.cov)


def result_is_close(test, ref, trajs_test, trajs_ref=None, nevecs=None):
    """Check that eigenvalues and transform results are similar."""

    if trajs_ref is None:
        trajs_ref = trajs_test

    assert np.allclose(test.evals[:nevecs], ref.evals[:nevecs])
    assert np.allclose(test.its[:nevecs], ref.its[:nevecs])

    test_evecs = test.transform(trajs_test)
    ref_evecs = ref.transform(trajs_ref)

    # check that evecs have the right shape
    assert all(evec.ndim == 2 for evec in test_evecs)
    assert all(evec.ndim == 2 for evec in ref_evecs)
    assert all(evec.shape[-1] == nevecs for evec in test_evecs)
    assert all(evec.shape[-1] == nevecs for evec in ref_evecs)

    assert allclose_trajs_sign(test_evecs, ref_evecs)


def result_is_orthogonal(test):
    """Check that training trajectories yield orthogonal eigenvectors.

    For adjust=False only.
    """
    evecs = test.transform(test.trajs)
    numer = []
    denom = 0.0
    if test.weights is None:
        for evec in evecs:
            numer.append(evec.T @ evec)
            denom += len(evec)
    else:
        for evec, weight in zip(evecs, test.weights):
            numer.append(evec.T @ (weight[:, None] * evec))
            denom += np.sum(weight)
    corr = np.sum(numer, axis=0) / denom
    assert np.allclose(corr, np.identity(len(corr)))


def result_has_ones(test):
    """Check that the trivial eigenvector consists of ones.

    For adjust=True only.
    """
    evecs = test.transform(test.trajs)
    if evecs[0][0, 0] > 0.0:
        sign = 1.0
    else:
        sign = -1.0
    for evec in evecs:
        assert np.allclose(sign * evec[:, 0], 1.0)


def test_vac():
    """Tests for LinearVAC."""

    trajs = make_data([1000, 1500, 2000], 0.1, 10)
    lag = 5
    nevecs = 5

    test = ivac.LinearVAC(lag, nevecs=nevecs)
    test.fit(trajs)

    # parameters are correct
    assert test.lag == lag
    assert test.nevecs == nevecs
    assert test.addones == False

    # transform is consistent
    evecs = test.transform(trajs)
    assert allclose_trajs_sign(test.transform(trajs), evecs)

    # different runs give the same result
    test2 = ivac.LinearVAC(lag, nevecs=nevecs)
    test2.fit(trajs)
    fit_is_close(test2, test, nevecs=nevecs)
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # idempotency
    test2.fit(trajs)
    fit_is_close(test2, test, nevecs=nevecs)
    result_is_close(test2, test, trajs, nevecs=nevecs)

    # adjust=True tests
    result_has_ones(test)

    # adjust=False tests
    test2 = ivac.LinearVAC(lag, adjust=False, nevecs=nevecs)
    test2.fit(trajs)
    evecs2 = test2.transform(trajs)
    # orthogonal results for adjust=False
    result_is_orthogonal(test2)
    # evals for adjust=False
    mat = sym(ref_corr.ct_all(evecs2, lag))
    assert np.allclose(np.diag(mat), test2.evals[:nevecs])
    assert np.allclose(mat, np.diag(test2.evals[:nevecs]))

    # addones gives the same result
    trajs2 = [traj[:, 1:] for traj in trajs]
    test2 = ivac.LinearVAC(lag, nevecs=nevecs, addones=True)
    test2.fit(trajs2)
    assert test2.addones == True

    result_is_close(test2, test, trajs2, trajs, nevecs=nevecs)

    # trajectory order doesn't matter
    test2 = ivac.LinearVAC(lag, nevecs=nevecs)
    test2.fit(trajs[::-1])
    result_is_close(test2, test, trajs, nevecs=nevecs)

    # scaling of features doesn't matter
    trajs2 = [2.0 * traj for traj in trajs]
    test2 = ivac.LinearVAC(lag, nevecs=nevecs)
    test2.fit(trajs2)
    result_is_close(test2, test, trajs2, trajs, nevecs=nevecs)

    # order of features doesn't matter
    trajs2 = [traj[:, ::-1] for traj in trajs]
    test2 = ivac.LinearVAC(lag, nevecs=nevecs)
    test2.fit(trajs2)
    result_is_close(test2, test, trajs2, trajs, nevecs=nevecs)


def test_ivac():
    """Tests for LinearIVAC."""

    trajs = make_data([1000, 1500, 2000], 0.1, 10)
    minlag = 0
    maxlag = 10
    nevecs = 5

    # IVAC with minlag=maxlag is the same as VAC
    ref = ivac.LinearVAC(5, nevecs=nevecs)
    ref.fit(trajs)
    test = ivac.LinearIVAC(5, 5, nevecs=nevecs)
    test.fit(trajs)
    assert np.allclose(test.evals[:nevecs], ref.evals[:nevecs])
    assert np.allclose(test.its[:nevecs], ref.its[:nevecs])
    assert allclose_trajs_sign(test.transform(trajs), ref.transform(trajs))
    assert test.minlag == 5
    assert test.maxlag == 5
    assert test.nevecs == nevecs
    assert test.addones == False

    test = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs)
    test.fit(trajs)

    # parameters are correct
    assert test.minlag == minlag
    assert test.maxlag == maxlag
    assert test.nevecs == nevecs
    assert test.addones == False

    # transform is consistent
    evecs = test.transform(trajs)
    assert allclose_trajs_sign(test.transform(trajs), evecs)

    # different runs give the same result
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs)
    test2.fit(trajs)
    fit_is_close(test2, test, nevecs=nevecs)
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # evals
    mat = sym(ref_corr.ic_all(evecs, lags=np.arange(minlag, maxlag + 1)))
    assert np.allclose(np.diag(mat), test.evals[:nevecs])
    assert np.allclose(mat, np.diag(test.evals[:nevecs]))

    # idempotency
    test2.fit(trajs)
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_sign(test2.evecs[:nevecs], test.evecs[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # addones gives the same result
    trajs2 = [traj[:, 1:] for traj in trajs]
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs, addones=True)
    test2.fit(trajs2)
    assert test2.addones == True
    result_is_close(test2, test, trajs2, trajs, nevecs=nevecs)

    # direct gives the same result
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs, method="direct")
    test2.fit(trajs)
    assert test2.method == "direct"
    result_is_close(test2, test, trajs, nevecs=nevecs)

    # trajectory order doesn't matter
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs)
    test2.fit(trajs[::-1])
    result_is_close(test2, test, trajs, nevecs=nevecs)

    # scaling of features doesn't matter
    trajs2 = [2.0 * traj for traj in trajs]
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs)
    test2.fit(trajs2)
    result_is_close(test2, test, trajs2, trajs, nevecs=nevecs)

    # order of features doesn't matter
    trajs2 = [traj[:, ::-1] for traj in trajs]
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs)
    test2.fit(trajs2)
    result_is_close(test2, test, trajs2, trajs, nevecs=nevecs)


def test_vac_scan():
    """Test that LinearVACScan and LinearVAC give the same results."""

    lags = np.array([1, 2, 3, 4, 5, 6])
    nevecs = 5

    for addones in [True, False]:
        trajs = make_data([1000, 1500, 2000], 0.1, 10)
        if addones:
            trajs = [traj[:, 1:] for traj in trajs]
        for method in ["direct", "fft-all"]:
            scan = ivac.LinearVACScan(
                lags,
                nevecs=nevecs,
                addones=addones,
                adjust=False,
                method=method,
            )
            scan.fit(trajs)

            for lag in lags:

                # check parameters are correct
                test = scan[lag]
                assert test.lag == lag
                assert test.nevecs == nevecs
                assert test.addones == addones

                # check against LinearVAC reference
                ref = ivac.LinearVAC(
                    lag, nevecs=nevecs, addones=addones, adjust=False
                )
                ref.fit(trajs)
                fit_is_close(test, ref, nevecs=nevecs)


def test_ivac_scan():
    """Test that LinearIVACScan and LinearIVAC give the same results."""

    lags = np.array([0, 2, 4, 6, 8, 10])
    nevecs = 5

    for addones in [True, False]:
        trajs = make_data([1000, 1500, 2000], 0.1, 10)
        if addones:
            trajs = [traj[:, 1:] for traj in trajs]
        for method in ["direct", "fft", "fft-all"]:
            for lagstep in [1, 2]:
                scan = ivac.LinearIVACScan(
                    lags,
                    lagstep=lagstep,
                    nevecs=nevecs,
                    addones=addones,
                    adjust=False,
                    method=method,
                )
                scan.fit(trajs)

                nlags = len(lags)
                for i in range(nlags):
                    for j in range(i, nlags):
                        minlag = lags[i]
                        maxlag = lags[j]

                        # skip the identity operator
                        if minlag == 0 and maxlag == 0:
                            continue

                        # check parameters are correct
                        test = scan[minlag, maxlag]
                        assert test.minlag == minlag
                        assert test.maxlag == maxlag
                        assert test.lagstep == lagstep
                        assert test.nevecs == nevecs
                        assert test.addones == addones

                        # check against LinearIVAC reference
                        ref = ivac.LinearIVAC(
                            minlag,
                            maxlag,
                            lagstep=lagstep,
                            nevecs=nevecs,
                            adjust=False,
                            addones=addones,
                        )
                        ref.fit(trajs)
                        fit_is_close(test, ref, nevecs=nevecs)
