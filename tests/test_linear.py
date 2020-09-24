import numpy as np
from data import indicator_basis, ou_process
from utils import allclose_sign, allclose_trajs, allclose_trajs_sign

import ivac


def make_data(ns, dt, nbins):
    """Generate featurized data for testing."""
    return [indicator_basis(ou_process(n, dt), nbins) for n in ns]


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
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_sign(test2.evecs[:nevecs], test.evecs[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # evals
    mat = ivac.linear._sym(ivac.covmat(evecs, lag=lag))
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
    test2 = ivac.LinearVAC(lag, nevecs=nevecs, addones=True)
    test2.fit(trajs2)
    assert test2.addones == True
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs2), evecs)

    # trajectory order doesn't matter
    test2 = ivac.LinearVAC(lag, nevecs=nevecs)
    test2.fit(trajs[::-1])
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # scaling of features doesn't matter
    trajs2 = [2.0 * traj for traj in trajs]
    test2 = ivac.LinearVAC(lag, nevecs=nevecs)
    test2.fit(trajs2)
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs2), evecs)

    # order of features doesn't matter
    trajs2 = [traj[:, ::-1] for traj in trajs]
    test2 = ivac.LinearVAC(lag, nevecs=nevecs)
    test2.fit(trajs2)
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs2), evecs)


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
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_sign(test2.evecs[:nevecs], test.evecs[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # evals
    mat = ivac.linear._sym(
        ivac.linear._icov(evecs, lags=np.arange(minlag, maxlag + 1))
    )
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
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs2), evecs)

    # fft gives the same result
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs, method="fft")
    test2.fit(trajs)
    assert test2.method == "fft"
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # conv gives the same result
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs, method="conv")
    test2.fit(trajs)
    assert test2.method == "conv"
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # fftconv gives the same result
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs, method="fftconv")
    test2.fit(trajs)
    assert test2.method == "fftconv"
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # trajectory order doesn't matter
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs)
    test2.fit(trajs[::-1])
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs), evecs)

    # scaling of features doesn't matter
    trajs2 = [2.0 * traj for traj in trajs]
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs)
    test2.fit(trajs2)
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs2), evecs)

    # order of features doesn't matter
    trajs2 = [traj[:, ::-1] for traj in trajs]
    test2 = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs)
    test2.fit(trajs2)
    assert np.allclose(test2.evals[:nevecs], test.evals[:nevecs])
    assert np.allclose(test2.its[:nevecs], test.its[:nevecs])
    assert allclose_trajs_sign(test2.transform(trajs2), evecs)


def test_vac_scan():
    """Test that LinearVACScan and LinearVAC give the same results."""

    lags = np.array([1, 2, 3, 4, 5, 6])
    nevecs = 5

    for addones in [True, False]:
        trajs = make_data([1000, 1500, 2000], 0.1, 10)
        if addones:
            trajs = [traj[:, 1:] for traj in trajs]
        for method in ["direct", "fft"]:
            scan = ivac.LinearVACScan(
                lags, nevecs=nevecs, addones=addones, method=method
            )
            scan.fit(trajs)

            for lag in lags:

                # check parameters are correct
                test = scan[lag]
                assert test.lag == lag
                assert test.nevecs == nevecs
                assert test.addones == addones

                # check against LinearVAC reference
                ref = ivac.LinearVAC(lag, nevecs=nevecs, addones=addones)
                ref.fit(trajs)
                assert np.allclose(test.evals[:nevecs], ref.evals[:nevecs])
                assert np.allclose(test.its[:nevecs], ref.its[:nevecs])
                assert allclose_sign(
                    test.evecs[:, :nevecs], ref.evecs[:, :nevecs]
                )


def test_ivac_scan():
    """Test that LinearIVACScan and LinearIVAC give the same results."""

    lags = np.array([0, 2, 4, 6, 8, 10])
    nevecs = 5

    for addones in [True, False]:
        trajs = make_data([1000, 1500, 2000], 0.1, 10)
        if addones:
            trajs = [traj[:, 1:] for traj in trajs]
        for method in ["direct", "fft"]:
            for lagstep in [1, 2]:
                scan = ivac.LinearIVACScan(
                    lags,
                    lagstep=lagstep,
                    nevecs=nevecs,
                    addones=addones,
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
                            addones=addones,
                        )
                        ref.fit(trajs)
                        assert np.allclose(
                            test.evals[:nevecs], ref.evals[:nevecs]
                        )
                        assert np.allclose(test.its[:nevecs], ref.its[:nevecs])
                        assert allclose_sign(
                            test.evecs[:, :nevecs], ref.evecs[:, :nevecs]
                        )
