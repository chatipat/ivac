import numpy as np
from ivac.linear import _ivac_weights
from ivac.utils import (
    compute_ic,
    compute_c0,
    batch_compute_ic,
    batch_compute_c0,
)
import ref_corr as ref
from data import indicator_basis, ou_process


def test_corr():
    """Test correlation matrix calculations."""

    # generate trajectories
    ns = [1000, 1500, 2000]
    dt = 0.1
    nbins = 10
    trajs = [indicator_basis(ou_process(n, dt), nbins) for n in ns]

    vac_lags1 = [1, 2, 5, 10, 20]
    ivac_lags1 = [np.arange(0, 20, 2), np.arange(5, 20), np.arange(10, 20)]
    weights1 = _ivac_weights(trajs, np.arange(0, 21), 20, method="fft")

    vac_lags2 = [1, 2, 5, 10]
    ivac_lags2 = [np.arange(0, 5), np.arange(0, 10), np.arange(5, 10)]
    weights2 = _ivac_weights(trajs, np.arange(0, 11), 10, method="direct")

    check_corr_all(trajs, vac_lags1, ivac_lags1 + ivac_lags2)

    check_corr_trunc(trajs, 20, vac_lags1, ivac_lags1)
    check_corr_trunc(trajs, 10, vac_lags2, ivac_lags2)

    check_corr_rt(trajs, 20, weights1, vac_lags1, ivac_lags1)
    check_corr_rt(trajs, 10, weights2, vac_lags2, ivac_lags2)

    check_corr_batch(trajs)


def check_corr_all(trajs, vac_lags, ivac_lags):
    """Test correlation matrices for equilibrium IVAC."""

    # check C(0) against reference
    assert np.allclose(
        compute_c0(trajs),
        ref.c0_all(trajs),
    )

    # C(t) = C(0) when t = 0
    assert np.allclose(
        compute_ic(trajs, 0),
        ref.c0_all(trajs),
    )

    # check C(t) against reference
    for lag in vac_lags:
        assert np.allclose(
            compute_ic(trajs, lag),
            ref.ct_all(trajs, lag),
        )
        assert np.allclose(
            compute_c0(trajs, lags=lag),
            ref.c0_all_adj_ct(trajs, lag),
        )

    # C(tmin, tmax) = C(t) when t = tmin = tmax
    for lag in vac_lags:
        assert np.allclose(
            compute_ic(trajs, [lag]),
            ref.ct_all(trajs, lag),
        )
        assert np.allclose(
            compute_c0(trajs, lags=[lag]),
            ref.c0_all_adj_ct(trajs, lag),
        )

    # check C(tmin, tmax) against reference
    for method in ["direct", "fft"]:
        for lags in ivac_lags:
            assert np.allclose(
                compute_ic(trajs, lags, method=method),
                ref.ic_all(trajs, lags),
            )
            assert np.allclose(
                compute_c0(trajs, lags=lags, method=method),
                ref.c0_all_adj_ic(trajs, lags),
            )


def check_corr_trunc(trajs, cutlag, vac_lags, ivac_lags):
    """Test correlation matrices for reweighting."""

    # check C(0) against reference
    assert np.allclose(
        compute_c0(trajs, weights=cutlag),
        ref.c0_trunc(trajs, cutlag),
    )

    # C(t) = C(0) when t = 0
    assert np.allclose(
        compute_ic(trajs, 0, weights=cutlag),
        ref.c0_trunc(trajs, cutlag),
    )

    # check C(t) against reference
    for lag in vac_lags:
        assert np.allclose(
            compute_ic(trajs, lag, weights=cutlag),
            ref.ct_trunc(trajs, lag, cutlag),
        )

    # C(tmin, tmax) = C(t) when t = tmin = tmax
    for lag in vac_lags:
        assert np.allclose(
            compute_ic(trajs, [lag], weights=cutlag),
            ref.ct_trunc(trajs, lag, cutlag),
        )

    # check C(tmin, tmax) against reference
    for method in ["direct", "fft"]:
        for lags in ivac_lags:
            assert np.allclose(
                compute_ic(trajs, lags, weights=cutlag, method=method),
                ref.ic_trunc(trajs, lags, cutlag),
            )


def check_corr_rt(trajs, cutlag, weights, vac_lags, ivac_lags):
    """Test correlation matrices for reweighted IVAC."""

    # check C(0) against reference
    assert np.allclose(
        compute_c0(trajs, weights=weights),
        ref.c0_rt(trajs, cutlag, weights),
    )

    # C(t) = C(0) when t = 0
    assert np.allclose(
        compute_ic(trajs, 0, weights=weights),
        ref.c0_rt(trajs, cutlag, weights),
    )

    # check C(t) against reference
    for lag in vac_lags:
        assert np.allclose(
            compute_ic(trajs, lag, weights=weights),
            ref.ct_rt(trajs, lag, cutlag, weights),
        )
        assert np.allclose(
            compute_c0(trajs, lags=lag, weights=weights),
            ref.c0_rt_adj_ct(trajs, lag, cutlag, weights),
        )

    # C(tmin, tmax) = C(t) when t = tmin = tmax
    for lag in vac_lags:
        assert np.allclose(
            compute_ic(trajs, [lag], weights=weights),
            ref.ct_rt(trajs, lag, cutlag, weights),
        )
        assert np.allclose(
            compute_c0(trajs, lags=[lag], weights=weights),
            ref.c0_rt_adj_ct(trajs, lag, cutlag, weights),
        )

    # check C(tmin, tmax) against reference
    for method in ["direct", "fft"]:
        for lags in ivac_lags:
            assert np.allclose(
                compute_ic(trajs, lags, weights=weights, method=method),
                ref.ic_rt(trajs, lags, cutlag, weights),
            )
            assert np.allclose(
                compute_c0(trajs, lags=lags, weights=weights, method=method),
                ref.c0_rt_adj_ic(trajs, lags, cutlag, weights),
            )


def check_corr_batch(trajs):
    """Test batch correlation matrix calculations."""

    params = [np.arange(0, 11, 2) + i for i in range(1, 11)]
    lags = np.unique(np.concatenate(params))
    cutlag = max(lags)
    weights = _ivac_weights(trajs, lags, cutlag, method="fft")

    for method in ["direct", "fft", "fft-all"]:

        # equilibrium IVAC

        batch = batch_compute_ic(trajs, lags, method=method)
        for lag, test in zip_check(lags, batch):
            assert np.allclose(test, ref.ct_all(trajs, lag))

        batch = batch_compute_ic(trajs, params, method=method)
        for param, test in zip_check(params, batch):
            assert np.allclose(test, ref.ic_all(trajs, param))

        # nonequilibrium IVAC

        batch = batch_compute_ic(trajs, lags, weights=cutlag, method=method)
        for lag, test in zip_check(lags, batch):
            assert np.allclose(test, ref.ct_trunc(trajs, lag, cutlag))

        batch = batch_compute_ic(trajs, params, weights=cutlag, method=method)
        for param, test in zip_check(params, batch):
            assert np.allclose(test, ref.ic_trunc(trajs, param, cutlag))

        batch = batch_compute_ic(trajs, lags, weights=weights, method=method)
        for lag, test in zip_check(lags, batch):
            assert np.allclose(test, ref.ct_rt(trajs, lag, cutlag, weights))

        batch = batch_compute_ic(trajs, params, weights=weights, method=method)
        for param, test in zip_check(params, batch):
            assert np.allclose(test, ref.ic_rt(trajs, param, cutlag, weights))

        batch = batch_compute_c0(
            trajs, params=lags, weights=weights, method=method
        )
        for lag, test in zip_check(lags, batch):
            assert np.allclose(
                test, ref.c0_rt_adj_ct(trajs, lag, cutlag, weights)
            )

        batch = batch_compute_c0(
            trajs, params=params, weights=weights, method=method
        )
        for param, test in zip_check(params, batch):
            assert np.allclose(
                test, ref.c0_rt_adj_ic(trajs, param, cutlag, weights)
            )


def zip_check(params, batch):
    """Zip, but check that the first parameter is completely used."""
    batch = iter(batch)
    for param in params:
        try:
            test = next(batch)
        except StopIteration:
            raise ValueError("batch stopped early")
        yield param, test
