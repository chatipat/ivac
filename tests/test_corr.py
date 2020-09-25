import numpy as np
import ivac.utils as impl
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
    coeffs1 = np.random.rand(nbins)

    vac_lags2 = [1, 2, 5, 10]
    ivac_lags2 = [np.arange(0, 5), np.arange(0, 10), np.arange(5, 10)]
    coeffs2 = np.random.rand(nbins)

    check_corr_all(trajs, vac_lags1, ivac_lags1 + ivac_lags2)

    check_corr_trunc(trajs, 20, vac_lags1, ivac_lags1)
    check_corr_trunc(trajs, 10, vac_lags2, ivac_lags2)

    check_corr_rt(trajs, 20, coeffs1, vac_lags1, ivac_lags1)
    check_corr_rt(trajs, 10, coeffs2, vac_lags2, ivac_lags2)


def check_corr_all(trajs, vac_lags, ivac_lags):
    """Test correlation matrices for equilibrium IVAC."""

    # check C(0) against reference
    assert np.allclose(
        impl.c0_all(trajs),
        ref.c0_all(trajs),
    )

    # C(t) = C(0) when t = 0
    assert np.allclose(
        impl.ct_all(trajs, 0),
        ref.c0_all(trajs),
    )

    # check C(t) against reference
    for lag in vac_lags:
        assert np.allclose(
            impl.ct_all(trajs, lag),
            ref.ct_all(trajs, lag),
        )
        assert np.allclose(
            impl.c0_all_adj_ct(trajs, lag),
            ref.c0_all_adj_ct(trajs, lag),
        )

    # C(tmin, tmax) = C(t) when t = tmin = tmax
    for lag in vac_lags:
        assert np.allclose(
            impl.ic_all(trajs, [lag]),
            ref.ct_all(trajs, lag),
        )
        assert np.allclose(
            impl.c0_all_adj_ic(trajs, [lag]),
            ref.c0_all_adj_ct(trajs, lag),
        )

    # check C(tmin, tmax) against reference
    for lags in ivac_lags:
        assert np.allclose(
            impl.ic_all(trajs, lags),
            ref.ic_all(trajs, lags),
        )
        assert np.allclose(
            impl.c0_all_adj_ic(trajs, lags),
            ref.c0_all_adj_ic(trajs, lags),
        )


def check_corr_trunc(trajs, cutlag, vac_lags, ivac_lags):
    """Test correlation matrices for reweighting."""

    # check C(0) against reference
    assert np.allclose(
        impl.c0_trunc(trajs, cutlag),
        ref.c0_trunc(trajs, cutlag),
    )

    # C(t) = C(0) when t = 0
    assert np.allclose(
        impl.ct_trunc(trajs, 0, cutlag),
        ref.c0_trunc(trajs, cutlag),
    )

    # check C(t) against reference
    for lag in vac_lags:
        assert np.allclose(
            impl.ct_trunc(trajs, lag, cutlag),
            ref.ct_trunc(trajs, lag, cutlag),
        )

    # C(tmin, tmax) = C(t) when t = tmin = tmax
    for lag in vac_lags:
        assert np.allclose(
            impl.ic_trunc(trajs, [lag], cutlag),
            ref.ct_trunc(trajs, lag, cutlag),
        )

    # check C(tmin, tmax) against reference
    for lags in ivac_lags:
        assert np.allclose(
            impl.ic_trunc(trajs, lags, cutlag),
            ref.ic_trunc(trajs, lags, cutlag),
        )


def check_corr_rt(trajs, cutlag, coeffs, vac_lags, ivac_lags):
    """Test correlation matrices for reweighted IVAC."""

    # check C(0) against reference
    assert np.allclose(
        impl.c0_rt(trajs, cutlag, coeffs),
        ref.c0_rt(trajs, cutlag, coeffs),
    )

    # C(t) = C(0) when t = 0
    assert np.allclose(
        impl.ct_rt(trajs, 0, cutlag, coeffs),
        ref.c0_rt(trajs, cutlag, coeffs),
    )

    # check C(t) against reference
    for lag in vac_lags:
        assert np.allclose(
            impl.ct_rt(trajs, lag, cutlag, coeffs),
            ref.ct_rt(trajs, lag, cutlag, coeffs),
        )
        assert np.allclose(
            impl.c0_rt_adj_ct(trajs, lag, cutlag, coeffs),
            ref.c0_rt_adj_ct(trajs, lag, cutlag, coeffs),
        )

    # C(tmin, tmax) = C(t) when t = tmin = tmax
    for lag in vac_lags:
        assert np.allclose(
            impl.ic_rt(trajs, [lag], cutlag, coeffs),
            ref.ct_rt(trajs, lag, cutlag, coeffs),
        )
        assert np.allclose(
            impl.c0_rt_adj_ic(trajs, [lag], cutlag, coeffs),
            ref.c0_rt_adj_ct(trajs, lag, cutlag, coeffs),
        )

    # check C(tmin, tmax) against reference
    for lags in ivac_lags:
        assert np.allclose(
            impl.ic_rt(trajs, lags, cutlag, coeffs),
            ref.ic_rt(trajs, lags, cutlag, coeffs),
        )
        assert np.allclose(
            impl.c0_rt_adj_ic(trajs, lags, cutlag, coeffs),
            ref.c0_rt_adj_ic(trajs, lags, cutlag, coeffs),
        )
