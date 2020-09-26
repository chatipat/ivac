"""Reference implementations for correlation matrices."""

import numpy as np
from ivac.utils import get_nfeatures


# equilibrium IVAC


def c0_all(trajs):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj in trajs:
        numer += traj.T @ traj
        denom += len(traj)
    return numer / denom


def ct_all(trajs, lag):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj in trajs:
        x = traj[: len(traj) - lag]
        y = traj[lag:]
        numer += x.T @ y
        denom += len(x)
    return numer / denom


def c0_all_adj_ct(trajs, lag):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj in trajs:
        x = traj[: len(traj) - lag]
        y = traj[lag:]
        numer += x.T @ x + y.T @ y
        denom += 2 * len(x)
    return numer / denom


def ic_all(trajs, lags):
    nfeatures = get_nfeatures(trajs)
    ic = np.zeros((nfeatures, nfeatures))
    for lag in lags:
        ic += ct_all(trajs, lag)
    return ic


def c0_all_adj_ic(trajs, lags):
    nfeatures = get_nfeatures(trajs)
    c0 = np.zeros((nfeatures, nfeatures))
    for lag in lags:
        c0 += c0_all_adj_ct(trajs, lag)
    return c0 / len(lags)


# nonequilibrium IVAC: weight estimation


def c0_trunc(trajs, cutlag):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj in trajs:
        x = traj[: len(traj) - cutlag]
        numer += x.T @ x
        denom += len(x)
    return numer / denom


def ct_trunc(trajs, lag, cutlag):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj in trajs:
        x = traj[: len(traj) - cutlag]
        y = traj[lag : len(traj) - cutlag + lag]
        numer += x.T @ y
        denom += len(x)
    return numer / denom


def ic_trunc(trajs, lags, cutlag):
    nfeatures = get_nfeatures(trajs)
    ic = np.zeros((nfeatures, nfeatures))
    for lag in lags:
        ic += ct_trunc(trajs, lag, cutlag)
    return ic


# nonequilibrium IVAC: reweighted matrices with truncated data


def c0_rt(trajs, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj, weight in zip(trajs, weights):
        x = traj[: len(traj) - cutlag]
        w = weight[: len(traj) - cutlag]
        numer += np.einsum("n,ni,nj", w, x, x)
        denom += np.sum(w)
    return numer / denom


def ct_rt(trajs, lag, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj, weight in zip(trajs, weights):
        x = traj[: len(traj) - cutlag]
        y = traj[lag : len(traj) - cutlag + lag]
        w = weight[: len(traj) - cutlag]
        numer += np.einsum("n,ni,nj", w, x, y)
        denom += np.sum(w)
    return numer / denom


def c0_rt_adj_ct(trajs, lag, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj, weight in zip(trajs, weights):
        x = traj[: len(traj) - cutlag]
        y = traj[lag : len(traj) - cutlag + lag]
        w = weight[: len(traj) - cutlag]
        numer += np.einsum("n,ni,nj", w, x, x)
        numer += np.einsum("n,ni,nj", w, y, y)
        denom += 2.0 * np.sum(w)
    return numer / denom


def ic_rt(trajs, lags, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    ic = np.zeros((nfeatures, nfeatures))
    for lag in lags:
        ic += ct_rt(trajs, lag, cutlag, weights)
    return ic


def c0_rt_adj_ic(trajs, lags, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    c0 = np.zeros((nfeatures, nfeatures))
    for lag in lags:
        c0 += c0_rt_adj_ct(trajs, lag, cutlag, weights)
    return c0 / len(lags)


# nonequilibrium IVAC: reweighted matrices with all data


def c0_ra(trajs, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj, weight in zip(trajs, weights):
        w = weight[: len(traj) - cutlag]
        for shift in range(cutlag + 1):
            x = traj[shift : shift + len(traj) - cutlag]
            numer += np.einsum("n,ni,nj", w, x, x)
            denom += np.sum(w)
    return numer / denom


def ct_ra(trajs, lag, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj, weight in zip(trajs, weights):
        w = weight[: len(traj) - cutlag]
        for shift in range(cutlag - lag + 1):
            x = traj[shift : shift + len(traj) - cutlag]
            y = traj[shift + lag : shift + len(traj) - cutlag + lag]
            numer += np.einsum("n,ni,nj", w, x, y)
            denom += np.sum(w)
    return numer / denom


def c0_ra_adj_ct(trajs, lag, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    for traj, weight in zip(trajs, weights):
        w = weight[: len(traj) - cutlag]
        for shift in range(cutlag - lag + 1):
            x = traj[shift : shift + len(traj) - cutlag]
            y = traj[shift + lag : shift + len(traj) - cutlag + lag]
            numer += np.einsum("n,ni,nj", w, x, x)
            numer += np.einsum("n,ni,nj", w, y, y)
            denom += 2.0 * np.sum(w)
    return numer / denom


def ic_ra(trajs, lags, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    ic = np.zeros((nfeatures, nfeatures))
    for lag in lags:
        ic += ct_ra(trajs, lag, cutlag, weights)
    return ic


def c0_ra_adj_ic(trajs, lags, cutlag, weights):
    nfeatures = get_nfeatures(trajs)
    c0 = np.zeros((nfeatures, nfeatures))
    for lag in lags:
        c0 += c0_ra_adj_ct(trajs, lags, cutlag, weights)
    return c0 / len(lags)
