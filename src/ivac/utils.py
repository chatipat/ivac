import numpy as np
import numba as nb
from scipy import linalg, signal


def preprocess_trajs(trajs):
    """Convert trajectories a list of 2d ndarray."""
    nfeatures = get_nfeatures(trajs)
    result = []
    for traj in trajs:
        traj = np.asarray(traj, dtype=np.float64)
        if traj.ndim != 2:
            raise ValueError("each trajectory must be a 2d array")
        if traj.shape[-1] != nfeatures:
            raise ValueError("all trajectories must have the same features")
        result.append(traj)
    return result


def get_nfeatures(trajs):
    """Get the number of features in a list of trajectories."""
    return np.shape(trajs[0])[-1]


def corr(
    func1,
    func2,
    trajs,
    maxlag=0,
    weights=None,
):
    """Compute a correlation matrix from trajectories.

    Parameters
    ----------
    func1, func2 : callable
        Transformations representing the action of an operator.
        These are functions taking a trajectory and a length,
        and returning a trajectory of that length.
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
    maxlag : int
        Maximum lag time.
    weights : list of (n_frames[i],) ndarray, optional
        Weight of trajectory starting at each configuration.

    Returns
    -------
    (n_features, n_features) ndarray
        Correlation matrix.

    """
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    if weights is None:
        for traj in trajs:
            length = len(traj) - maxlag
            if length > 0:
                x = func1(traj, length)
                y = func2(traj, length)
                numer += x.T @ y
                denom += len(x)
    else:
        for traj, weight in zip(trajs, weights):
            length = len(traj) - maxlag
            if length > 0:
                w = weight[:length]
                x = func1(traj, length)
                y = func2(traj, length)
                xw = x * w[:, None]
                numer += xw.T @ y
                denom += np.sum(w)
    return numer / denom


# functions for use with corr


def delay(lag):
    """Create a function that applies a delay to a trajectory."""

    def func(traj, length):
        return traj[lag : length + lag]

    return func


def integrate_all(lags, weights, mode="direct"):
    """Create a function for IVAC's integrated correlation matrix."""

    if mode == "direct":

        def func(traj, length):
            iy = np.zeros_like(traj[:length])
            for lag, weight in zip(lags, weights):
                if length > lag:
                    iy[: length - lag] += weight * traj[lag:length]
            return iy

    elif mode == "fft":

        minlag = min(lags)
        maxlag = max(lags)
        window = np.zeros(maxlag - minlag + 1)
        window[lags - minlag] = weights
        window = window[::-1, None]

        def func(traj, length):
            iy = np.zeros_like(traj[:length])
            traj = traj[minlag:length]
            conv = signal.fftconvolve(traj, window, mode="full", axes=0)
            iy[: length - minlag] = conv[maxlag - minlag :]
            return iy

    else:
        raise ValueError("mode must be 'direct' or 'fft'")

    return func


def integrate(lags, mode="direct"):
    """Create a function that integrates a trajectory over lag times."""

    if mode == "direct":

        def func(traj, length):
            result = np.zeros_like(traj[:length])
            for lag in lags:
                result += traj[lag : length + lag]
            return result

    elif mode == "fft":

        minlag = min(lags)
        maxlag = max(lags)
        window = np.zeros(maxlag - minlag + 1)
        window[lags - minlag] = 1.0
        window = window[::-1, None]

        def func(traj, length):
            traj = traj[minlag : length + maxlag]
            return signal.fftconvolve(traj, window, mode="valid", axes=0)

    else:
        raise ValueError("mode must be 'direct' or 'fft'")

    return func


# equilibrium IVAC


def c0_all(trajs):
    return corr(delay(0), delay(0), trajs)


def ct_all(trajs, lag):
    return corr(delay(0), delay(lag), trajs, lag)


def c0_all_adj_ct(trajs, lag):
    return c0_all_adj_ic(trajs, np.array([lag]))


def ic_all(trajs, lags, mode="direct"):
    lags = np.asarray(lags)
    lengths = np.array([len(traj) for traj in trajs])
    samples = np.sum(np.maximum(lengths[None, :] - lags[:, None], 0), axis=-1)
    weights = np.sum(lengths) / samples
    return corr(delay(0), integrate_all(lags, weights, mode=mode), trajs)


def c0_all_adj_ic(trajs, lags, mode="direct"):
    lags = np.asarray(lags)
    lengths = np.array([len(traj) for traj in trajs])
    samples = np.sum(np.maximum(lengths[None, :] - lags[:, None], 0), axis=-1)
    wlags = 1.0 / samples
    weights = [_adj_all(len(traj), lags, wlags) for traj in trajs]
    return corr(delay(0), delay(0), trajs, weights=weights)


@nb.njit
def _adj_all(wlen, lags, wlags):

    # create window
    minlag = min(lags)
    maxlag = max(lags)
    nlags = maxlag - minlag + 1
    window = np.zeros(nlags)
    for i in range(len(lags)):
        window[lags[i] - minlag] = wlags[i]

    # weights form right triangles starting from each end
    a = 0.0
    weight = np.zeros(wlen)
    for i in range(minlag, wlen):
        if i <= maxlag:
            a += window[i - minlag]
        weight[i] += a
        weight[wlen - 1 - i] += a
    return weight


# nonequilibrium IVAC: weight estimation


def c0_trunc(trajs, cutlag):
    return corr(delay(0), delay(0), trajs, cutlag)


def ct_trunc(trajs, lag, cutlag):
    return corr(delay(0), delay(lag), trajs, cutlag)


def ic_trunc(trajs, lags, cutlag, mode="direct"):
    return corr(delay(0), integrate(lags, mode=mode), trajs, cutlag)


# nonequilibrium IVAC: reweighted matrices with truncated data


def c0_rt(trajs, cutlag, weights):
    return corr(delay(0), delay(0), trajs, cutlag, weights=weights)


def ct_rt(trajs, lag, cutlag, weights):
    return corr(delay(0), delay(lag), trajs, cutlag, weights=weights)


def c0_rt_adj_ct(trajs, lag, cutlag, weights):
    return c0_rt_adj_ic(trajs, [lag], cutlag, weights)


def ic_rt(trajs, lags, cutlag, weights, mode="direct"):
    return corr(
        delay(0), integrate(lags, mode=mode), trajs, cutlag, weights=weights
    )


def c0_rt_adj_ic(trajs, lags, cutlag, weights, mode="direct"):
    lags = np.asarray(lags)
    if mode == "direct":
        weights = [_adj_rt_direct(weight, lags, cutlag) for weight in weights]
    elif mode == "fft":
        weights = [_adj_rt_fft(weight, lags, cutlag) for weight in weights]
    else:
        raise ValueError("mode must be 'direct' or 'fft'")
    return corr(delay(0), delay(0), trajs, weights=weights)


@nb.njit
def _adj_rt_direct(weight, lags, cutlag):
    result = np.zeros(len(weight))
    length = len(weight) - cutlag
    if length > 0:
        nlags = len(lags)
        for i in range(length):
            result[i] += weight[i] * nlags
        for n in range(nlags):
            lag = lags[n]
            for i in range(length):
                result[lag + i] += weight[i]
    return result


def _adj_rt_fft(weight, lags, cutlag):
    result = np.zeros(len(weight))
    length = len(weight) - cutlag
    if length > 0:

        # create window
        minlag = min(lags)
        maxlag = max(lags)
        window = np.zeros(maxlag - minlag + 1)
        window[lags - minlag] = 1.0

        # initial and integrated points
        weight = weight[:length]
        result[:length] += weight * len(lags)
        result[minlag : length + maxlag] += signal.fftconvolve(weight, window)

    return result
