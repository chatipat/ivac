import numpy as np
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
    weights : callable or (n_features,) ndarray, optional
        Function from features to weights or coefficients for the
        stationary distribution expressed in terms of the features.

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
        if not callable(weights):
            coeffs = np.asarray(weights)
            weights = lambda traj: traj @ coeffs
        for traj in trajs:
            length = len(traj) - maxlag
            if length > 0:
                w = weights(traj[:length])
                x = func1(traj, length)
                y = func2(traj, length)
                numer += np.einsum("n,ni,nj", w, x, y)
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


def adjust_all(lags, weights):
    """Create a function for IVAC's adjusted correlation matrix."""

    def func(traj):
        w = np.zeros(len(traj))
        for lag, weight in zip(lags, weights):
            length = len(traj) - lag
            if length > 0:
                w[:length] += weight
                w[lag:] += weight
        return w

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


def adjust(lags, cutlag, wfunc):
    """Create a function for IVAC's adjusted correlation matrix."""
    if not callable(wfunc):
        coeffs = np.asarray(wfunc)
        wfunc = lambda traj: traj @ coeffs

    def func(traj):
        w = np.zeros(len(traj))
        length = len(traj) - cutlag
        if length > 0:
            weights = wfunc(traj[:length])
            for lag in lags:
                w[:length] += weights
                w[lag : length + lag] += weights
        return w

    return func


# equilibrium IVAC


def c0_all(trajs):
    return corr(delay(0), delay(0), trajs)


def ct_all(trajs, lag):
    return corr(delay(0), delay(lag), trajs, lag)


def c0_all_adj_ct(trajs, lag):
    lengths = np.array([len(traj) for traj in trajs])
    samples = np.sum(np.maximum(lengths - lag, 0))
    weight = 1.0 / samples
    return corr(delay(0), delay(0), trajs, weights=adjust_all([lag], [weight]))


def ic_all(trajs, lags, mode="direct"):
    lags = np.asarray(lags)
    lengths = np.array([len(traj) for traj in trajs])
    samples = np.sum(np.maximum(lengths[None, :] - lags[:, None], 0), axis=-1)
    weights = np.sum(lengths) / samples
    return corr(delay(0), integrate_all(lags, weights, mode=mode), trajs)


def c0_all_adj_ic(trajs, lags):
    lags = np.asarray(lags)
    lengths = np.array([len(traj) for traj in trajs])
    samples = np.sum(np.maximum(lengths[None, :] - lags[:, None], 0), axis=-1)
    weights = 1.0 / samples
    return corr(delay(0), delay(0), trajs, weights=adjust_all(lags, weights))


# nonequilibrium IVAC: weight estimation


def c0_trunc(trajs, cutlag):
    return corr(delay(0), delay(0), trajs, cutlag)


def ct_trunc(trajs, lag, cutlag):
    return corr(delay(0), delay(lag), trajs, cutlag)


def ic_trunc(trajs, lags, cutlag, mode="direct"):
    return corr(delay(0), integrate(lags, mode=mode), trajs, cutlag)


# nonequilibrium IVAC: reweighted matrices with truncated data


def c0_rt(trajs, cutlag, w):
    return corr(delay(0), delay(0), trajs, cutlag, w)


def ct_rt(trajs, lag, cutlag, w):
    return corr(delay(0), delay(lag), trajs, cutlag, w)


def c0_rt_adj_ct(trajs, lag, cutlag, w):
    return corr(delay(0), delay(0), trajs, weights=adjust([lag], cutlag, w))


def ic_rt(trajs, lags, cutlag, w, mode="direct"):
    return corr(delay(0), integrate(lags, mode=mode), trajs, cutlag, w)


def c0_rt_adj_ic(trajs, lags, cutlag, w):
    return corr(delay(0), delay(0), trajs, weights=adjust(lags, cutlag, w))
