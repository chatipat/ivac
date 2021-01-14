import itertools
import numpy as np
import numba as nb
from scipy import linalg, signal


# -----------------------------------------------------------------------------
# trajectory functions


class LazyTrajectories:
    """Load trajectories lazily from disk.

    Parameters
    ----------
    filenames : list of string
        List of file names. Each file should contain a single trajectory
        in the .npy format.

    """

    def __init__(self, filenames):
        self.filenames = filenames
        self._len = len(filenames)

    def __getitem__(self, key):
        return np.load(self.filenames[key])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return self._len


class PreprocessedTrajectories:
    """Prepare trajectories for further calculation.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_input_features) array-like
        List of featurized trajectories.
    addones : bool, optional
        If True, add a feature of all ones to featurized trajectories.

    """

    def __init__(self, trajs, addones=False):
        self.trajs = trajs
        self.addones = addones
        self._len = len(trajs)

    def __getitem__(self, key):
        nfeatures = get_nfeatures(self.trajs)
        traj = np.asarray(self.trajs[key], dtype=np.float64)
        if traj.ndim != 2:
            raise ValueError("each trajectory must be a 2d array")
        if traj.shape[-1] != nfeatures:
            raise ValueError("all trajectories must have the same features")
        if self.addones:
            ones = np.ones((len(traj), 1))
            traj = np.concatenate([ones, traj], axis=-1)
        return traj

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return self._len


def preprocess_trajs(trajs, addones=False):
    """Prepare trajectories for further calculation.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_input_features) array-like
        List of featurized trajectories.
    addones : bool, optional
        If True, add a feature of all ones to featurized trajectories.

    Returns
    -------
    list of (n_frames[i], n_output_features) ndarray
        Trajectories with an additional feature of all ones.

    """
    return PreprocessedTrajectories(trajs, addones=addones)


def get_nfeatures(trajs):
    """Get the number of features in a list of trajectories.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) array-like
        List of featurized trajectories.

    Returns
    -------
    int
        Number of features.

    """
    return np.shape(trajs[0])[-1]


def trajs_matmul(trajs, coeffs):
    """Right matrix multiply coefficients to all trajectories."""
    result = []
    for traj in trajs:
        result.append(traj @ coeffs)
    return result


@nb.njit
def find_cutlag(weight):
    """Compute the maximum valid lag time from trajectory weights.

    Parameters
    ----------
    weights : (n_frames,) ndarray
        Weight of each configuration in a trajectory.

    Returns
    -------
    int
        Maximum valid lag time in units of frames.

    """

    last = len(weight) - 1  # index of last element

    # index of first nonzero element in the reversed list
    # is the number of zeros at the end
    for lag in range(len(weight)):
        if weight[last - lag] != 0.0:
            return lag

    return len(weight)  # all zeros


def is_cutlag(weights):
    """Return True if weights is given as an integer.

    Parameters
    ----------
    weights : int or list of (n_frames[i],) array-like
        Weights, as either an int (representing uniform weights, but
        with the last int weights set to zero) or a list of weights for
        each configuration.

    Returns
    -------
    bool
        True if int is given, False if a list of weights is given.

    """
    weights = np.asarray(weights, dtype=object)
    return weights.ndim == 0


# -----------------------------------------------------------------------------
# linear algebra


def symeig(a, b=None, nevecs=None):
    """Symmetrize and solve the symmetric eigenvalue problem.

    Parameters
    ----------
    a : (n_features, n_features) ndarray
        Real square matrix for which eigenvalues and eigenvectors will
        be computed. This matrix will be symmetrized.
    b : (n_features, n_features) ndarray, optional
        Real symmetric positive definite matrix.
        If None, use the identity matrix.
    nevecs : int, optional
        Number of eigenvalues and eigenvectors to return.
        If None, return all eigenvalues and eigenvectors.

    Returns
    -------
    evals : (n_evecs,) ndarray
        Eigenvalues in order of decreasing magnitude.
    evecs : (n_features, n_evecs) ndarray)
        Eigenvectors corresponding to the eigenvalues.

    """
    a = 0.5 * (a + a.T)
    evals, evecs = linalg.eigh(a, b)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    return evals[:nevecs], evecs[:, :nevecs]


def solve_stationary(a, b=None):
    """Solve for the left eigenvector with eigenvalue 1.

    Parameters
    ----------
    a : (n_features, n_features) ndarray
        Real square matrix for which to find the left eigenvector with
        eigenvalue 1.
    b : (n_features, n_features) ndarray, optional
        Real symmetric positive definite matrix.
        If None, use the identity matrix.

    Returns
    -------
    (n_features,) ndarray
        Coefficients for the stationary distribution projected onto the
        input features.

    """
    if b is None:
        b = np.identity(len(a))
    w = np.squeeze(linalg.null_space((a - b).T))
    if w.ndim != 1:
        raise ValueError(
            "{} stationary distributions found".format(w.shape[-1])
        )
    return w


# -----------------------------------------------------------------------------
# calculation of single correlation matrices
#
# compute_ic and compute_c0 are the only routines that should be called
# by other code. They should select the correct implementations from
# ic_*, ct_*, and c0_* based on the arguments.


def compute_ic(trajs, lags, *, weights=None, method="fft"):
    """Compute the time-lagged or integrated correlation matrix.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
    lags : int or 1d array-like
        VAC lag time or IVAC lag times.
    weights : int or list of (n_frames[i],) ndarray, optional
        Weight of trajectory starting at each configuration.
        If int, assume uniform weights except for the last int frames,
        which have zero weight.
    method : str, optional
        Method to use to compute the integrated correlation matrix.
        Currently, 'direct' and 'fft' are supported. Method 'direct'
        performs a convolution by direct summation. It tends to be
        faster for fewer or widely separated lag times. Method 'fft'
        performs a convolution using a FFT. Its speed is almost
        independent of the number or range of lag times, but has a
        higher constant cost.

    Returns
    -------
    (n_features, n_features) ndarray
        Time-lagged or integrated correlation matrix.

    """
    lags = np.squeeze(lags)
    assert lags.ndim in [0, 1]
    if weights is None:
        if lags.ndim == 0:
            return ct_all(trajs, lags)
        else:
            return ic_all(trajs, lags, method)
    else:
        if lags.ndim == 0:
            return ct_rt(trajs, lags, weights)
        else:
            return ic_rt(trajs, lags, weights, method)


def compute_c0(trajs, *, lags=None, weights=None, method="fft"):
    """Compute the correlation matrix.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
    lags : int or 1d array-like, optional
        If provided, adjust the correlation matrix so that the constant
        eigenvector is an exact solution (up to numerical precision) of
        linear VAC or IVAC.
    weights : int or list of (n_frames[i],) ndarray, optional
        Weight of trajectory starting at each configuration.
        If int, assume uniform weights except for the last int frames,
        which have zero weight.
    method : str, optional
        Method to use to compute the correlation matrix. Currently,
        'direct' and 'fft' are supported. Method 'direct' performs a
        convolution by direct summation. It tends to be faster for fewer
        or widely separated lag times. Method 'fft' performs a
        convolution using a FFT. Its speed is almost independent of the
        number or range of lag times, but has a higher constant cost.

    Returns
    -------
    (n_features, n_features) ndarray
        Correlation matrix.

    """
    if lags is not None:
        lags = np.squeeze(lags)
        assert lags.ndim in [0, 1]
    if weights is None:
        if lags is None:
            return c0_all(trajs)
        elif lags.ndim == 0:
            return c0_all_adj_ct(trajs, lags)
        else:
            return c0_all_adj_ic(trajs, lags)
    else:
        if lags is None:
            return c0_rt(trajs, weights)
        elif lags.ndim == 0:
            return c0_rt_adj_ct(trajs, lags, weights)
        else:
            return c0_rt_adj_ic(trajs, lags, weights, method)


# The weight parameter can be None, int, or List[ndarray].
#
# None means that the trajectories are sampled from equilibrium.
#
# List[ndarray] gives the weight of each frame. If a lag time is to be
# applied, there must be at least that many frames with weight zero
# at the end of the trajectory.
#
# int is the same as List[ndarray] with uniform weights, except that
# the last int frames have weight zero. This is so that the maximum
# lag time that can be applied is int.


def corr(func1, func2, trajs, weights=0):
    """Compute a correlation matrix from trajectories.

    Parameters
    ----------
    func1, func2 : callable
        Transformations representing the action of an operator.
        These are functions taking a trajectory and a length,
        and returning a trajectory of that length.
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
    weights : int or list of (n_frames[i],) ndarray, optional
        Weight of trajectory starting at each configuration.
        If int, assume uniform weights except for the last int frames,
        which have zero weight.

    Returns
    -------
    (n_features, n_features) ndarray
        Correlation matrix.

    """
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((nfeatures, nfeatures))
    denom = 0.0
    if is_cutlag(weights):
        for traj in trajs:
            length = len(traj) - weights
            if length > 0:
                x = func1(traj, length)
                y = func2(traj, length)
                assert len(x) == length
                assert len(y) == length
                numer += x.T @ y
                denom += length
    else:
        for traj, weight in zip(trajs, weights):
            length = len(traj) - find_cutlag(weight)
            if length > 0:
                w = weight[:length]
                x = func1(traj, length)
                y = func2(traj, length)
                assert len(w) == length
                assert len(x) == length
                assert len(y) == length
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


def integrate_all(lags, wlags, method):
    """Create a function for IVAC's integrated correlation matrix."""

    if method == "direct":

        def func(traj, length):
            iy = np.zeros_like(traj[:length])
            for lag, weight in zip(lags, wlags):
                if length > lag:
                    iy[: length - lag] += weight * traj[lag:length]
            return iy

    elif method == "fft":

        minlag = min(lags)
        maxlag = max(lags)
        window = np.zeros(maxlag - minlag + 1)
        window[lags - minlag] = wlags
        window = window[::-1, None]

        def func(traj, length):
            iy = np.zeros_like(traj[:length])
            traj = traj[minlag:length]
            conv = signal.fftconvolve(traj, window, mode="full", axes=0)
            iy[: length - minlag] = conv[maxlag - minlag :]
            return iy

    else:
        raise ValueError("method must be 'direct' or 'fft'")

    return func


def integrate(lags, method):
    """Create a function that integrates a trajectory over lag times."""

    if method == "direct":

        def func(traj, length):
            result = np.zeros_like(traj[:length])
            for lag in lags:
                result += traj[lag : length + lag]
            return result

    elif method == "fft":

        minlag = min(lags)
        maxlag = max(lags)
        window = np.zeros(maxlag - minlag + 1)
        window[lags - minlag] = 1.0
        window = window[::-1, None]

        def func(traj, length):
            traj = traj[minlag : length + maxlag]
            return signal.fftconvolve(traj, window, mode="valid", axes=0)

    else:
        raise ValueError("method must be 'direct' or 'fft'")

    return func


# equilibrium IVAC


def c0_all(trajs):
    return corr(delay(0), delay(0), trajs)


def ct_all(trajs, lag):
    return corr(delay(0), delay(lag), trajs, lag)


def c0_all_adj_ct(trajs, lag):
    return c0_all_adj_ic(trajs, [lag])


def ic_all(trajs, lags, method):
    lags = np.asarray(lags)
    lengths = np.array([len(traj) for traj in trajs])
    samples = np.sum(np.maximum(lengths[None, :] - lags[:, None], 0), axis=-1)
    wlags = np.sum(lengths) / samples
    return corr(delay(0), integrate_all(lags, wlags, method), trajs)


def c0_all_adj_ic(trajs, lags):
    lags = np.asarray(lags)
    lengths = np.array([len(traj) for traj in trajs])
    samples = np.sum(np.maximum(lengths[None, :] - lags[:, None], 0), axis=-1)
    wlags = 1.0 / samples
    weights = [_adj_all(len(traj), lags, wlags) for traj in trajs]
    return corr(delay(0), delay(0), trajs, weights)


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


# nonequilibrium IVAC


def c0_rt(trajs, weights):
    return corr(delay(0), delay(0), trajs, weights)


def ct_rt(trajs, lag, weights):
    return corr(delay(0), delay(lag), trajs, weights)


def c0_rt_adj_ct(trajs, lag, weights):
    return c0_rt_adj_ic(trajs, [lag], weights, "direct")


def ic_rt(trajs, lags, weights, method="fft"):
    return corr(delay(0), integrate(lags, method), trajs, weights)


def c0_rt_adj_ic(trajs, lags, weights, method):
    lags = np.asarray(lags)
    if method == "direct":
        weights = [_adj_rt_direct(weight, lags) for weight in weights]
    elif method == "fft":
        weights = [_adj_rt_fft(weight, lags) for weight in weights]
    else:
        raise ValueError("method must be 'direct' or 'fft'")
    return corr(delay(0), delay(0), trajs, weights)


@nb.njit
def _adj_rt_direct(weight, lags):
    result = np.zeros(len(weight))
    length = len(weight) - find_cutlag(weight)
    if length > 0:
        nlags = len(lags)
        for i in range(length):
            result[i] += weight[i] * nlags
        for n in range(nlags):
            lag = lags[n]
            for i in range(length):
                result[lag + i] += weight[i]
    return result


def _adj_rt_fft(weight, lags):
    result = np.zeros(len(weight))
    length = len(weight) - find_cutlag(weight)
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


# -----------------------------------------------------------------------------
# batch calculation of correlation matrices
#
# batch_compute_ic and batch_compute_c0 are the only routines that
# should be called by other code. They should select the correct
# implementations from the other batch_* and single correlation matrix
# functions based on the arguments.


def batch_compute_ic(trajs, params, *, weights=None, method="fft"):
    """Compute a batch of integrated correlated matrices.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
    params : 1d array-like or list of 1d array-like
        If a 1d array-like, list of VAC lag times.
        If a list of 1d array-like, list of IVAC lag times.
    weights : int or list of (n_frames[i],) ndarray
        Weight of trajectory starting at each configuration.
        If int, assume uniform weights except for the last int frames,
        which have zero weight.
    method : str, optional
        Method to compute integrated correlation matrices. Must be
        'direct', 'fft', or 'fft-all'. Methods 'direct' and 'fft'
        compute the matrices one by one using the compute_ic function.
        Method 'fft-all' computes all of the correlation matrices at
        once using a FFT convolution between each pair of features.

    Returns
    -------
    iterable of (n_features, n_features) ndarray
        Iterable of integrated correlation matrices.

    """

    if np.asarray(params[0]).ndim == 0:
        params = np.asarray(params)
        assert params.ndim == 1
        flag = True
    else:
        params = [np.asarray(param) for param in params]
        assert np.all([param.ndim == 1 for param in params])
        flag = False

    if method == "fft-all":
        if weights is None:
            if flag:
                return batch_ct_all(trajs, params)
            else:
                return batch_ic_all(trajs, params)
        else:
            if flag:
                return batch_ct_rt(trajs, params, weights)
            else:
                return batch_ic_rt(trajs, params, weights)

    # compute each matrix one by one
    return (
        compute_ic(trajs, param, weights=weights, method=method)
        for param in params
    )


def batch_compute_c0(trajs, *, params=None, weights=None, method="fft"):
    """Compute a batch of correlation matrices.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
    params : 1d array-like or list of 1d array-like, optional
        If provided, adjust the correlation matrix so that the constant
        eigenvector is an exact solution (up to numerical precision) of
        linear VAC or IVAC.
    weights : int or list of (n_frames[i],) ndarray, optional
        Weight of trajectory starting at each configuration.
        If int, assume uniform weights except for the last int frames,
        which have zero weight.
    method : str, optional
        Method to compute integrated correlation matrices. Must be
        'direct', 'fft', or 'fft-all'. Methods 'direct' and 'fft'
        compute the matrices one by one using the compute_ic function.
        Method 'fft-all' computes all of the correlation matrices at
        once using a FFT convolution between each pair of features.
        It is not yet implemented for the equilibrium case (when weights
        are not provided), in which case 'fft' is used instead.

    Returns
    -------
    iterable of (n_features, n_features) ndarray
        Iterable of correlation matrices.

    """

    if params is None:
        if weights is None:
            c0 = c0_all(trajs)
        else:
            c0 = c0_rt(trajs, weights)
        return itertools.repeat(c0)

    if np.asarray(params[0]).ndim == 0:
        params = np.asarray(params)
        assert params.ndim == 1
        flag = True
    else:
        params = [np.asarray(param) for param in params]
        assert np.all([param.ndim == 1 for param in params])
        flag = False

    if method == "fft-all":
        if weights is None:
            method = "fft"  # fall back to "fft"
        else:
            if flag:
                return batch_c0_rt_adj_ct(trajs, params, weights)
            else:
                return batch_c0_rt_adj_ic(trajs, params, weights)

    return (
        compute_c0(trajs, lags=param, weights=weights, method=method)
        for param in params
    )


# helper functions


def _batch_fft_all(x, y):
    conv = signal.fftconvolve(
        x[::-1, :, None], y[:, None, :], mode="full", axes=0
    )
    return conv[len(x) - 1 :]


def _batch_fft_trunc(x, y):
    conv = signal.fftconvolve(
        x[::-1, :, None], y[:, None, :], mode="valid", axes=0
    )
    return conv


def _batch_fft_trunc_adj(w, y):
    return signal.fftconvolve(
        w[::-1, None, None],
        y[:, :, None] * y[:, None, :],
        mode="valid",
        axes=0,
    )


# equilibrium IVAC


def batch_ct_all(trajs, lags):
    minlag = min(lags)
    nfeatures = get_nfeatures(trajs)

    lens = np.array([len(traj) for traj in trajs])
    wlags = 1.0 / np.sum(np.maximum(0, lens[None, :] - lags[:, None]), axis=-1)

    result = np.zeros((len(lags), nfeatures, nfeatures))
    for traj in trajs:
        conv = _batch_fft_all(traj[: len(traj) - minlag], traj[minlag:])
        result += conv[lags - minlag] * wlags[:, None, None]
    return result


def batch_ic_all(trajs, params):
    all_lags = np.concatenate(params)
    minlag = min(all_lags)
    nfeatures = get_nfeatures(trajs)

    lens = np.array([len(traj) for traj in trajs])
    wlags = []
    for lags in params:
        counts = np.sum(np.maximum(0, lens[None, :] - lags[:, None]), axis=-1)
        wlags.append(1.0 / counts)

    result = np.zeros((len(params), nfeatures, nfeatures))
    for traj in trajs:
        conv = _batch_fft_all(traj[: len(traj) - minlag], traj[minlag:])
        for n, lags in enumerate(params):
            result[n] += np.sum(
                conv[lags - minlag] * wlags[n][:, None, None], axis=0
            )
    return result


# nonequilibrium IVAC


def batch_ct_rt(trajs, lags, weights):
    minlag, maxlag = min(lags), max(lags)
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((len(lags), nfeatures, nfeatures))
    denom = 0.0
    if is_cutlag(weights):
        for traj in trajs:
            length = len(traj) - weights
            if length > 0:
                x = traj[:length]
                y = traj[minlag : length + maxlag]
                assert len(x) == length
                assert len(y) == length + maxlag - minlag
                conv = _batch_fft_trunc(x, y)
                assert len(conv) == maxlag - minlag + 1
                numer += conv[lags - minlag]
                denom += length
    else:
        for traj, weight in zip(trajs, weights):
            length = len(traj) - find_cutlag(weight)
            if length > 0:
                w = weight[:length]
                x = traj[:length]
                y = traj[minlag : length + maxlag]
                assert len(w) == length
                assert len(x) == length
                assert len(y) == length + maxlag - minlag
                conv = _batch_fft_trunc(x * w[:, None], y)
                assert len(conv) == maxlag - minlag + 1
                numer += conv[lags - minlag]
                denom += np.sum(w)
    return numer / denom


def batch_ic_rt(trajs, params, weights):
    all_lags = np.concatenate(params)
    minlag, maxlag = min(all_lags), max(all_lags)
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((len(params), nfeatures, nfeatures))
    denom = 0.0
    if is_cutlag(weights):
        for traj in trajs:
            length = len(traj) - weights
            if length > 0:
                x = traj[:length]
                y = traj[minlag : length + maxlag]
                assert len(x) == length
                assert len(y) == length + maxlag - minlag
                conv = _batch_fft_trunc(x, y)
                assert len(conv) == maxlag - minlag + 1
                for n, lags in enumerate(params):
                    numer[n] += np.sum(conv[lags - minlag], axis=0)
                denom += length
    else:
        for traj, weight in zip(trajs, weights):
            length = len(traj) - find_cutlag(weight)
            if length > 0:
                w = weight[:length]
                x = traj[:length]
                y = traj[minlag : length + maxlag]
                assert len(w) == length
                assert len(x) == length
                assert len(y) == length + maxlag - minlag
                conv = _batch_fft_trunc(x * w[:, None], y)
                assert len(conv) == maxlag - minlag + 1
                for n, lags in enumerate(params):
                    numer[n] += np.sum(conv[lags - minlag], axis=0)
                denom += np.sum(w)
    return numer / denom


def batch_c0_rt_adj_ct(trajs, lags, weights):
    maxlag = max(lags)
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((len(lags), nfeatures, nfeatures))
    denom = 0.0
    for traj, weight in zip(trajs, weights):
        length = len(traj) - find_cutlag(weight)
        if length > 0:
            w = weight[:length]
            y = traj[: length + maxlag]
            assert len(w) == length
            assert len(y) == length + maxlag
            conv = _batch_fft_trunc_adj(w, y)
            assert len(conv) == maxlag + 1
            numer += conv[0]
            numer += conv[lags]
            denom += 2.0 * np.sum(w)
    return numer / denom


def batch_c0_rt_adj_ic(trajs, params, weights):
    all_lags = np.concatenate(params)
    maxlag = max(all_lags)
    nfeatures = get_nfeatures(trajs)
    numer = np.zeros((len(params), nfeatures, nfeatures))
    denom = 0.0
    for traj, weight in zip(trajs, weights):
        length = len(traj) - find_cutlag(weight)
        if length > 0:
            w = weight[:length]
            y = traj[: length + maxlag]
            assert len(w) == length
            assert len(y) == length + maxlag
            conv = _batch_fft_trunc_adj(w, y)
            assert len(conv) == maxlag + 1
            numer += conv[0]
            for n, lags in enumerate(params):
                numer[n] += np.mean(conv[lags], axis=0)
            denom += 2.0 * np.sum(w)
    return numer / denom
