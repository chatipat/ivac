import numba as nb
import numpy as np
import warnings
from scipy import optimize
from .utils import (
    preprocess_trajs,
    get_nfeatures,
    trajs_matmul,
    symeig,
    solve_stationary,
    compute_ic,
    compute_c0,
    batch_compute_ic,
    batch_compute_c0,
    is_cutlag,
)


# -----------------------------------------------------------------------------
# linear VAC and IVAC


class LinearVAC:
    r"""Solve linear VAC at a given lag time.

    Linear VAC solves the equation

    .. math::

        C(\tau) v_i = \lambda_i C(0) v_i

    for eigenvalues :math:`\lambda_i`
    and eigenvector coefficients :math:`v_i`.

    The correlation matrices are given by

    .. math::

        C_{ij}(\tau) = E[\phi_i(x_t) \phi_j(x_{t+\tau})]

        C_{ij}(0) = E[\phi_i(x_t) \phi_j(x_t)]

    where :math:`\phi_i` are the input features
    and :math:`\tau` is the lag time parameter.

    This implementation assumes that the constant feature can be
    represented by a linear combination of the other features.
    If this is not the case, addones=True will augment the input
    features with the constant feature.

    Parameters
    ----------
    lag : int
        Lag time, in units of frames.
    nevecs : int, optional
        Number of eigenvectors (including the trivial eigenvector)
        to compute.
        If None, use the maximum possible number of eigenvectors
        (n_features).
    addones : bool, optional
        If True, add a feature of ones before solving VAC.
        This increases n_features by 1.
        This should only be set to True if the constant feature
        is not contained within the span of the input features.
    reweight : bool, optional
        If True, reweight trajectories to equilibrium.
    adjust : bool, optional
        If True, adjust :math:`C(0)` to ensure that the trivial
        eigenvector is exactly solved.

    Attributes
    ----------
    lag : int
        VAC lag time in units of frames.
    evals : (n_evecs,) ndarray
        VAC eigenvalues in decreasing order.
        This includes the trivial eigenvalue.
    its : (n_evecs,) ndarray
        Implied timescales corresponding to the eigenvalues,
        in units of frames.
    evecs : (n_features, n_evecs) ndarray
        Coefficients of the VAC eigenvectors
        corresponding to the eigenvalues.
    cov : (n_features, n_features) ndarray
        Covariance matrix of the fitted data.
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories used to solve VAC.
    weights : list of (n_frames[i],) ndarray
        Equilibrium weight of trajectories starting at each configuration.

    """

    def __init__(
        self,
        lag,
        nevecs=None,
        addones=False,
        reweight=False,
        adjust=True,
    ):
        self.lag = lag
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self._isfit = False

    def fit(self, trajs, weights=None):
        """Compute VAC results from input trajectories.

        Calculate and store VAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.
        weights : int or list of (n_frames[i],) ndarray, optional
            If int, the number of frames to drop from the end of each
            trajectory, which must be greater than or equal to the VAC
            lag time. This is equivalent to passing a list of uniform
            weights but with the last int frames having zero weight.
            If a list of ndarray, the weight of the trajectory starting
            at each configuration. Note that the last frames of each
            trajectory must have zero weight. This number of ending
            frames with zero weight must be at least the VAC lag time.

        """
        trajs = preprocess_trajs(trajs, addones=self.addones)

        if self.reweight:
            if weights is None:
                weights = _ivac_weights(trajs, self.lag)
        else:
            if weights is not None:
                raise ValueError("weights provided but not reweighting")

        c0, evals, evecs = _solve_ivac(
            trajs,
            self.lag,
            weights=weights,
            adjust=self.adjust,
        )

        its = _vac_its(evals, self.lag)
        self._set_fit_data(c0, evals, evecs, its, trajs, weights)

    def transform(self, trajs):
        """Compute VAC eigenvectors on the input trajectories.

        Use the fitted VAC eigenvector coefficients to calculate
        the values of the VAC eigenvectors on the input trajectories.

        Parameters
        ----------
        trajs : list of (traj_len[i], n_features) ndarray
            List of featurized trajectories.

        Returns
        -------
        list of (traj_len[i], n_evecs) ndarray
            VAC eigenvectors at each frame of the input trajectories.

        """
        trajs = preprocess_trajs(trajs, addones=self.addones)
        return trajs_matmul(trajs, self.evecs[:, : self.nevecs])

    def _set_fit_data(self, cov, evals, evecs, its, trajs, weights):
        """Set fields computed by the fit method."""
        self._isfit = True
        self._cov = cov
        self._evals = evals
        self._evecs = evecs
        self._its = its
        self._trajs = trajs
        self._weights = weights

    @property
    def cov(self):
        if self._isfit:
            return self._cov
        raise ValueError("object has not been fit to data")

    @property
    def evals(self):
        if self._isfit:
            return self._evals
        raise ValueError("object has not been fit to data")

    @property
    def evecs(self):
        if self._isfit:
            return self._evecs
        raise ValueError("object has not been fit to data")

    @property
    def its(self):
        if self._isfit:
            return self._its
        raise ValueError("object has not been fit to data")

    @property
    def trajs(self):
        if self._isfit:
            return self._trajs
        raise ValueError("object has not been fit to data")

    @property
    def weights(self):
        if self._isfit:
            return self._weights
        raise ValueError("object has not been fit to data")


class LinearIVAC:
    r"""Solve linear IVAC for a given range of lag times.

    Linear IVAC solves the equation

    .. math::

        \sum_\tau C(\tau) v_i = \lambda_i C(0) v_i

    for eigenvalues :math:`\lambda_i`
    and eigenvector coefficients :math:`v_i`.

    The covariance matrices are given by

    .. math::

        C_{ij}(\tau) = E[\phi_i(x_t) \phi_j(x_{t+\tau})]

        C_{ij}(0) = E[\phi_i(x_t) \phi_j(x_t)]

    where :math:`\phi_i` are the input features
    and :math:`\tau` is the lag time parameter.

    This implementation assumes that the constant feature can be
    represented by a linear combination of the other features.
    If this is not the case, addones=True will augment the input
    features with the constant feature.

    Parameters
    ----------
    minlag : int
        Minimum lag time in units of frames.
    maxlag : int
        Maximum lag time (inclusive) in units of frames.
        If minlag == maxlag, this is equivalent to VAC.
    lagstep : int, optional
        Number of frames between each lag time.
        This must evenly divide maxlag - minlag.
        The integrated covariance matrix is computed using lag times
        (minlag, minlag + lagstep, ..., maxlag)
    nevecs : int, optional
        Number of eigenvectors (including the trivial eigenvector)
        to compute.
        If None, use the maximum possible number of eigenvectors
        (n_features).
    addones : bool, optional
        If True, add a feature of ones before solving VAC.
        This increases n_features by 1.
    reweight : bool, optional
        If True, reweight trajectories to equilibrium.
    adjust : bool, optional
        If True, adjust :math:`C(0)` to ensure that the trivial
        eigenvector is exactly solved.
    method : str, optional
        Method to compute the integrated covariance matrix.
        Currently, 'direct', 'fft' are supported.
        Both 'direct' and 'fft' integrate features over lag times before
        computing the correlation matrix.
        Method 'direct' does so by summing the time-lagged features.
        Its runtime increases linearly with the number of lag times.
        Method 'fft' does so by performing an FFT convolution.
        It takes around the same amount of time to run regardless
        of the number of lag times, and is faster than 'direct' when
        there is more than around 100 lag times.

    Attributes
    ----------
    minlag : int
        Minimum IVAC lag time in units of frames.
    maxlag : int
        Maximum IVAC lag time in units of frames.
    lagstep : int
        Interval between IVAC lag times, in units of frames.
    evals : (n_evecs,) ndarray
        IVAC eigenvalues in decreasing order.
        This includes the trivial eigenvalue.
    its : (n_evecs,) ndarray
        Implied timescales corresponding to the eigenvalues,
        in units of frames.
    evecs : (n_features, n_evecs) ndarray
        Coefficients of the IVAC eigenvectors
        corresponding to the eigenvalues.
    cov : (n_features, n_features) ndarray
        Covariance matrix of the fitted data.
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories used to solve IVAC.
    weights : list of (n_frames[i],) ndarray
        Equilibrium weight of trajectories starting at each configuration.

    """

    def __init__(
        self,
        minlag,
        maxlag,
        lagstep=1,
        nevecs=None,
        addones=False,
        reweight=False,
        adjust=True,
        method="fft",
    ):
        if minlag > maxlag:
            raise ValueError("minlag must be less than or equal to maxlag")
        if (maxlag - minlag) % lagstep != 0:
            raise ValueError("lag time interval must be a multiple of lagstep")
        if method not in ["direct", "fft"]:
            raise ValueError("method must be 'direct', or 'fft'")

        self.minlag = minlag
        self.maxlag = maxlag
        self.lagstep = lagstep
        self.lags = np.arange(self.minlag, self.maxlag + 1, self.lagstep)
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self.method = method
        self._isfit = False

    def fit(self, trajs, weights=None):
        """Compute IVAC results from input trajectories.

        Calculate and store IVAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.
        weights : int or list of (n_frames[i],) ndarray, optional
            If int, the number of frames to drop from the end of each
            trajectory, which must be greater than or equal to the
            maximum IVAC lag time. This is equivalent to passing a list
            of uniform weights but with the last int frames having zero
            weight.
            If a list of ndarray, the weight of the trajectory starting
            at each configuration. Note that the last frames of each
            trajectory must have zero weight. This number of ending
            frames with zero weight must be at least the maximum IVAC
            lag time.

        """
        trajs = preprocess_trajs(trajs, addones=self.addones)

        if self.reweight:
            if weights is None:
                weights = _ivac_weights(trajs, self.lags, method=self.method)
        else:
            if weights is not None:
                raise ValueError("weights provided but not reweighting")

        c0, evals, evecs = _solve_ivac(
            trajs,
            self.lags,
            weights=weights,
            adjust=self.adjust,
            method=self.method,
        )
        its = _ivac_its(evals, self.minlag, self.maxlag, self.lagstep)
        self._set_fit_data(c0, evals, evecs, its, trajs, weights)

    def transform(self, trajs):
        """Compute IVAC eigenvectors on the input trajectories.

        Use the fitted IVAC eigenvector coefficients to calculate
        the values of the IVAC eigenvectors on the input trajectories.

        Parameters
        ----------
        trajs : list of (traj_len[i], n_features) ndarray
            List of featurized trajectories.

        Returns
        -------
        list of (traj_len[i], n_evecs) ndarray
            IVAC eigenvectors at each frame of the input trajectories.

        """
        trajs = preprocess_trajs(trajs, addones=self.addones)
        return trajs_matmul(trajs, self.evecs[:, : self.nevecs])

    def _set_fit_data(self, cov, evals, evecs, its, trajs, weights):
        """Set fields computed by the fit method."""
        self._isfit = True
        self._cov = cov
        self._evals = evals
        self._evecs = evecs
        self._its = its
        self._trajs = trajs
        self._weights = weights

    @property
    def cov(self):
        if self._isfit:
            return self._cov
        raise ValueError("object has not been fit to data")

    @property
    def evals(self):
        if self._isfit:
            return self._evals
        raise ValueError("object has not been fit to data")

    @property
    def evecs(self):
        if self._isfit:
            return self._evecs
        raise ValueError("object has not been fit to data")

    @property
    def its(self):
        if self._isfit:
            return self._its
        raise ValueError("object has not been fit to data")

    @property
    def trajs(self):
        if self._isfit:
            return self._trajs
        raise ValueError("object has not been fit to data")

    @property
    def weights(self):
        if self._isfit:
            return self._weights
        raise ValueError("object has not been fit to data")


def _solve_ivac(
    trajs,
    lags,
    *,
    weights=None,
    adjust=True,
    method="fft",
):
    """Solve IVAC with the given parameters.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
    lags : int or 1d array-like of int
        VAC lag time or IVAC lag times, in units of frames.
        For IVAC, this should be a list of lag times that will be used,
        not the 2 or 3 values specifying the range.
    weights : int or list of (n_frames[i],) ndarray, optional
        If int, the number of frames to drop from the end of each
        trajectory, which must be greater than or equal to the maximum
        IVAC lag time. This is equivalent to passing a list of uniform
        weights but with the last int frames having zero weight.
        If a list of ndarray, the weight of the trajectory starting at
        each configuration. Note that the last frames of each trajectory
        must have zero weight. This number of ending frames with zero
        weight must be at least the maximum IVAC lag time.
    adjust : bool, optional
        If True, adjust :math:`C(0)` to ensure that the trivial
        eigenvector is exactly solved.
    method : str, optional
        Method to compute the integrated covariance matrix.
        Currently, 'direct', 'fft' are supported.
        Both 'direct' and 'fft' integrate features over lag times before
        computing the correlation matrix.
        Method 'direct' does so by summing the time-lagged features.
        Its runtime increases linearly with the number of lag times.
        Method 'fft' does so by performing an FFT convolution.
        It takes around the same amount of time to run regardless
        of the number of lag times, and is faster than 'direct' when
        there is more than around 100 lag times.

    """
    ic = compute_ic(trajs, lags, weights=weights, method=method)

    if adjust:
        c0 = compute_c0(trajs, lags=lags, weights=weights, method=method)
    else:
        c0 = compute_c0(trajs, weights=weights, method=method)

    evals, evecs = symeig(ic, c0)
    return c0, evals, evecs


# -----------------------------------------------------------------------------
# linear VAC and IVAC scans


class LinearVACScan:
    """Solve linear VAC at each given lag time.

    This class provides a more optimized way of solving linear VAC at a
    set of lag times with the same input trajectories. The code

    .. code-block:: python

        scan = LinearVACScan(lags)
        vac = scan[lags[i]]

    is equivalent to

    .. code-block:: python

        vac = LinearVAC(lags[i])

    Parameters
    ----------
    lag : int
        Lag time, in units of frames.
    nevecs : int, optional
        Number of eigenvectors (including the trivial eigenvector)
        to compute.
        If None, use the maximum possible number of eigenvectors
        (n_features).
    addones : bool, optional
        If True, add a feature of ones before solving VAC.
        This increases n_features by 1.
        This should only be set to True if the constant feature
        is not contained within the span of the input features.
    reweight : bool, optional
        If True, reweight trajectories to equilibrium.
    adjust : bool, optional
        If True, adjust :math:`C(0)` to ensure that the trivial
        eigenvector is exactly solved.
    method : str, optional
        Method used to compute the time lagged covariance matrices.
        Currently supported methods are 'direct',
        which computes each time lagged covariance matrix separately,
        and 'fft-all', which computes all time-lagged correlation
        matrices at once by convolving each pair of features.
        The runtime of 'fft-all' is almost independent of the number
        of lag times, and is faster then 'direct' when scanning a
        large number of lag times.

    Attributes
    ----------
    lags : 1d array-like of int
        VAC lag time, in units of frames.
    cov : (n_features, n_features) ndarray
        Covariance matrix of the fitted data.

    """

    def __init__(
        self,
        lags,
        nevecs=None,
        addones=False,
        reweight=False,
        adjust=True,
        method="direct",
    ):
        maxlag = np.max(lags)
        if method not in ["direct", "fft-all"]:
            raise ValueError("method must be 'direct' or 'fft-all'")

        self.lags = lags
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self.method = method

    def fit(self, trajs, weights=None):
        """Compute VAC results from input trajectories.

        Calculate and store VAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.
        weights : int or list of (n_frames[i],) ndarray, optional
            If int, the number of frames to drop from the end of each
            trajectory, which must be greater than or equal to the VAC
            lag time. This is equivalent to passing a list of uniform
            weights but with the last int frames having zero weight.
            If a list of ndarray, the weight of the trajectory starting
            at each configuration. Note that the last frames of each
            trajectory must have zero weight. This number of ending
            frames with zero weight must be at least the VAC lag time.

        """
        trajs = preprocess_trajs(trajs, addones=self.addones)
        nfeatures = get_nfeatures(trajs)
        nlags = len(self.lags)
        nevecs = self.nevecs
        if nevecs is None:
            nevecs = nfeatures

        cts = batch_compute_ic(
            trajs,
            self.lags,
            weights=weights,
            method=self.method,
        )
        if self.adjust:
            c0s = batch_compute_c0(
                trajs,
                lags=self.lags,
                weights=weights,
                method=self.method,
            )
        else:
            c0s = batch_compute_c0(
                trajs,
                weights=weights,
                method=self.method,
            )

        self.evals = np.empty((nlags, nevecs))
        self.evecs = np.empty((nlags, nfeatures, nevecs))
        self.its = np.empty((nlags, nevecs))

        for n, (ct, c0, lag) in enumerate(zip(cts, c0s, self.lags)):
            evals, evecs = symeig(ct, c0, nevecs)
            self.evals[n] = evals
            self.evecs[n] = evecs
            self.its[n] = _vac_its(evals, lag)

        if self.adjust:
            self.cov = None
        else:
            self.cov = c0
        self.trajs = trajs
        self.weights = weights

    def __getitem__(self, lag):
        """Get a fitted LinearVAC with the specified lag time.

        Parameters
        ----------
        lag : int
            Lag time, in units of frames.

        Returns
        -------
        LinearVAC
            Fitted LinearVAC instance.

        """
        i = np.argwhere(self.lags == lag)[0, 0]
        vac = LinearVAC(lag, nevecs=self.nevecs, addones=self.addones)
        vac._set_fit_data(
            self.cov,
            self.evals[i],
            self.evecs[i],
            self.its[i],
            self.trajs,
            self.weights,
        )
        return vac


class LinearIVACScan:
    """Solve linear IVAC for each pair of lag times.

    This class provides a more optimized way of solving linear IVAC
    with the same input trajectories
    for all intervals within a set of lag times,
    The code

    .. code-block:: python

        scan = LinearIVACScan(lags)
        ivac = scan[lags[i], lags[j]]

    is equivalent to

    .. code-block:: python

        ivac = LinearVAC(lags[i], lags[j])

    Parameters
    ----------
    lags : int
        Lag times, in units of frames.
    lagstep : int, optional
        Number of frames between each lag time.
        This must evenly divide maxlag - minlag.
        The integrated covariance matrix is computed using lag times
        (minlag, minlag + lagstep, ..., maxlag)
    nevecs : int, optional
        Number of eigenvectors (including the trivial eigenvector)
        to compute.
        If None, use the maximum possible number of eigenvectors
        (n_features).
    addones : bool, optional
        If True, add a feature of ones before solving VAC.
        This increases n_features by 1.
    reweight : bool, optional
        If True, reweight trajectories to equilibrium.
    adjust : bool, optional
        If True, adjust :math:`C(0)` to ensure that the trivial
        eigenvector is exactly solved.
    method : str, optional
        Method to compute the integrated covariance matrix.
        Currently, 'direct', 'fft', and 'fft-all' are supported.
        Both 'direct' and 'fft' integrate features over lag times before
        computing the correlation matrix. They scale linearly with
        the number of parameter sets.
        Method 'direct' does so by summing the time-lagged features.
        Its runtime increases linearly with the number of lag times.
        Method 'fft' does so by performing an FFT convolution.
        It takes around the same amount of time to run regardless
        of the number of lag times, and is faster than 'direct' when
        there is more than around 100 lag times.
        Method 'fft-all' computes all time-lagged correlation matrices
        at once by convolving each pair of features, before summing
        up those correlation matrices to obtain integrated correlation
        matrices. It is the slowest of these methods for calculating
        a few sets of parameters, but is almost independent of the
        number of lag times or parameter sets.

    Attributes
    ----------
    lags : 1d array-like of int
        VAC lag time, in units of frames.
    cov : (n_features, n_features) ndarray
        Covariance matrix of the fitted data.

    """

    def __init__(
        self,
        lags,
        lagstep=1,
        nevecs=None,
        addones=False,
        reweight=False,
        adjust=True,
        method="fft",
    ):
        if np.any(lags[1:] < lags[:-1]):
            raise ValueError("lags must be nondecreasing")
        if np.any((lags[1:] - lags[:-1]) % lagstep != 0):
            raise ValueError(
                "lags time intervals must be multiples of lagstep"
            )
        maxlag = np.max(lags)
        if method not in ["direct", "fft", "fft-all"]:
            raise ValueError("method must be 'direct', 'fft', or 'fft-all")

        self.lags = lags
        self.lagstep = lagstep
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self.method = method

    def fit(self, trajs, weights=None):
        """Compute IVAC results from input trajectories.

        Calculate and store IVAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.
        weights : int or list of (n_frames[i],) ndarray, optional
            If int, the number of frames to drop from the end of each
            trajectory, which must be greater than or equal to the
            maximum IVAC lag time. This is equivalent to passing a list
            of uniform weights but with the last int frames having zero
            weight.
            If a list of ndarray, the weight of the trajectory starting
            at each configuration. Note that the last frames of each
            trajectory must have zero weight. This number of ending
            frames with zero weight must be at least the maximum IVAC
            lag time.

        """
        trajs = preprocess_trajs(trajs, addones=self.addones)
        nfeatures = get_nfeatures(trajs)
        nlags = len(self.lags)
        nevecs = self.nevecs
        if nevecs is None:
            nevecs = nfeatures

        params = [
            np.arange(start + self.lagstep, end + 1, self.lagstep)
            for start, end in zip(self.lags[:-1], self.lags[1:])
        ]

        ics = list(
            batch_compute_ic(
                trajs,
                params,
                weights=weights,
                method=self.method,
            )
        )
        if self.adjust:
            c0s = list(
                batch_compute_c0(
                    trajs,
                    params,
                    weights=weights,
                    method=self.method,
                )
            )
        else:
            c0 = compute_c0(trajs, weights=weights, method=self.method)
            denom = 1

        self.evals = np.full((nlags, nlags, nevecs), np.nan)
        self.evecs = np.full((nlags, nlags, nfeatures, nevecs), np.nan)
        self.its = np.full((nlags, nlags, nevecs), np.nan)

        for i in range(nlags):
            ic = compute_ic(
                trajs,
                self.lags[i],
                weights=weights,
                method=self.method,
            )
            if self.adjust:
                c0 = compute_c0(
                    trajs,
                    lags=self.lags[i],
                    weights=weights,
                    method=self.method,
                )
                denom = 1
            evals, evecs = symeig(ic, c0, nevecs)
            if self.lags[i] > 0:
                self.evals[i, i] = evals
                self.evecs[i, i] = evecs
                self.its[i, i] = _ivac_its(
                    evals, self.lags[i], self.lags[i], self.lagstep
                )
            for j in range(i + 1, nlags):
                ic += ics[j - 1]
                if self.adjust:
                    count = (self.lags[j] - self.lags[j - 1]) // self.lagstep
                    c0 += c0s[j - 1] * count
                    denom += count
                evals, evecs = symeig(ic, c0 / denom, nevecs)
                self.evals[i, j] = evals
                self.evecs[i, j] = evecs
                self.its[i, j] = _ivac_its(
                    evals, self.lags[i], self.lags[j], self.lagstep
                )

        if self.adjust:
            self.cov = c0
        else:
            self.cov = None
        self.trajs = trajs
        self.weights = weights

    def __getitem__(self, lags):
        """Get a fitted LinearIVAC with the specified lag times.

        Parameters
        ----------
        lags : Tuple[int, int]
            Minimum and maximum lag times, in units of frames.

        Returns
        -------
        LinearIVAC
            Fitted LinearIVAC instance.

        """
        minlag, maxlag = lags
        i = np.argwhere(self.lags == minlag)[0, 0]
        j = np.argwhere(self.lags == maxlag)[0, 0]
        ivac = LinearIVAC(
            minlag,
            maxlag,
            lagstep=self.lagstep,
            nevecs=self.nevecs,
            addones=self.addones,
        )
        ivac._set_fit_data(
            self.cov,
            self.evals[i, j],
            self.evecs[i, j],
            self.its[i, j],
            self.trajs,
            self.weights,
        )
        return ivac


# -----------------------------------------------------------------------------
# reweighting


def _ivac_weights(trajs, lags, weights=None, method="fft"):
    """Estimate weights for IVAC.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
        The features must be able to represent constant features.
    lags : array-like of int
        Lag times at which to evaluate IVAC, in units of frames.
    weights : int or list of (n_frames[i],) ndarray, optional
        If int, the number of frames to drop from the end of each
        trajectory, which must be greater than or equal to the maximum
        IVAC lag time. This is equivalent to passing a list of uniform
        weights but with the last int frames having zero weight.
        If a list of ndarray, the weight of the trajectory starting at
        each configuration. Note that the last frames of each trajectory
        must have zero weight. This number of ending frames with zero
        weight must be at least the maximum IVAC lag time.
    method : string, optional
        Method to use for calculating the integrated correlation matrix.
        Currently, 'direct' and 'fft' are supported. Method 'direct', is
        usually faster for smaller numbers of lag times. The speed of
        method 'fft' is mostly independent of the number of lag times
        used.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Weight of trajectory starting at each configuration.

    """
    lags = np.atleast_1d(lags)
    assert lags.ndim == 1
    if weights is None:
        weights = np.max(lags)
    elif is_cutlag(weights):
        assert weights >= np.max(lags)
    ic = compute_ic(trajs, lags, weights=weights, method=method)
    c0 = compute_c0(trajs, weights=weights)
    w = solve_stationary(ic / len(lags), c0)
    return _build_weights(trajs, w, weights)


def _build_weights(trajs, coeffs, old_weights):
    """Build weights from reweighting coefficients.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
    coeffs : (n_features,) ndarray
        Expansion coefficients of the new weights.
    old_weights : list of (n_frames[i],) ndarray
        Initial weight of trajectory starting at each configuration,
        which was used to estimate the expansion coefficients.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Weight of trajectory starting at each configuration.
    """
    weights = []
    total = 0.0
    if is_cutlag(old_weights):
        for traj in trajs:
            weight = traj @ coeffs
            weight[len(traj) - old_weights :] = 0.0
            total += np.sum(weight)
            weights.append(weight)
    else:
        for traj, old_weight in zip(trajs, old_weights):
            weight = traj @ coeffs
            weight *= old_weight
            total += np.sum(weight)
            weights.append(weight)
    # normalize weights so that their sum is 1
    for weight in weights:
        weight /= total
    return weights


# -----------------------------------------------------------------------------
# implied timescales


def _vac_its(evals, lag):
    """Calculate implied timescales from VAC eigenvalues.

    Parameters
    ----------
    evals : (n_evecs,) array-like
        VAC eigenvalues.
    lag : int
        VAC lag time in units of frames.

    Returns
    -------
    (n_evecs,) ndarray
        Estimated implied timescales.
        This is NaN when the VAC eigenvalues are negative.

    """
    its = np.full(len(evals), np.nan)
    its[evals >= 1.0] = np.inf
    mask = np.logical_and(0.0 < evals, evals < 1.0)
    its[mask] = -lag / np.log(evals[mask])
    return its


def _ivac_its(evals, minlag, maxlag, lagstep=1):
    """Calculate implied timescales from IVAC eigenvalues.

    Parameters
    ----------
    evals : (n_evecs,) array-like
        IVAC eigenvalues.
    minlag, maxlag : int
        Minimum and maximum lag times (inclusive) in units of frames.
    lagstep : int, optional
        Number of frames between adjacent lag times.
        Lag times are given by minlag, minlag + lagstep, ..., maxlag.

    Returns
    -------
    (n_evecs,) ndarray
        Estimated implied timescales.
        This is NaN when the IVAC eigenvalues are negative
        or when the calculation did not converge.

    """
    its = np.full(len(evals), np.nan)

    if minlag == 0:
        # remove component corresponding to zero lag time
        evals = evals - 1.0
        minlag = lagstep

    for i, val in enumerate(evals):
        dlag = maxlag - minlag + lagstep
        nlags = dlag / lagstep
        assert nlags > 0
        avg = val / nlags
        if avg >= 1.0:
            its[i] = np.inf
        elif avg > 0.0:
            # eigenvalues are bound by
            #   exp(-sigma * tmin) <= eval
            # and
            #   nlags * exp(-sigma * tmax) <= eval <= nlags * exp(-sigma * tmin)
            lower = max(
                0.0,
                -np.log(val) / minlag,
                -np.log(avg) / maxlag,
            )
            upper = -np.log(avg) / minlag

            # make sure solution is inside bracket
            lower *= 0.999
            upper *= 1.001

            sol = optimize.root_scalar(
                _ivac_its_f,
                args=(val, minlag, dlag, lagstep),
                method="brentq",
                bracket=[lower, upper],
            )
            if sol.converged and sol.root > 0.0:
                its[i] = 1.0 / sol.root
            else:
                warnings.warn("implied timescale calculation did not converge")
    return its


@nb.njit
def _ivac_its_f(sigma, val, minlag, dlag, lagstep=1):
    """Objective function for IVAC implied timescale calculation.

    Parameters
    ----------
    sigma : float
        Inverse implied timescale.
    val : float
        IVAC eigenvalue.
    minlag : int
        Minimum lag time in units of frames.
    dlag : int
        Number of frames in the interval from the minimum lag time
        to the maximum lag time (inclusive).
    lagstep : int, optional
        Number of frames between adjacent lag times.
        Lag times are given by minlag, minlag + lagstep, ..., maxlag.

    Returns
    -------
    float
        Difference between given and predicted IVAC eigenvalue.

    """
    return (
        np.exp(-sigma * minlag)
        * np.expm1(-sigma * dlag)
        / np.expm1(-sigma * lagstep)
    ) - val
