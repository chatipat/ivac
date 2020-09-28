import itertools
import numba as nb
import numpy as np
import warnings
from scipy import linalg, optimize
from . import utils


# -----------------------------------------------------------------------------
# linear VAC and IVAC


class LinearVAC:
    r"""Solve linear VAC at a given lag time.

    Linear VAC solves the equation

    .. math::

        C(\tau) v_i = \lambda_i C(0) v_i

    for eigenvalues :math:`\lambda_i`
    and eigenvector coefficients :math:`v_i`.

    The covariance matrices are given by

    .. math::

        C_{ij}(\tau) = E[\phi_i(x_t) \phi_j(x_{t+\tau})]

        C_{ij}(0) = E[\phi_i(x_t) \phi_j(x_t)]

    where :math:`\phi_i` are the input features
    and :math:`\tau` is the lag time parameter.

    Parameters
    ----------
    lag : int
        Lag time, in units of frames.
    nevecs : int, optional
        Number of eigenvectors (including the trivial eigenvector)
        to compute.
        If None, return the maximum possible number of eigenvectors
        (n_features).
    addones : bool, optional
        If True, add a feature of ones before solving VAC.
        This increases n_features by 1.
    reweight : bool, optional
        If True, reweight trajectories to equilibrium.
    adjust : bool, optional
        If True, adjust :math:`C(0)` to ensure that the trivial
        eigenvector is exactly solved.
    truncate : bool or int, optional
        Truncate trajectories so that :math:`C(t)` and :math:`C(0)`
        use the same number of data points.
        If int, this is the number of data points removed.
        If True, lag data points are removed.
        By default, trajectories are not truncated
        unless reweight is True.

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
        adjust=False,
        truncate=None,
    ):
        if truncate is None:
            truncate = reweight
        if truncate is True:
            truncate = lag
        if truncate is not False and truncate < lag:
            raise ValueError("truncate is less than lag")
        if reweight and truncate is False:
            raise ValueError(
                "reweighting is only supported for mode 'truncate'"
            )

        self.lag = lag
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self.truncate = truncate

    def fit(self, trajs, weights=None):
        """Compute VAC results from input trajectories.

        Calculate and store VAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.

        """
        trajs = utils.preprocess_trajs(trajs, addones=self.addones)

        if self.reweight:
            if weights is None:
                weights = _vac_weights(trajs, self.lag, self.truncate)
        else:
            if weights is not None:
                raise ValueError("weights provided but not reweighting")

        if self.truncate is False:
            ct = utils.ct_all(trajs, self.lag)
            if self.adjust:
                c0 = utils.c0_all_adj_ct(trajs, self.lag)
            else:
                c0 = utils.c0_all(trajs)
        else:
            ct = utils.ct_rt(trajs, self.lag, self.truncate, weights)
            if self.adjust:
                c0 = utils.c0_rt_adj_ct(
                    trajs, self.lag, self.truncate, weights
                )
            else:
                c0 = utils.c0_rt(trajs, self.truncate, weights)

        evals, evecs = utils.symeig(ct, c0)
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
        trajs = utils.preprocess_trajs(trajs, addones=self.addones)
        return utils.trajs_matmul(trajs, self.evecs[:, : self.nevecs])

    def _set_fit_data(self, cov, evals, evecs, its, trajs, weights):
        """Set fields computed by the fit method."""
        self.cov = cov
        self.evals = evals
        self.evecs = evecs
        self.its = its
        self.trajs = trajs
        self.weights = weights


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
        If None, return the maximum possible number of eigenvectors
        (n_features).
    addones : bool, optional
        If True, add a feature of ones before solving VAC.
        This increases n_features by 1.
    reweight : bool, optional
        If True, reweight trajectories to equilibrium.
    adjust : bool, optional
        If True, adjust :math:`C(0)` to ensure that the trivial
        eigenvector is exactly solved.
    truncate : bool or int, optional
        Truncate trajectories so that each :math:`C(t)` and :math:`C(0)`
        use the same number of data points.
        If int, this is the number of data points removed.
        If True, maxlag data points are removed.
        By default, trajectories are not truncated
        unless reweight is True.
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
        adjust=False,
        truncate=None,
        method="direct",
    ):
        if minlag > maxlag:
            raise ValueError("minlag must be less than or equal to maxlag")
        if (maxlag - minlag) % lagstep != 0:
            raise ValueError("lag time interval must be a multiple of lagstep")
        if truncate is None:
            truncate = reweight
        if truncate is True:
            truncate = maxlag
        if truncate is not False and truncate < maxlag:
            raise ValueError("truncate is less than maxlag")
        if reweight and truncate is False:
            raise ValueError(
                "reweighting is only supported for mode 'truncate'"
            )
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
        self.truncate = truncate
        self.method = method

    def fit(self, trajs, weights=None):
        """Compute IVAC results from input trajectories.

        Calculate and store IVAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.

        """
        trajs = utils.preprocess_trajs(trajs, addones=self.addones)

        if self.reweight:
            if weights is None:
                weights = _ivac_weights(
                    trajs, self.lags, self.truncate, mode=self.method
                )
        else:
            if weights is not None:
                raise ValueError("weights provided but not reweighting")

        if self.truncate is False:
            ic = utils.ic_all(trajs, self.lags, mode=self.method)
            if self.adjust:
                c0 = utils.c0_all_adj_ic(trajs, self.lags, mode=self.method)
            else:
                c0 = utils.c0_all(trajs)
        else:
            ic = utils.ic_rt(
                trajs, self.lags, self.truncate, weights, mode=self.method
            )
            if self.adjust:
                c0 = utils.c0_rt_adj_ic(
                    trajs, self.lags, self.truncate, weights, mode=self.method
                )
            else:
                c0 = utils.c0_rt(trajs, self.truncate, weights)

        evals, evecs = utils.symeig(ic, c0)
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
        trajs = utils.preprocess_trajs(trajs, addones=self.addones)
        return utils.trajs_matmul(trajs, self.evecs[:, : self.nevecs])

    def _set_fit_data(self, cov, evals, evecs, its, trajs, weights):
        """Set fields computed by the fit method."""
        self.cov = cov
        self.evals = evals
        self.evecs = evecs
        self.its = its
        self.trajs = trajs
        self.weights = weights


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
    lags : int
        Lag times, in units of frames.
    nevecs : int, optional
        Number of eigenvectors (including the trivial eigenvector)
        to compute.
        If None, use the maximum possible number of eigenvectors
        (n_features).
    addones : bool, optional
        If True, add a feature of ones before solving VAC.
        This increases n_features by 1.
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
        adjust=False,
        truncate=None,
        method="direct",
    ):
        maxlag = np.max(lags)
        if truncate is None:
            truncate = reweight
        if truncate is True:
            truncate = maxlag
        if truncate is not False and truncate < maxlag:
            raise ValueError("truncate is less than maxlag")
        if reweight and truncate is False:
            raise ValueError(
                "reweighting is only supported for mode 'truncate'"
            )
        if method not in ["direct", "fft-all"]:
            raise ValueError("method must be 'direct' or 'fft-all'")

        self.lags = lags
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self.truncate = truncate
        self.method = method

    def fit(self, trajs, weights=None):
        """Compute VAC results from input trajectories.

        Calculate and store VAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.

        """
        trajs = utils.preprocess_trajs(trajs, addones=self.addones)
        nfeatures = utils.get_nfeatures(trajs)
        nlags = len(self.lags)
        nevecs = self.nevecs
        if nevecs is None:
            nevecs = nfeatures

        if self.truncate is False:
            cutlag = None
        else:
            cutlag = self.truncate

        cts = utils.batch_compute_ic(
            trajs, self.lags, cutlag, weights, mode=self.method
        )
        if self.adjust:
            c0s = utils.batch_compute_c0(
                trajs, self.lags, cutlag, weights, mode=self.method
            )
        else:
            c0s = utils.batch_compute_c0(
                trajs, None, cutlag, weights, mode=self.method
            )

        self.evals = np.empty((nlags, nevecs))
        self.evecs = np.empty((nlags, nfeatures, nevecs))
        self.its = np.empty((nlags, nevecs))

        for n, (ct, c0, lag) in enumerate(zip(cts, c0s, self.lags)):
            evals, evecs = utils.symeig(ct, c0, nevecs)
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
    nevecs : int, optional
        Number of eigenvectors (including the trivial eigenvector)
        to compute.
        If None, use the maximum possible number of eigenvectors
        (n_features).
    addones : bool, optional
        If True, add a feature of ones before solving VAC.
        This increases n_features by 1.
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
        adjust=False,
        truncate=None,
        method="direct",
    ):
        if np.any(lags[1:] < lags[:-1]):
            raise ValueError("lags must be nondecreasing")
        if np.any((lags[1:] - lags[:-1]) % lagstep != 0):
            raise ValueError(
                "lags time intervals must be multiples of lagstep"
            )
        maxlag = np.max(lags)
        if truncate is None:
            truncate = reweight
        if truncate is True:
            truncate = maxlag
        if truncate is not False and truncate < maxlag:
            raise ValueError("truncate is less than maxlag")
        if reweight and truncate is False:
            raise ValueError(
                "reweighting is only supported for mode 'truncate'"
            )
        if method not in ["direct", "fft", "fft-all"]:
            raise ValueError("method must be 'direct', 'fft', or 'fft-all")

        self.lags = lags
        self.lagstep = lagstep
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self.truncate = truncate
        self.method = method

    def fit(self, trajs, weights=None):
        """Compute IVAC results from input trajectories.

        Calculate and store IVAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.

        """
        trajs = utils.preprocess_trajs(trajs, addones=self.addones)
        nfeatures = utils.get_nfeatures(trajs)
        nlags = len(self.lags)
        nevecs = self.nevecs
        if nevecs is None:
            nevecs = nfeatures

        if self.truncate is False:
            cutlag = None
        else:
            cutlag = self.truncate

        params = [
            np.arange(start + self.lagstep, end + 1, self.lagstep)
            for start, end in zip(self.lags[:-1], self.lags[1:])
        ]

        ics = list(
            utils.batch_compute_ic(
                trajs, params, cutlag, weights, mode=self.method
            )
        )
        if self.adjust:
            c0s = list(
                utils.batch_compute_c0(
                    trajs, params, cutlag, weights, mode=self.method
                )
            )
        else:
            c0 = utils.compute_c0(
                trajs, None, cutlag, weights, mode=self.method
            )
            denom = 1

        self.evals = np.full((nlags, nlags, nevecs), np.nan)
        self.evecs = np.full((nlags, nlags, nfeatures, nevecs), np.nan)
        self.its = np.full((nlags, nlags, nevecs), np.nan)

        for i in range(nlags):
            ic = utils.compute_ic(
                trajs, self.lags[i], cutlag, weights, mode=self.method
            )
            if self.adjust:
                c0 = utils.compute_c0(
                    trajs, self.lags[i], cutlag, weights, method=self.method
                )
                denom = 1
            evals, evecs = utils.symeig(ic, c0, nevecs)
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
                evals, evecs = utils.symeig(ic, c0 / denom, nevecs)
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
# projection distance


def projection_distance(u, v, weights=None, ortho=False):
    r"""Compute the projection distance between sets of features.

    The projection distance between subspaces :math:`U` and :math:`V` is

    .. math::

        d_F(U, V) =
            \lVert \mathrm{proj}[V^\bot] \mathrm{proj}[U] \rVert_F

    where :math:`V^\bot` is the orthogonal complement of :math:`V`.

    This can be intuited as how well :math:`V` can represent :math:`U`.

    Parameters
    ----------
    u, v : list of (n_frames[i], n_features) array-like
        Featurized trajectories.
    weights : list of (n_frames[i],) array-like, optional
        Weights for each frame of the trajectories.
        If None, weights are assumed to be uniform.
    ortho : bool, optional
        If True, orthonormalize the features
        before calculating the projection distance.
        If False, the features are assumed to be already orthonormal.

    Returns
    -------
    float
        Projection distance.

    """
    if ortho:
        u = orthonormalize(u)
        v = orthonormalize(v)
    cov = covmat(u, v, weights=weights)
    s = linalg.svdvals(cov)
    s = np.clip(s, 0.0, 1.0)
    return np.sqrt(len(cov) - np.sum(s ** 2))


def projection_distance_coeffs(u, v, cov=None, ortho=False):
    r"""Compute the projection distance between sets of features.

    The projection distance between subspaces :math:`U` and :math:`V` is

    .. math::

        d_F(U, V) =
            \lVert \mathrm{proj}[V^\bot] \mathrm{proj}[U] \rVert_F

    where :math:`V^\bot` is the orthogonal complement of :math:`V`.

    This can be intuited as how well :math:`V` can represent :math:`U`.

    In this function, :math:`u_{ij}` and :math:`v_{ij}` are coefficients
    of bases :math:`\phi_i` and :math:`\psi_i`, respectively.
    The corresponding features :math:`f_j` and :math:`g_j` are

    .. math::

        f_j = \sum_i u_{ij} \phi_i

        g_j = \sum_i v_{ij} \psi_i

    The covariance matrix between the bases is

    .. math::

        C_{ij} = \mathbf{E}[\phi_i \psi_j]

    Note that this function can orthonormalize the coefficients
    only when the two bases are identical.

    Parameters
    ----------
    u, v : (n_bases, n_features) ndarray
        Coefficients of the features.
        Featurized trajectories.
    cov : (n_bases, n_bases) ndarray
        Covariance matrix of the basis.
    ortho : bool, optional
        If True, orthonormalize the features
        before calculating the projection distance.
        This is valid only when u and v
        are coefficients of the same basis.
        If False, the features are assumed to be already orthonormal.

    Returns
    -------
    float
        Projection distance.

    """
    if ortho:
        u = orthonormalize_coeffs(u, cov)
        v = orthonormalize_coeffs(v, cov)
    if cov is None:
        cov = u.T @ v
    else:
        cov = u.T @ cov @ v
    s = linalg.svdvals(cov)
    s = np.clip(s, 0.0, 1.0)
    return np.sqrt(len(cov) - np.sum(s ** 2))


def orthonormalize(features, weights=None):
    r"""Orthonormalize features.

    Features are orthonormalized using the Cholesky decomposition
    in a way that is equivalent to performing the Gram-Schmidt process.
    If :math:`v_i` is an input feature, the corresponding output
    feature :math:`u_i` is given by the algorithm

    .. math::

        u_0 = v_0

        u_1 = v_1 - \mathrm{proj}_{u_0}(v_1)

        u_2 = v_2 - \mathrm{proj}_{u_0}(v_2) - \mathrm{proj}_{u_1}(v_2)

    Parameters
    ----------
    features : list of (n_frames[i], n_features) array-like
        List of featurized trajectories.
    weights : list of (n_frames[i],) array-like, optional
        Weights for each frame of the trajectories.
        If None, weights are assumed to be uniform.

    Returns
    -------
    list of (n_frames[i], n_features) array-like
        Orthonormalized features for the input trajectories.

    """
    features = utils.preprocess_trajs(features)
    cov = covmat(features, weights=weights)
    if np.allclose(cov, np.identity(len(cov))):
        return features
    u = linalg.cholesky(cov)
    uinv = linalg.inv(u)
    return utils.trajs_matmul(features, uinv)


def orthonormalize_coeffs(coeffs, cov=None):
    r"""Orthonormalize linear combination coefficients of basis vectors.

    The coefficients :math:`c_{ik}` represent the features

    .. math::

        v_k = \sum_i \phi_i c_{ik}

    where :math:`\phi_i` are features which form the basis.
    The auto-covariance matrix of the basis is

    .. math::
        \mathbf{C} = E[\phi_i \phi_j]

    Features are orthonormalized using the Cholesky decomposition
    in a way that is equivalent to performing the Gram-Schmidt process.
    If :math:`v_i` is an input feature, the corresponding output
    feature :math:`u_i` is given by the algorithm

    .. math::

        u_0 = v_0

        u_1 = v_1 - \mathrm{proj}_{u_0}(v_1)

        u_2 = v_2 - \mathrm{proj}_{u_0}(v_2) - \mathrm{proj}_{u_1}(v_2)

    Parameters
    ----------
    coeffs : (n_bases, n_features) ndarray
        Coefficients of the features.
    cov : (n_bases, n_bases) ndarray
        Covariance matrix of the basis.

    Returns
    -------
    (n_bases, n_features) ndarray
        Orthonormalized coefficients.

    """
    if cov is None:
        cov = coeffs.T @ coeffs
    else:
        cov = coeffs.T @ cov @ coeffs
    if np.allclose(cov, np.identity(len(cov))):
        return coeffs
    u = linalg.cholesky(cov)
    uinv = linalg.inv(u)
    return coeffs @ uinv


def covmat(u, v=None, weights=None):
    r"""Compute the correlation matrix between features.

    For a single trajectory with features
    :math:`\vec{u}_t`, :math:`\vec{v}_t` and weights :math:`w_t`
    at each frame :math:`n = 1, 2, \ldots, T`,
    this function calculates

    .. math::

        \mathbf{C} =
            \frac{\sum_{t=1}^{T} w_t \vec{u}_t \vec{v}_{t}^T}
            {\sum_{t=1}^{T} w_t}

    Parameters
    ----------
    u : list of (n_frames[i], n_features1) array-like
        First set of features for a list of trajectories.
    v : list of (n_frames[i], n_features2) array-like, optional
        Second set of features for a list of trajectories.
        If None, the first set of features is used.
    weights : list of (n_frames,) array-like, optional
        Weights for each frame of the trajectories.
        If None, weights are assumed to be uniform.

    Returns
    -------
    (n_features1, n_features2) ndarray
        Covariance matrix.

    """
    if v is None:
        v = u
    if len(u) != len(v):
        raise ValueError("mismatch in the number of trajectories")
    nfeatures1 = utils.get_nfeatures(u)
    nfeatures2 = utils.get_nfeatures(v)
    cov = np.zeros((nfeatures1, nfeatures2))
    count = 0.0
    if weights is None:
        for x, y in zip(u, v):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            cov += x.T @ y
            count += len(x)
    else:
        if len(u) != len(weights):
            raise ValueError("mismatch in the number of trajectories")
        for x, y, w in zip(u, v, weights):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            w = np.asarray(w, dtype=np.float64)
            cov += x.T @ (w[:, None] * y)
            count += np.sum(w)
    return cov / count


# -----------------------------------------------------------------------------
# reweighting


def _vac_weights(trajs, lag, truncate):
    """Estimate weights for VAC.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
        The features must be able to represent constant features.
    lag : int
        VAC lag time, in units of frames.
    truncate : int
        Number of frames to drop from the end of each trajectory.
        This must be greater than or equal to the VAC lag time.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Weight of trajectory starting at each configuration.

    """
    ct = utils.ct_rt(trajs, lag, truncate)
    c0 = utils.c0_rt(trajs, truncate)
    w = utils.solve_stationary(ct, c0)
    return _build_weights(trajs, w, truncate)


def _ivac_weights(trajs, lags, truncate, method="direct"):
    """Estimate weights for IVAC.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) ndarray
        List of featurized trajectories.
        The features must be able to represent constant features.
    lags : array-like of int
        Lag times at which to evaluate IVAC, in units of frames.
    truncate : int
        Number of frames to drop from the end of each trajectory.
        This must be greater than or equal to the maximum IVAC lag time.
    method : string, optional
        Method to use for calculating the integrated correlation matrix.
        Currently, 'direct' and 'fft' are supported.
        The default method, 'direct', is usually faster for smaller
        numbers of lag times. The speed of method 'fft' is mostly
        independent of the number of lag times used.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Weight of trajectory starting at each configuration.

    """
    ic = utils.ic_rt(trajs, lags, truncate, mode=method)
    c0 = utils.c0_rt(trajs, truncate)
    w = utils.solve_stationary(ic / len(lags), c0)
    return _build_weights(trajs, w, truncate)


def _build_weights(trajs, coeffs, truncate):
    """Build weights from reweighting coefficients."""
    weights = []
    total = 0.0
    for traj in trajs:
        weight = traj @ coeffs
        weight[len(traj) - truncate :] = 0.0
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
    for i, val in enumerate(evals):
        dlag = maxlag - minlag + lagstep
        avg = val * lagstep / dlag
        if avg >= 1.0:
            its[i] = np.inf
        elif avg > 0.0:
            guess = -2.0 * np.log(avg) / (minlag + maxlag)
            sol = optimize.root_scalar(
                _ivac_its_f_p,
                args=(val, minlag, dlag, lagstep),
                method="newton",
                x0=guess,
                fprime=True,
            )
            if sol.converged:
                its[i] = 1.0 / sol.root
            else:
                warnings.warn("implied timescale calculation did not converge")
    return its


@nb.njit
def _ivac_its_f_p(sigma, val, minlag, dlag, lagstep=1):
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

    """
    a = (
        np.exp(-sigma * minlag)
        * np.expm1(-sigma * dlag)
        / np.expm1(-sigma * lagstep)
    )
    b = (
        minlag
        + lagstep / np.expm1(sigma * lagstep)
        - dlag / np.expm1(sigma * dlag)
    )
    return a - val, -a * b
