import numpy as np
import numba as nb
import warnings
from scipy import linalg, optimize, signal
from . import utils


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

        self.lag = lag
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self.truncate = truncate

    def fit(self, trajs):
        """Compute VAC results from input trajectories.

        Calculate and store VAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.

        """
        trajs = utils.preprocess_trajs(trajs)
        if self.addones:
            trajs = _addones(trajs)

        f0 = utils.delay(0)
        f1 = utils.delay(self.lag)

        if self.truncate is False:

            if self.reweight:
                raise ValueError(
                    "reweighting is only supported for mode 'truncate'"
                )

            ct = utils.corr(f0, f1, trajs, self.lag)
            if self.adjust:
                c0 = 0.5 * (
                    utils.corr(f0, f0, trajs, self.lag)
                    + utils.corr(f1, f1, trajs, self.lag)
                )
            else:
                c0 = utils.corr(f0, f0, trajs)

        else:

            w = None
            if self.reweight:
                ct = utils.corr(f0, f1, trajs, self.truncate)
                c0 = utils.corr(f0, f0, trajs, self.truncate)

                evals, evecs = linalg.eig(ct.T, c0)
                order = np.argsort(np.abs(evals))[::-1]
                evals = evals[order]
                evecs = evecs[:, order]
                w = np.real_if_close(evecs[:, 0])

                if not np.isclose(evals[0], 1.0):
                    warnings.warn("dominant eigenvalue is not 1")
                if not np.all(np.abs(evals[1:]) < 1.0):
                    warnings.warn(
                        "multiple eigenvalues have magnitudes above 1"
                    )
                if not np.isrealobj(w):
                    warnings.warn("estimated weights are complex")

            ct = utils.corr(f0, f1, trajs, self.truncate, w)
            if self.adjust:
                c0 = 0.5 * (
                    utils.corr(f0, f0, trajs, self.truncate, w)
                    + utils.corr(f1, f1, trajs, self.truncate, w)
                )
            else:
                c0 = utils.corr(f0, f0, trajs, self.truncate, w)

        evals, evecs = linalg.eigh(_sym(ct), c0)
        self.cov = c0
        self.evals = evals[::-1]
        self.evecs = evecs[:, ::-1]
        self.its = _vac_its(self.evals, self.lag)

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
        if self.addones:
            trajs = _addones(trajs)
        result = []
        for traj in trajs:
            traj = np.asarray(traj, dtype=np.float64)
            result.append(traj @ self.evecs[:, : self.nevecs])
        return result


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
        Currently, 'direct', 'conv', 'fftconv', and 'fft' are supported.
        Method 'direct' sums the time lagged covariance matrices.
        Method 'conv' integrates features over lag times before
        computing the covariance matrices.
        Method 'fftconv' does the same as 'conv', but uses a FFT.
        Method 'fft' uses a FFT to compute the time lagged correlation
        for each pair of basis functions.

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
        method="conv",
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
        self.minlag = minlag
        self.maxlag = maxlag
        self.lagstep = lagstep
        self.nevecs = nevecs
        self.addones = addones
        self.reweight = reweight
        self.adjust = adjust
        self.truncate = truncate
        self.method = method

    def fit(self, trajs):
        """Compute IVAC results from input trajectories.

        Calculate and store IVAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.

        """
        trajs = utils.preprocess_trajs(trajs)
        if self.addones:
            trajs = _addones(trajs)

        if self.method in ["direct", "fft"]:

            if self.truncate is not False:
                raise ValueError(
                    "truncate is only supported with method 'conv'"
                )
            if self.reweight:
                raise ValueError(
                    "reweight is only supported with method 'conv'"
                )
            if self.adjust:
                raise ValueError("adjust is only supported with method 'conv'")

            lags = np.arange(self.minlag, self.maxlag + 1, self.lagstep)
            c0 = covmat(trajs)
            ic = _icov(trajs, lags=lags, method=self.method)

        elif self.method in ["conv", "fftconv"]:
            mode = {"conv": "direct", "fftconv": "fft"}[self.method]

            if self.truncate is False:
                if self.reweight:
                    raise ValueError(
                        "reweighting is only supported with mode 'truncate'"
                    )

                lengths = np.array([len(traj) for traj in trajs])

                f0 = utils.delay(0)
                f1 = utils.integrate_all(
                    self.minlag, self.maxlag, self.lagstep, lengths, mode=mode
                )

                fw = None
                if self.adjust:
                    fw = utils.adjust_all(
                        self.minlag, self.maxlag, self.lagstep, lengths
                    )

                ic = utils.corr(f0, f1, trajs)
                c0 = utils.corr(f0, f0, trajs, weights=fw)

            else:

                f0 = utils.delay(0)
                f1 = utils.integrate(
                    self.minlag, self.maxlag, self.lagstep, mode=mode
                )

                w = None
                if self.reweight:
                    ic = utils.corr(f0, f1, trajs, self.truncate)
                    c0 = utils.corr(f0, f0, trajs, self.truncate)

                    evals, evecs = linalg.eig(ic.T, c0)
                    order = np.argsort(np.abs(evals))[::-1]
                    evals = evals[order]
                    evecs = evecs[:, order]
                    w = np.real_if_close(evecs[:, 0])

                    nlags = (self.maxlag - self.minlag) // self.lagstep + 1.0
                    if not np.isclose(evals[0], nlags):
                        warnings.warn(
                            "dominant eigenvalue ({}) is not {}".format(
                                evals[0], nlags
                            )
                        )
                    if not np.all(np.abs(evals[1:]) < nlags):
                        warnings.warn(
                            "multiple eigenvalues have magnitudes above {}".format(
                                nlags
                            )
                        )
                    if not np.isrealobj(w):
                        warnings.warn("estimated weights are complex")

                ic = utils.corr(f0, f1, trajs, self.truncate, w)
                if self.adjust:
                    c0 = utils.corr(
                        f0,
                        f0,
                        trajs,
                        weights=utils.adjust(
                            self.minlag, self.maxlag, self.lagstep, w
                        ),
                    )
                else:
                    c0 = utils.corr(f0, f0, trajs, self.truncate, w)

        else:
            raise ValueError(
                "method must be 'direct', 'fft', 'fftconv', or 'conv'"
            )

        evals, evecs = linalg.eigh(_sym(ic), c0)
        self.cov = c0
        self.evals = evals[::-1]
        self.evecs = evecs[:, ::-1]
        self.its = _ivac_its(
            self.evals, self.minlag, self.maxlag, self.lagstep
        )

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
        if self.addones:
            trajs = _addones(trajs)
        result = []
        for traj in trajs:
            traj = np.asarray(traj, dtype=np.float64)
            result.append(traj @ self.evecs[:, : self.nevecs])
        return result


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
        and 'fft',
        which uses a FFT to compute the time lagged correlations,
        between each pair of features.

    Attributes
    ----------
    lags : 1d array-like of int
        VAC lag time, in units of frames.
    cov : (n_features, n_features) ndarray
        Covariance matrix of the fitted data.

    """

    def __init__(self, lags, nevecs=None, addones=False, method="direct"):
        self.lags = lags
        self.nevecs = nevecs
        self.addones = addones
        self.method = method

    def fit(self, trajs):
        """Compute VAC results from input trajectories.

        Calculate and store VAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.

        """
        if self.addones:
            trajs = _addones(trajs)
        self.cov = covmat(trajs)
        nlags = len(self.lags)
        nfeatures = len(self.cov)
        nevecs = self.nevecs
        if nevecs is None:
            nevecs = nfeatures

        self.evals = np.empty((nlags, nevecs))
        self.evecs = np.empty((nlags, nfeatures, nevecs))
        self.its = np.empty((nlags, nevecs))

        if self.method == "direct":
            for n, lag in enumerate(self.lags):
                ct = covmat(trajs, lag=lag)
                evals, evecs = linalg.eigh(_sym(ct), self.cov)
                self.evals[n] = evals[::-1][:nevecs]
                self.evecs[n] = evecs[:, ::-1][:, :nevecs]
                self.its[n] = _vac_its(self.evals[n], lag)
        elif self.method == "fft":
            cts = np.zeros((nlags, nfeatures, nfeatures))
            for i in range(nfeatures):
                for j in range(i, nfeatures):
                    corr1, corr2 = _fftcorr(trajs, self.lags, i, j)
                    cts[:, i, j] += corr1
                    cts[:, j, i] += corr2
                    if i == j:
                        cts[:, i, j] *= 0.5
            for n, (ct, lag) in enumerate(zip(cts, self.lags)):
                evals, evecs = linalg.eigh(_sym(ct), self.cov)
                self.evals[n] = evals[::-1][:nevecs]
                self.evecs[n] = evecs[:, ::-1][:, :nevecs]
                self.its[n] = _vac_its(self.evals[n], lag)
        else:
            raise ValueError("method must be 'direct' or 'fft'")

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
        vac.cov = self.cov
        vac.evals = self.evals[i]
        vac.evecs = self.evecs[i]
        vac.its = self.its[i]
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
        Method used to compute the integrated covariance matrices.
        Currently supported methods are 'direct',
        which sums together time lagged covariance matrices,
        and 'fft',
        which uses a FFT to compute the time lagged correlations
        between each pair of features.

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
        method="direct",
    ):
        if np.any(lags[1:] < lags[:-1]):
            raise ValueError("lags must be nondecreasing")
        if np.any((lags[1:] - lags[:-1]) % lagstep != 0):
            raise ValueError(
                "lags time intervals must be multiples of lagstep"
            )
        self.lags = lags
        self.lagstep = lagstep
        self.nevecs = nevecs
        self.addones = addones
        self.method = method

    def fit(self, trajs):
        """Compute IVAC results from input trajectories.

        Calculate and store IVAC eigenvalues, eigenvector coefficients,
        and implied timescales from the input trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) ndarray
            List of featurized trajectories.

        """
        if self.addones:
            trajs = _addones(trajs)
        self.cov = covmat(trajs)
        nlags = len(self.lags)
        nfeatures = len(self.cov)
        nevecs = self.nevecs
        if nevecs is None:
            nevecs = nfeatures

        self.evals = np.full((nlags, nlags, nevecs), np.nan)
        self.evecs = np.full((nlags, nlags, nfeatures, nevecs), np.nan)
        self.its = np.full((nlags, nlags, nevecs), np.nan)

        ics = np.zeros((nlags - 1, nfeatures, nfeatures))
        if self.method == "direct":
            for n in range(nlags - 1):
                lags = np.arange(
                    self.lags[n] + self.lagstep,
                    self.lags[n + 1] + 1,
                    self.lagstep,
                )
                ics[n] = _icov(trajs, lags=lags)
        elif self.method == "fft":
            lags = np.arange(
                self.lags[0] + self.lagstep,
                self.lags[-1] + 1,
                self.lagstep,
            )
            for i in range(nfeatures):
                for j in range(i, nfeatures):
                    corr1, corr2 = _fftcorr(trajs, lags, i, j)
                    for n in range(nlags - 1):
                        start = (self.lags[n] - self.lags[0]) // self.lagstep
                        end = (self.lags[n + 1] - self.lags[0]) // self.lagstep
                        ics[n, i, j] += np.sum(corr1[start:end])
                        ics[n, j, i] += np.sum(corr2[start:end])
                        if i == j:
                            ics[n, i, j] *= 0.5
        else:
            raise ValueError("method must be 'direct' or 'fft'")

        for i in range(nlags):
            ic = covmat(trajs, lag=self.lags[i])
            evals, evecs = linalg.eigh(_sym(ic), self.cov)
            self.evals[i, i] = evals[::-1][:nevecs]
            self.evecs[i, i] = evecs[:, ::-1][:, :nevecs]
            self.its[i, i] = _ivac_its(
                self.evals[i, i], self.lags[i], self.lags[i], self.lagstep
            )
            for j in range(i + 1, nlags):
                ic += ics[j - 1]
                evals, evecs = linalg.eigh(_sym(ic), self.cov)
                self.evals[i, j] = evals[::-1][:nevecs]
                self.evecs[i, j] = evecs[:, ::-1][:, :nevecs]
                self.its[i, j] = _ivac_its(
                    self.evals[i, j], self.lags[i], self.lags[j], self.lagstep
                )

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
            self.lags[i],
            self.lags[j],
            lagstep=self.lagstep,
            nevecs=self.nevecs,
            addones=self.addones,
            method=self.method,
        )
        ivac.cov = self.cov
        ivac.evals = self.evals[i, j]
        ivac.evecs = self.evecs[i, j]
        ivac.its = self.its[i, j]
        return ivac


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
    cov = covmat(features, weights=weights)
    if np.allclose(cov, np.identity(len(cov))):
        return features
    u = linalg.cholesky(cov)
    uinv = linalg.inv(u)
    result = []
    for x in features:
        x = np.asarray(x, dtype=np.float64)
        result.append(x @ uinv)
    return result


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


def covmat(u, v=None, weights=None, lag=0):
    r"""Compute the (time lagged) covariance matrix between features.

    For a single trajectory with features
    :math:`\vec{u}_t`, :math:`\vec{v}_t` and weights :math:`w_t`
    at each frame :math:`n = 1, 2, \ldots, T`,
    this function calculates

    .. math::

        \mathbf{C}(t) =
            \frac{\sum_{t=1}^{T-\tau} w_t \vec{u}_t \vec{v}_{t+\tau}^T}
            {\sum_{t=1}^{T-\tau} w_t}

    where :math:`\tau` is the lag time.

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
    lag : int, optional
        Lag time for the second set of feature in units of frames.

    Returns
    -------
    (n_features1, n_features2) ndarray
        Covariance matrix or time lagged covariance matrix.

    """
    if v is None:
        v = u
    if len(u) != len(v):
        raise ValueError("mismatch in the number of trajectories")
    cov = np.zeros((np.shape(u[0])[-1], np.shape(v[0])[-1]))
    count = 0.0
    if weights is None:
        for x, y in zip(u, v):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            # apply time lag
            x = x[: len(x) - lag]
            y = y[lag:]

            cov += x.T @ y
            count += len(x)
    else:
        if len(u) != len(weights):
            raise ValueError("mismatch in the number of trajectories")
        for x, y, w in zip(u, v, weights):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            w = np.asarray(w, dtype=np.float64)

            # apply time lag
            x = x[: len(x) - lag]
            y = y[lag:]
            w = w[: len(w) - lag]

            cov += np.einsum("n,ni,nj->ij", w, x, y)
            count += np.sum(w)
    return cov / count


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


def _addones(trajs):
    """Add a feature of all ones to featurized trajectories.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) array-like
        List of featurized trajectories.

    Returns
    -------
    list of (n_frames[i], n_features + 1) ndarray
        Trajectories with an additional feature of all ones.

    """
    result = []
    for traj in trajs:
        ones = np.ones((len(traj), 1))
        result.append(np.concatenate([ones, traj], axis=-1))
    return result


def _icov(trajs, lags, weights=None, method="direct"):
    r"""Compute the integrated covariance matrix over given lag times.

    For a single trajectory
    :math:`\vec{x}_0, \vec{x}_1, \ldots, \vec{x}_{T-1}`,
    this function calculates
    :math:`\mathbf{I} = \sum_t \mathbf{C}(t)`
    where the time lagged covariance matrix :math:`\mathbf{C}(t)`
    is given by

    .. math::

        \mathbf{C}(t) =
            \frac{1}{T-t} \sum_{n=0}^{T-t-1} \vec{x}_n \vec{x}_{n+t}^T

    The method 'direct' performs this naive sum.
    When there is a large number of lag times,
    method 'fft' might be faster.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) array-like
        List of featurized trajectories.
    lags : (n_lags,) array-like of int
        Lag times to sum over.
    method : str, optional
        Method to compute the integrated covariance matrix.
        Currently, 'direct' and 'fft' are supported.

    Returns
    -------
    (n_features, n_features) ndarray
        Integrated covariance matrix.

    """
    nfeatures = np.shape(trajs[0])[-1]
    ic = np.zeros((nfeatures, nfeatures))
    if method == "direct":
        for lag in lags:
            ic += covmat(trajs, weights=weights, lag=lag)
    elif method == "fft":
        for i in range(nfeatures):
            for j in range(i, nfeatures):
                corr1, corr2 = _fftcorr(trajs, lags, i, j)
                ic[i, j] += np.sum(corr1)
                ic[j, i] += np.sum(corr2)
                if i == j:
                    ic[i, j] *= 0.5
    else:
        raise ValueError("method must be 'direct' or 'fft'")
    return ic


def _fftcorr(trajs, lags, i, j):
    r"""Compute the correlation between time lagged features using FFT.

    For features :math:`\phi_i` and :math:`\phi_j`, this function
    computes two time lagged correlations for each lag time :math:`t`:

    .. math::

        \mathrm{corr}_1 = \mathbf{E}[\phi_i(x_0) \phi_j(x_t)]

        \mathrm{corr}_2 = \mathbf{E}[\phi_i(x_t) \phi_j(x_0)]

    Parameters
    ----------
    trajs : list of (n_frames[k], n_features) ndarray
        List of featurized trajectories.
    lags : (n_lags,) ndarray of int
        Lag times for which to compute the time lagged correlations.
    i, j : int
        Indices of the features.

    Returns
    -------
    corr1, corr2 : (n_lags,) ndarray
        Time lagged correlations between specified features.

    """
    corr1 = np.zeros(len(lags))
    corr2 = np.zeros(len(lags))
    count = np.zeros(len(lags))
    for traj in trajs:
        corr = signal.correlate(traj[:, i], traj[:, j], method="fft")
        corr1 += corr[len(traj) - 1 - lags]
        corr2 += corr[len(traj) - 1 + lags]
        count += len(traj) - lags
    return corr1 / count, corr2 / count


def _sym(mat):
    """Symmetrize matrix.

    Parameters
    ----------
    mat : (N, N) ndarray
        Matrix.

    Returns
    -------
    (N, N) ndarray
        Symmetrized matrix.
    """
    return 0.5 * (mat + mat.T)
