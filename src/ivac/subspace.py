import numpy as np
from scipy import linalg
from .utils import get_nfeatures, preprocess_trajs, trajs_matmul


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
    features = preprocess_trajs(features)
    cov = covmat(features, weights=weights)
    if np.allclose(cov, np.identity(len(cov))):
        return features
    u = linalg.cholesky(cov)
    uinv = linalg.inv(u)
    return trajs_matmul(features, uinv)


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
    nfeatures1 = get_nfeatures(u)
    nfeatures2 = get_nfeatures(v)
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
