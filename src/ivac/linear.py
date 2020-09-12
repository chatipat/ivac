import numpy as np
import numba as nb
import warnings
from scipy import linalg, optimize


class LinearVAC:
    def __init__(self, lag, nevecs=None, addones=False):
        self.lag = lag
        self.nevecs = nevecs
        self.addones = addones

    def fit(self, trajs):
        if self.addones:
            trajs = _addones(trajs)
        c0 = _cov(trajs)
        ct = _cov(trajs, self.lag)
        evals, evecs = linalg.eigh(_sym(ct), c0)
        self.evals = evals[::-1]
        self.evecs = evecs[:, ::-1]
        self.its = _vac_its(self.evals, self.lag)

    def transform(self, trajs):
        if self.addones:
            trajs = _addones(trajs)
        result = []
        for traj in trajs:
            traj = np.asarray(traj, dtype=np.float64)
            result.append(traj @ self.evecs[:, : self.nevecs])
        return result


class LinearIVAC:
    def __init__(self, minlag, maxlag, lagstep=1, nevecs=None, addones=False):
        if minlag > maxlag:
            raise ValueError("minlag must be less than or equal to maxlag")
        if (maxlag - minlag) % lagstep != 0:
            raise ValueError("lag time interval must be a multiple of lagstep")
        self.minlag = minlag
        self.maxlag = maxlag
        self.lagstep = lagstep
        self.nevecs = nevecs
        self.addones = addones

    def fit(self, trajs):
        if self.addones:
            trajs = _addones(trajs)
        c0 = _cov(trajs)
        ic = np.zeros_like(c0)
        for lag in range(self.minlag, self.maxlag + 1, self.lagstep):
            ic += _cov(trajs, lag)
        evals, evecs = linalg.eigh(_sym(ic), c0)
        self.evals = evals[::-1]
        self.evecs = evecs[:, ::-1]
        self.its = _ivac_its(
            self.evals, self.minlag, self.maxlag, self.lagstep
        )

    def transform(self, trajs):
        if self.addones:
            trajs = _addones(trajs)
        result = []
        for traj in trajs:
            traj = np.asarray(traj, dtype=np.float64)
            result.append(traj @ self.evecs[:, : self.nevecs])
        return result


def projection_distance(u, v, weights=None, ortho=False):
    if ortho:
        u = orthonormalize(u)
        v = orthonormalize(v)
    cov = _cov2(u, v, weights=weights)
    s = linalg.svdvals(cov)
    s = np.clip(s, 0.0, 1.0)
    return np.sqrt(len(cov) - np.sum(s ** 2))


def orthonormalize(features, weights=None):
    cov = _cov2(features, weights=weights)
    if np.allclose(cov, np.identity(len(cov))):
        return features
    u = linalg.cholesky(cov)
    uinv = linalg.inv(u)
    result = []
    for x in features:
        x = np.asarray(x, dtype=np.float64)
        result.append(x @ uinv)
    return result


def _vac_its(evals, lag):
    its = np.full(len(evals), np.nan)
    its[evals >= 1.0] = np.inf
    mask = np.logical_and(0.0 < evals, evals < 1.0)
    its[mask] = -lag / np.log(evals[mask])
    return its


def _ivac_its(evals, minlag, maxlag, lagstep=1):
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
    result = []
    for traj in trajs:
        ones = np.ones((len(traj), 1))
        result.append(np.concatenate([ones, traj], axis=-1))
    return result


def _cov(trajs, lag=0):
    nfeatures = np.shape(trajs[0])[-1]
    cov = np.zeros((nfeatures, nfeatures))
    count = 0.0
    for traj in trajs:
        traj = np.asarray(traj, dtype=np.float64)
        x = traj[: len(traj) - lag]
        y = traj[lag:]
        cov += x.T @ y
        count += len(x)
    return cov / count


def _sym(mat):
    return 0.5 * (mat + mat.T)


def _cov2(u, v=None, weights=None):
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
            cov += x.T @ y
            count += len(x)
    else:
        if len(u) != len(weights):
            raise ValueError("mismatch in the number of trajectories")
        for x, y, w in zip(u, v, weights):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            w = np.asarray(w, dtype=np.float64)
            cov += np.einsum("n,ni,nj->ij", w, x, y)
            count += np.sum(w)
    return cov / count
