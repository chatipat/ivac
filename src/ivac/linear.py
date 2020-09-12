import numpy as np
from scipy import linalg


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


def _vac_its(evals, lag):
    its = np.full(len(evals), np.nan)
    its[evals >= 1.0] = np.inf
    mask = np.logical_and(0.0 < evals, evals < 1.0)
    its[mask] = -lag / np.log(evals[mask])
    return its


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
