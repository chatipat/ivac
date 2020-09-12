import numpy as np
import numba as nb
import warnings
from scipy import linalg, optimize, signal


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
    def __init__(
        self,
        minlag,
        maxlag,
        lagstep=1,
        nevecs=None,
        addones=False,
        method="direct",
    ):
        if minlag > maxlag:
            raise ValueError("minlag must be less than or equal to maxlag")
        if (maxlag - minlag) % lagstep != 0:
            raise ValueError("lag time interval must be a multiple of lagstep")
        self.minlag = minlag
        self.maxlag = maxlag
        self.lagstep = lagstep
        self.nevecs = nevecs
        self.addones = addones
        self.method = method

    def fit(self, trajs):
        if self.addones:
            trajs = _addones(trajs)
        c0 = _cov(trajs)
        lags = np.arange(self.minlag, self.maxlag + 1, self.lagstep)
        ic = _icov(trajs, lags, method=self.method)
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


class LinearVACScan:
    def __init__(self, lags, nevecs=None, addones=False, method="direct"):
        self.lags = lags
        self.nevecs = nevecs
        self.addones = addones
        self.method = method

    def fit(self, trajs):
        if self.addones:
            trajs = _addones(trajs)
        c0 = _cov(trajs)
        nlags = len(self.lags)
        nfeatures = len(c0)
        nevecs = self.nevecs
        if nevecs is None:
            nevecs = nfeatures

        self.evals = np.empty((nlags, nevecs))
        self.evecs = np.empty((nlags, nfeatures, nevecs))
        self.its = np.empty((nlags, nevecs))

        if self.method == "direct":
            for n, lag in enumerate(self.lags):
                ct = _cov(trajs, lag)
                evals, evecs = linalg.eigh(_sym(ct), c0)
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
                evals, evecs = linalg.eigh(_sym(ct), c0)
                self.evals[n] = evals[::-1][:nevecs]
                self.evecs[n] = evecs[:, ::-1][:, :nevecs]
                self.its[n] = _vac_its(self.evals[n], lag)
        else:
            raise ValueError("method must be 'direct' or 'fft'")

    def __getitem__(self, lag):
        i = np.argwhere(self.lags == lag)[0, 0]
        vac = LinearVAC(lag, nevecs=self.nevecs, addones=self.addones)
        vac.evals = self.evals[i]
        vac.evecs = self.evecs[i]
        vac.its = self.its[i]
        return vac


class LinearIVACScan:
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
        if self.addones:
            trajs = _addones(trajs)
        c0 = _cov(trajs)
        nlags = len(self.lags)
        nfeatures = len(c0)
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
                ics[n] = _icov(trajs, lags)
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
            ic = _cov(trajs, self.lags[i])
            evals, evecs = linalg.eigh(_sym(ic), c0)
            self.evals[i, i] = evals[::-1][:nevecs]
            self.evecs[i, i] = evecs[:, ::-1][:, :nevecs]
            self.its[i, i] = _ivac_its(
                self.evals[i, i], self.lags[i], self.lags[i], self.lagstep
            )
            for j in range(i + 1, nlags):
                ic += ics[j - 1]
                evals, evecs = linalg.eigh(_sym(ic), c0)
                self.evals[i, j] = evals[::-1][:nevecs]
                self.evecs[i, j] = evecs[:, ::-1][:, :nevecs]
                self.its[i, j] = _ivac_its(
                    self.evals[i, j], self.lags[i], self.lags[j], self.lagstep
                )

    def __getitem__(self, lags):
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
        ivac.evals = self.evals[i, j]
        ivac.evecs = self.evecs[i, j]
        ivac.its = self.its[i, j]
        return ivac


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


def _icov(trajs, lags, method="direct"):
    nfeatures = np.shape(trajs[0])[-1]
    ic = np.zeros((nfeatures, nfeatures))
    if method == "direct":
        for lag in lags:
            ic += _cov(trajs, lag)
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
