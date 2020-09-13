import torch


class TimeLaggedPairSampler:
    def __init__(
        self,
        trajs,
        dtype=torch.float32,
        device="cpu",
    ):
        self.dtype = dtype
        self.device = device

        features = []
        maxlags = []
        for traj in trajs:
            traj = torch.as_tensor(traj, dtype=dtype, device=device)
            features.append(traj)
            maxlags.append(torch.arange(len(traj) - 1, -1, -1, device=device))
        features = torch.cat(features, dim=0)
        maxlags = torch.cat(maxlags)
        lags = torch.arange(torch.max(maxlags) + 1, device=device)
        self.features = features
        self.indices = torch.argsort(maxlags)
        self.offsets = torch.bucketize(lags, maxlags[self.indices], right=True)
        self.lengths = len(features) - self.offsets

    def __call__(self, num, minlag, maxlag=None):
        if maxlag is None:
            maxlag = minlag
        lags = torch.randint(minlag, maxlag + 1, (num,), device=self.device)
        i = self.offsets[lags] + torch.floor(
            self.lengths[lags] * torch.rand(num, device=self.device)
        ).to(dtype=lags.dtype)
        ix = self.indices[i]
        iy = ix + lags
        return self.features[ix], self.features[iy]


class VAMPScore:
    def __init__(
        self,
        score=1,
        center=False,
        addones=False,
        minlag=None,
        maxlag=None,
        lagstep=1,
    ):
        self.score = score
        self.center = center
        self.addones = addones
        self.minlag = minlag
        self.maxlag = maxlag
        self.lagstep = lagstep

        self._factor = 1.0
        if minlag is not None or maxlag is not None:
            if minlag is None or maxlag is None:
                raise ValueError("both minlag and maxlag must be specified")
            if (maxlag - minlag) % lagstep != 0:
                raise ValueError(
                    "lag time interval must be a multiple of lagstep"
                )
            self._factor = ((maxlag - minlag) // lagstep + 1.0) ** score

    def __call__(self, x, y):
        if self.center:
            mean = 0.5 * (torch.mean(x, dim=0) + torch.mean(y, dim=0))
            x = x - mean
            y = y - mean
        if self.addones:
            x = _addones(x)
            y = _addones(y)

        c0 = x.t() @ x + y.t() @ y
        ct = x.t() @ y + y.t() @ x
        op = torch.inverse(c0) @ ct
        if self.score == 1:
            score = torch.trace(op)
        elif self.score == 2:
            score = torch.trace(op @ op)
        else:
            raise ValueError("score must be 1 or 2")

        if self.center and not self.addones:
            score += 1.0
        return self._factor * score


def _addones(x):
    ones = torch.ones(len(x), 1, dtype=x.dtype, device=x.device)
    return torch.cat([ones, x], dim=-1)
