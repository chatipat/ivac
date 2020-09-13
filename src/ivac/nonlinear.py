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
