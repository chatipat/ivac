import numpy as np
import torch
import pytorch_lightning as pl
from .linear import LinearVAC, LinearIVAC


class NonlinearIVAC:
    def __init__(
        self,
        minlag,
        maxlag=None,
        nevecs=None,
        batch_size=None,
        val_batch_size=None,
        val_every=1,
        hidden_widths=[],
        activation=torch.nn.Tanh,
        batchnorm=False,
        standardize=False,
        score="VAMP1",
        lr=0.001,
        patience=None,
        maxiter=None,
        dtype=torch.float,
        device="cpu",
        linear_method="direct",
    ):
        if nevecs is None:
            raise ValueError("nevecs must be specified")
        if batch_size is None:
            raise ValueError("batch_size must be specified")
        if val_batch_size is None:
            val_batch_size = batch_size
        if maxiter is None:
            raise ValueError("maxiter must be specified")

        self.minlag = minlag
        self.maxlag = maxlag
        self.nevecs = nevecs
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.val_every = val_every
        self.hidden_widths = hidden_widths
        self.activation = activation
        self.batchnorm = batchnorm
        self.standardize = standardize
        self.score = score
        self.lr = lr
        self.patience = patience
        self.maxiter = maxiter
        self.dtype = dtype
        self.device = device

        if maxlag is None:
            self.linear = LinearVAC(minlag, addones=True)
        else:
            self.linear = LinearIVAC(
                minlag, maxlag, addones=True, method=linear_method
            )

    def _make_dataloader(self, trajs, batch_size):
        dataset = TimeLaggedPairDataset(
            trajs,
            batch_size,
            self.minlag,
            self.maxlag,
            dtype=self.dtype,
            device=self.device,
        )
        return torch.utils.data.DataLoader(dataset, batch_size=None)

    def fit(self, train_trajs, val_trajs=None, save_dir=None):
        if val_trajs is None:
            val_trajs = train_trajs
        train_dataloader = self._make_dataloader(train_trajs, self.batch_size)
        val_dataloader = self._make_dataloader(val_trajs, self.val_batch_size)

        self.basis = NonlinearBasis(
            nfeatures=np.shape(train_trajs[0])[-1],
            nbasis=self.nevecs - 1,
            hidden_widths=self.hidden_widths,
            activation=self.activation,
            batchnorm=self.batchnorm,
            standardize=self.standardize,
            score=self.score,
            lr=self.lr,
        )

        if self.patience is None:
            early_stop_callback = False
        else:
            early_stop_callback = pl.callbacks.EarlyStopping(
                patience=self.patience, mode="min"
            )

        if self.device == "cpu":
            gpus = 0
        elif self.device == "cuda":
            gpus = 1
        else:
            _, gpu_id = self.device.split(":")
            gpus = [int(gpu_id)]
        precision = {torch.float16: 16, torch.float32: 32, torch.float64: 64}

        trainer = pl.Trainer(
            val_check_interval=1,
            check_val_every_n_epoch=self.val_every,
            default_root_dir=save_dir,
            early_stop_callback=early_stop_callback,
            gpus=gpus,
            limit_train_batches=1,
            limit_val_batches=1,
            max_epochs=self.maxiter,
            precision=precision[self.dtype],
        )
        trainer.fit(self.basis, train_dataloader, val_dataloader)

        self.linear.fit(self.transform_basis(train_trajs))
        self.evals = self.linear.evals
        self.its = self.linear.its

    def transform(self, trajs):
        return self.linear.transform(self.transform_basis(trajs))

    def transform_basis(self, trajs):
        features = []
        for traj in trajs:
            traj = torch.as_tensor(traj, dtype=self.dtype, device=self.device)
            features.append(self.basis(traj).detach().cpu().numpy())
        return features


class NonlinearBasis(pl.LightningModule):
    def __init__(
        self,
        nfeatures,
        nbasis,
        hidden_widths=[],
        activation=torch.nn.Tanh,
        batchnorm=False,
        standardize=False,
        score="VAMP1",
        lr=0.001,
    ):
        super().__init__()

        layers = []
        last_features = nfeatures
        for hidden_features in hidden_widths:
            layers.append(torch.nn.Linear(last_features, hidden_features))
            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(hidden_features))
            layers.append(activation())
            last_features = hidden_features
        layers.append(torch.nn.Linear(last_features, nbasis))
        if standardize:
            layers.append(torch.nn.BatchNorm1d(nbasis, affine=False))
        self.model = torch.nn.Sequential(*layers)

        if score == "VAMP1":
            self.score = VAMPScore(score=1, addones=True)
        elif score == "VAMP2":
            self.score = VAMPScore(score=2, addones=True)
        else:
            raise ValueError("score must be 'VAMP1' or 'VAMP2'")

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        xy = torch.cat([x, y])
        xy = self(xy)
        x, y = xy[: len(x)], xy[len(x) :]
        loss = -self.score(x, y)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        xy = torch.cat([x, y])
        xy = self(xy)
        x, y = xy[: len(x)], xy[len(x) :]
        loss = -self.score(x, y)
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log("val_loss", loss)
        return result


class TimeLaggedPairDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        trajs,
        num,
        minlag,
        maxlag=None,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()
        self.sampler = TimeLaggedPairSampler(trajs, dtype=dtype, device=device)
        self.num = num
        self.minlag = minlag
        self.maxlag = maxlag

    def __iter__(self):
        while True:
            yield self.sampler(self.num, self.minlag, self.maxlag)


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
