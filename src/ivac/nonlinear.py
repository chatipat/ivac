import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from .linear import LinearIVAC, LinearVAC


class NonlinearIVAC:
    """Solve nonlinear IVAC using a neural network basis.

    Parameters
    ----------
    minlag : int
        Minimum IVAC lag time in units of frames.
    maxlag : int, optional
        Maximum IVAC lag time (inclusive) in units of frames.
        If None, this is set to minlag.
    nevecs : int
        Number of eigenvectors (including the trivial eigenvector)
        to find.
    batch_size : int
        Number of samples to draw at each training iteration.
    val_batch_size : int, optional
        Number of samples to draw at each validation iteration.
    val_every : int, optional
        Number of training iterations between validation iterations.
    hidden_widths : list of int, optional
        Number of hidden features at each hidden layer.
        The length of this list is the number of hidden layers.
    activation : torch.nn.Module, optional
        Activation function to use in the neural network.
    batchnorm : bool, optional
        If True, use batch normalization in the neural network.
    standardize : bool, optional
        If True, remove the mean of the output
        and set its standard deviation to 1.
    score : str, optional
        Score function to maximize.
        This can currently be 'VAMP1' or 'VAMP2'.
    lr : float, optional
        Learning rate for optimization.
    patience : int, optional
        Patience parameter for early stopping.
    maxiter : int, optional
        Maximum number of optimization iterations to perform.
    dtype : torch.dtype, optional
        Data type to use for the neural network.
    device : torch.device, optional
        Device to use to optimize the neural network.
    linear_method : str, optional
        Method to use for solving linear IVAC
        using the optimized neural network basis set.
        Currently, 'direct' and 'fft' are supported.

    """

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
        """Prepare the data for training."""
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
        """Train the neural network using the input trajectories.

        Parameters
        ----------
        train_trajs : list of (n_frames[i], n_features)
            Featurized trajectories for the training dataset.
        val_trajs : list of (n_frames[j], n_features), optional
            Featurized trajectories for the validation dataset.
            These do not need to have the same number of frames
            as the training trajectories.
            If None, use the training data for validation.
        save_dir : str, optional
            Directory for saving training output.
            Uses the current directory by default.

        """
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

        callbacks = [ModelCheckpoint(monitor="val_loss")]
        if self.patience is not None:
            callbacks.append(
                EarlyStopping(monitor="val_loss", patience=self.patience)
            )

        if self.device == "cpu":
            accelerator = "cpu"
            devices = "auto"
        elif self.device == "cuda":
            accelerator = "gpu"
            devices = 1
        else:
            _, gpu_id = self.device.split(":")
            accelerator = "gpu"
            devices = [int(gpu_id)]
        precision = {torch.float16: 16, torch.float32: 32, torch.float64: 64}

        self.trainer = L.Trainer(
            val_check_interval=1,
            check_val_every_n_epoch=self.val_every,
            default_root_dir=save_dir,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            limit_train_batches=1,
            limit_val_batches=1,
            max_epochs=self.maxiter,
            precision=precision[self.dtype],
        )
        self.trainer.fit(self.basis, train_dataloader, val_dataloader)

        self.linear.fit(self.transform_basis(train_trajs))
        self.evals = self.linear.evals
        self.its = self.linear.its

    def transform(self, trajs):
        """Compute IVAC eigenvectors at each frame of the trajectories.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) array-like
            List of featurized trajectories.

        Returns
        -------
        list of (n_frames[i], n_evecs) ndarray
            IVAC eigenvectors at each frame of the trajectories.

        """
        return self.linear.transform(self.transform_basis(trajs))

    def transform_basis(self, trajs):
        """Apply the nonlinear combinations to the input features.

        Parameters
        ----------
        trajs : list of (n_frames[i], n_features) array-like
            List of featurized trajectories.

        Returns
        -------
        list of (n_frames[i], n_evecs - 1) ndarray
            Nonlinear combinations of input features.

        """
        dataset = (
            torch.as_tensor(traj, dtype=self.dtype, device=self.device)
            for traj in trajs
        )
        return [
            traj.numpy() for traj in self.trainer.predict(self.basis, dataset)
        ]


class NonlinearBasis(L.LightningModule):
    """Neural network for taking nonlinear combinations of features.

    This is meant to be used with a PyTorch Lightning Trainer.

    Parameters
    ----------
    nfeatures : int
        Number of input features.
    hidden_widths : list of int, optional
        Number of hidden features at each hidden layer.
        The length of this list is the number of hidden layers.
    activation : torch.nn.Module, optional
        Activation function to use in the neural network.
    batchnorm : bool, optional
        If True, use batch normalization in the neural network.
    standardize : bool, optional
        If True, remove the mean of the output
        and set its standard deviation to 1.
    score : str, optional
        Score function to maximize.
        This can currently be 'VAMP1' or 'VAMP2'.
    lr : float, optional
        Learning rate for optimization.

    """

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
        """Apply the nonlinear combinations to input features."""
        return self.model(x)

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """Compute the training loss."""
        x, y = batch
        xy = torch.cat([x, y])
        xy = self(xy)
        x, y = xy[: len(x)], xy[len(x) :]
        loss = -self.score(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute the validation loss."""
        x, y = batch
        xy = torch.cat([x, y])
        xy = self(xy)
        x, y = xy[: len(x)], xy[len(x) :]
        loss = -self.score(x, y)
        self.log("val_loss", loss)
        return loss


class TimeLaggedPairDataset(torch.utils.data.IterableDataset):
    r"""Dataset yielding time lagged pairs from trajectories.

    For a single trajectory :math:`x_0, x_1, \ldots, x_{T-1}`,
    this class samples pairs :math:`(x_t, x_{t+\tau})`.
    For each pair, :math:`\tau` is uniformly drawn from the set
    {minlag, minlag + 1, ..., maxlag}.

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) array-like
        List of featurized trajectories.
    num : int
        Number of pairs to sample and return at each iteration.
    minlag : int
        Minimum lag time in units of frames.
    maxlag : int, optional
        Maximum lag time in units of frames.
        If None, this is set to the minimum lag time.
    dtype : torch.dtype
        Data type of output tensor.
    device : torch.device
        Device of output tensor.

    """

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
        """Yields batches of sampled time lagged pairs indefinitely.

        Yields
        -------
        x, y : (n_samples, n_features) torch.Tensor
            Time lagged pairs.

        """
        while True:
            yield self.sampler(self.num, self.minlag, self.maxlag)


class TimeLaggedPairSampler:
    r"""Sample time lagged pairs from trajectories.

    For a single trajectory :math:`x_0, x_1, \ldots, x_{T-1}`,
    this class samples pairs :math:`(x_t, x_{t+\tau})`.
    For each pair, :math:`\tau` is uniformly drawn from the set
    {minlag, minlag + 1, ..., maxlag}.

    To draw n_samples pairs from trajectories trajs:

    .. code-block::

        sampler = TimeLaggedPairSampler(trajs)
        x, y = sampler(n_samples, minlag, maxlag)

    Parameters
    ----------
    trajs : list of (n_frames[i], n_features) array-like
        List of featurized trajectories.
    dtype : torch.dtype
        Data type of output tensor.
    device : torch.device
        Device of output tensor.

    """

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
        """Sample time lagged pairs from trajectories.

        Parameters
        ----------
        num : int
            Number of pairs to sample.
        minlag : int
            Minimum lag time in units of frames.
        maxlag : int, optional
            Maximum lag time in units of frames.
            If None, this is set to the minimum lag time.

        Returns
        -------
        x, y : (n_samples, n_features) torch.Tensor
            Time lagged pairs.

        """
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
    r"""Compute reversible VAMP scores.

    The VAMP-:math:`k` score is defined as

    .. math::

        \sum_i \lvert \lambda_i \rvert^k

    where :math:`\lambda_i` are the eigenvalues of the estimated
    (integrated) transition operator.

    This class assumes that dynamics are reversible and that the
    constant feature is contained within the feature space.
    If the constant feature is not within the feature space,
    center=True or addones=True (or both) should be set.
    This class can be used as

    .. code-block::

        score_fn = VAMPScore(score=1, addones=True)
        score = score_fn(x, y)

    Parameters
    ----------
    score : int, optional
        VAMP score to compute. Currently, only 1 and 2 are supported.
    center : bool, optional
        If True, remove the mean from each feature before calculating
        the VAMP score, and adjust the resulting score appropriately.
    addones : bool, optional
        If True, adds a feature of all ones before computing the
        VAMP score.
    minlag, maxlag, lagstep : int, optional
        IVAC parameters. If specified, scales the VAMP score to conform
        with the integrated (rather than averaged) covariance matrix.

    """

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
        """Evaluate VAMP score on input features.

        Parameters
        ----------
        x, y : (n_frames, n_features) torch.Tensor
            Tensor of features.

        Returns
        -------
        () torch.Tensor
            VAMP score.

        """
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
    """Add a feature of all ones.

    Parameters
    ----------
    x : (n_frames, n_features) torch.Tensor
        Tensor of features.

    Returns
    -------
    (n_frames, n_features + 1) torch.Tensor
        Input tensor with an additional feature of all ones.

    """
    ones = torch.ones(len(x), 1, dtype=x.dtype, device=x.device)
    return torch.cat([ones, x], dim=-1)
