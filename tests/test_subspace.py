import numpy as np
from data import indicator_basis, ou_process
from utils import allclose_sign, allclose_trajs

import ivac


def make_data(ns, dt, nbins):
    """Generate featurized data for testing."""
    return [indicator_basis(ou_process(n, dt), nbins) for n in ns]


def gram_schmidt(x):
    """Perform the Gram-Schmidt process."""
    x = x.copy()
    nfeatures = x.shape[-1]
    for i in range(nfeatures):
        x[:, i] /= np.sqrt(np.mean(x[:, i] ** 2))
        for j in range(i + 1, nfeatures):
            x[:, j] -= np.mean(x[:, i] * x[:, j])

    mat = (x.T @ x) / len(x)
    assert np.allclose(mat, np.identity(len(mat)))
    return x


def test_orthonormal():
    """Test that orthonormalization works."""

    trajs = make_data([1000, 1500, 2000], 0.1, 10)

    trajs1 = ivac.orthonormalize(trajs)

    # sanity checks
    assert trajs1[0].shape == trajs[0].shape

    # same results if concatenated
    cat1 = np.concatenate(trajs1, axis=0)
    (cat2,) = ivac.orthonormalize([np.concatenate(trajs1, axis=0)])
    assert np.allclose(cat1, cat2)

    # first feature is only rescaled
    for traj, traj1 in zip(trajs, trajs1):
        traj = traj[:, 0]
        traj1 = traj1[:, 0]
        scale = np.sqrt(np.sum(traj ** 2))
        scale1 = np.sqrt(np.sum(traj1 ** 2))
        allclose_sign(scale * traj1, scale1 * traj)
    del scale, scale1, traj, traj1

    # result is actually orthonormal
    mat = ivac.covmat(trajs1)
    assert np.allclose(mat, np.identity(len(mat)))
    del mat

    # orthonormalization is idempotent
    trajs2 = ivac.orthonormalize(trajs1)
    allclose_trajs(trajs1, trajs2)

    # constant weights give the same results
    weights = []
    for traj in trajs:
        weights.append(np.ones(len(traj)))
    trajs2 = ivac.orthonormalize(trajs, weights=weights)
    assert allclose_trajs(trajs1, trajs2)

    # similar results as the Gram-Schmidt procedure
    ref = gram_schmidt(np.concatenate(trajs, axis=0))
    assert np.allclose(cat1, ref)


def test_projection():
    """Test that projection distance calculation works."""

    def isclose(a, b):
        return np.isclose(a, b, rtol=0.0, atol=1e-6)

    trajs = make_data([1000, 1500, 2000], 0.1, 10)

    vac = ivac.LinearVAC(5, adjust=False)
    vac.fit(trajs)
    evecs = vac.transform(trajs)

    # same space has zero projection distance
    assert isclose(ivac.projection_distance(trajs, trajs, ortho=True), 0.0)
    assert isclose(ivac.projection_distance(trajs, evecs, ortho=True), 0.0)

    # orthonormalization doesn't change results if basis is already orthonormal
    assert isclose(ivac.projection_distance(evecs, evecs), 0.0)
    evecs1 = [evec[:, :5] for evec in evecs]
    trajs1 = [traj[:, :5] for traj in trajs]
    normal = ivac.orthonormalize(trajs1)
    assert isclose(
        ivac.projection_distance(trajs1, evecs1, ortho=True),
        ivac.projection_distance(normal, evecs1),
    )

    # rescaling features doesn't change results
    evecs2 = [2.0 * evec for evec in evecs1]
    assert isclose(
        ivac.projection_distance(trajs1, evecs1, ortho=True),
        ivac.projection_distance(trajs1, evecs2, ortho=True),
    )

    # swapping features doesn't change results
    evecs2 = [evec[:, ::-1] for evec in evecs1]
    assert isclose(
        ivac.projection_distance(trajs1, evecs1, ortho=True),
        ivac.projection_distance(trajs1, evecs2, ortho=True),
    )

    # trajectory order doesn't change results
    evecs2 = [evec[::-1] for evec in evecs1]
    trajs2 = [traj[::-1] for traj in trajs1]
    assert isclose(
        ivac.projection_distance(trajs2, evecs2, ortho=True),
        ivac.projection_distance(trajs1, evecs1, ortho=True),
    )
    evecs2 = [np.concatenate(evecs1, axis=0)]
    trajs2 = [np.concatenate(trajs1, axis=0)]
    assert isclose(
        ivac.projection_distance(trajs2, evecs2, ortho=True),
        ivac.projection_distance(trajs1, evecs1, ortho=True),
    )

    left = 0.0
    right = np.sqrt(5.0)
    for i in range(1, 11):
        trajs1 = [traj[:, :i] for traj in trajs]

        # monotonic increase when more dimensions are added to the left
        distl = ivac.projection_distance(trajs1, evecs1, ortho=True)
        assert distl >= left or isclose(distl, left)
        left = distl

        # monotonic decrease when more dimensions are added to the right
        distr = ivac.projection_distance(evecs1, trajs1, ortho=True)
        assert distr <= right or isclose(distr, right)
        right = distr

        # projection distance is bounded by sqrt(dim(left))
        assert 0.0 <= distl <= np.sqrt(i)
        assert 0.0 <= distr <= np.sqrt(5.0)
