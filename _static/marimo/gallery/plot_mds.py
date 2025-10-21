import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Multi-dimensional scaling

    An illustration of the metric and non-metric MDS on generated noisy data.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dataset preparation

    We start by uniformly generating 20 points in a 2D space.


    """
    )
    return


@app.cell
def _():
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from sklearn import manifold
    from sklearn.decomposition import PCA
    from sklearn.metrics import euclidean_distances
    EPSILON = np.finfo(np.float32).eps
    n_samples = 20
    # Generate the data
    rng = np.random.RandomState(seed=3)
    X_true = rng.randint(0, 20, 2 * n_samples).astype(float)
    X_true = X_true.reshape((n_samples, 2))
    # Center the data
    X_true = X_true - X_true.mean()
    return (
        EPSILON,
        LineCollection,
        PCA,
        X_true,
        euclidean_distances,
        manifold,
        n_samples,
        np,
        plt,
        rng,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we compute pairwise distances between all points and add
    a small amount of noise to the distance matrix. We make sure
    to keep the noisy distance matrix symmetric.


    """
    )
    return


@app.cell
def _(X_true, euclidean_distances, n_samples, np, rng):
    # Compute pairwise Euclidean distances
    distances = euclidean_distances(X_true)
    noise = rng.rand(n_samples, n_samples)
    # Add noise to the distances
    noise = noise + noise.T
    np.fill_diagonal(noise, 0)
    distances = distances + noise
    return (distances,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here we compute metric, non-metric, and classical MDS of the noisy distance matrix.


    """
    )
    return


@app.cell
def _(distances, manifold):
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        n_init=1,
        random_state=42,
        dissimilarity="precomputed",
        n_jobs=1,
    )
    X_mds = mds.fit(distances).embedding_

    nmds = manifold.MDS(
        n_components=2,
        metric=False,
        max_iter=3000,
        eps=1e-12,
        dissimilarity="precomputed",
        random_state=42,
        n_jobs=1,
        n_init=1,
    )
    X_nmds = nmds.fit_transform(distances)

    cmds = manifold.ClassicalMDS(
        n_components=2,
        metric="precomputed",
    )
    X_cmds = cmds.fit_transform(distances)
    return X_cmds, X_mds, X_nmds


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Rescaling the non-metric MDS solution to match the spread of the original data.


    """
    )
    return


@app.cell
def _(X_nmds, X_true, np):
    X_nmds_1 = X_nmds * (np.sqrt((X_true ** 2).sum()) / np.sqrt((X_nmds ** 2).sum()))
    return (X_nmds_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To make the visual comparisons easier, we rotate the original data and all MDS
    solutions to their PCA axes. And flip horizontal and vertical MDS axes, if needed,
    to match the original data orientation.


    """
    )
    return


@app.cell
def _(PCA, X_cmds, X_mds, X_nmds_1, X_true, np):
    # Rotate the data (CMDS does not need to be rotated, it is inherently PCA-aligned)
    pca = PCA(n_components=2)
    X_true_1 = pca.fit_transform(X_true)
    X_mds_1 = pca.fit_transform(X_mds)
    X_nmds_2 = pca.fit_transform(X_nmds_1)
    for i in [0, 1]:
    # Align the sign of PCs
        if np.corrcoef(X_mds_1[:, i], X_true_1[:, i])[0, 1] < 0:
            X_mds_1[:, i] = X_mds_1[:, i] * -1
        if np.corrcoef(X_nmds_2[:, i], X_true_1[:, i])[0, 1] < 0:
            X_nmds_2[:, i] = X_nmds_2[:, i] * -1
        if np.corrcoef(X_cmds[:, i], X_true_1[:, i])[0, 1] < 0:
            X_cmds[:, i] = X_cmds[:, i] * -1
    return X_mds_1, X_nmds_2, X_true_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Finally, we plot the original data and all MDS reconstructions.


    """
    )
    return


@app.cell
def _(
    EPSILON,
    LineCollection,
    X_cmds,
    X_mds_1,
    X_nmds_2,
    X_true_1,
    distances,
    np,
    plt,
):
    fig = plt.figure(1)
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])
    s = 100
    plt.scatter(X_true_1[:, 0], X_true_1[:, 1], color='navy', s=s, lw=0, label='True Position')
    plt.scatter(X_mds_1[:, 0], X_mds_1[:, 1], color='turquoise', s=s, lw=0, label='MDS')
    plt.scatter(X_nmds_2[:, 0], X_nmds_2[:, 1], color='darkorange', s=s, lw=0, label='Non-metric MDS')
    plt.scatter(X_cmds[:, 0], X_cmds[:, 1], color='lightcoral', s=s, lw=0, label='Classical MDS')
    plt.legend(scatterpoints=1, loc='best', shadow=False)
    start_idx, end_idx = X_mds_1.nonzero()
    segments = [[X_true_1[i, :], X_true_1[j, :]] for i in range(len(X_true_1)) for j in range(len(X_true_1))]
    edges = distances.max() / (distances + EPSILON) * 100
    np.fill_diagonal(edges, 0)
    edges = np.abs(edges)
    lc = LineCollection(segments, zorder=0, cmap=plt.cm.Blues, norm=plt.Normalize(0, edges.max()))
    # Plot the edges
    lc.set_array(edges.flatten())
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    lc.set_linewidths(np.full(len(segments), 0.5))
    ax.add_collection(lc)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
