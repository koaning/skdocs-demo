import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Demo of HDBSCAN clustering algorithm
    .. currentmodule:: sklearn

    In this demo we will take a look at :class:`cluster.HDBSCAN` from the
    perspective of generalizing the :class:`cluster.DBSCAN` algorithm.
    We'll compare both algorithms on specific datasets. Finally we'll evaluate
    HDBSCAN's sensitivity to certain hyperparameters.

    We first define a couple utility functions for convenience.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import DBSCAN, HDBSCAN
    from sklearn.datasets import make_blobs

    def plot(X, labels, probabilities=None, parameters=None, ground_truth=False, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        _labels = _labels if _labels is not None else np.ones(X.shape[0])
        probabilities = probabilities if probabilities is not None else np.ones(X.shape[0])
        unique_labels = set(_labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        proba_map = {_idx: probabilities[_idx] for _idx in range(len(_labels))}
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_index = (_labels == k).nonzero()[0]
            for ci in class_index:
                ax.plot(X[ci, 0], X[ci, 1], 'x' if k == -1 else 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=4 if k == -1 else 1 + 5 * proba_map[ci])
        n_clusters_ = len(set(_labels)) - (1 if -1 in _labels else 0)
        preamble = 'True' if ground_truth else 'Estimated'
        title = f'{preamble} number of clusters: {n_clusters_}'
        if parameters is not None:
            parameters_str = ', '.join((f'{k}={v}' for k, v in parameters.items()))
            title = title + f' | {parameters_str}'
        ax.set_title(title)
        plt.tight_layout()
    return DBSCAN, HDBSCAN, make_blobs, plot, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Generate sample data
    One of the greatest advantages of HDBSCAN over DBSCAN is its out-of-the-box
    robustness. It's especially remarkable on heterogeneous mixtures of data.
    Like DBSCAN, it can model arbitrary shapes and distributions, however unlike
    DBSCAN it does not require specification of an arbitrary and sensitive
    `eps` hyperparameter.

    For example, below we generate a dataset from a mixture of three bi-dimensional
    and isotropic Gaussian distributions.


    """
    )
    return


@app.cell
def _(make_blobs, plot):
    _centers = [[1, 1], [-1, -1], [1.5, -1.5]]
    X, _labels_true = make_blobs(n_samples=750, centers=_centers, cluster_std=[0.4, 0.1, 0.75], random_state=0)
    plot(X, labels=_labels_true, ground_truth=True)
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Scale Invariance
    It's worth remembering that, while DBSCAN provides a default value for `eps`
    parameter, it hardly has a proper default value and must be tuned for the
    specific dataset at use.

    As a simple demonstration, consider the clustering for a `eps` value tuned
    for one dataset, and clustering obtained with the same value but applied to
    rescaled versions of the dataset.


    """
    )
    return


@app.cell
def _(DBSCAN, X, plot, plt):
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 12))
    _dbs = DBSCAN(eps=0.3)
    for _idx, _scale in enumerate([1, 0.5, 3]):
        _dbs.fit(X * _scale)
        plot(X * _scale, _dbs.labels_, parameters={'scale': _scale, 'eps': 0.3}, ax=_axes[_idx])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Indeed, in order to maintain the same results we would have to scale `eps` by
    the same factor.


    """
    )
    return


@app.cell
def _(DBSCAN, X, plot, plt):
    _fig, axis = plt.subplots(1, 1, figsize=(12, 5))
    _dbs = DBSCAN(eps=0.9).fit(3 * X)
    plot(3 * X, _dbs.labels_, parameters={'scale': 3, 'eps': 0.9}, ax=axis)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    While standardizing data (e.g. using
    :class:`sklearn.preprocessing.StandardScaler`) helps mitigate this problem,
    great care must be taken to select the appropriate value for `eps`.

    HDBSCAN is much more robust in this sense: HDBSCAN can be seen as
    clustering over all possible values of `eps` and extracting the best
    clusters from all possible clusters (see `User Guide <HDBSCAN>`).
    One immediate advantage is that HDBSCAN is scale-invariant.


    """
    )
    return


@app.cell
def _(HDBSCAN, X, plot, plt):
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 12))
    _hdb = HDBSCAN(copy=True)
    for _idx, _scale in enumerate([1, 0.5, 3]):
        _hdb.fit(X * _scale)
        plot(X * _scale, _hdb.labels_, _hdb.probabilities_, ax=_axes[_idx], parameters={'scale': _scale})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Multi-Scale Clustering
    HDBSCAN is much more than scale invariant though -- it is capable of
    multi-scale clustering, which accounts for clusters with varying density.
    Traditional DBSCAN assumes that any potential clusters are homogeneous in
    density. HDBSCAN is free from such constraints. To demonstrate this we
    consider the following dataset


    """
    )
    return


@app.cell
def _(make_blobs, plot):
    _centers = [[-0.85, -0.85], [-0.85, 0.85], [3, 3], [3, -3]]
    X_1, _labels_true = make_blobs(n_samples=750, centers=_centers, cluster_std=[0.2, 0.35, 1.35, 1.35], random_state=0)
    plot(X_1, labels=_labels_true, ground_truth=True)
    return (X_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This dataset is more difficult for DBSCAN due to the varying densities and
    spatial separation:

    - If `eps` is too large then we risk falsely clustering the two dense
      clusters as one since their mutual reachability will extend
      clusters.
    - If `eps` is too small, then we risk fragmenting the sparser clusters
      into many false clusters.

    Not to mention this requires manually tuning choices of `eps` until we
    find a tradeoff that we are comfortable with.


    """
    )
    return


@app.cell
def _(DBSCAN, X_1, plot, plt):
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 8))
    params = {'eps': 0.7}
    _dbs = DBSCAN(**params).fit(X_1)
    plot(X_1, _dbs.labels_, parameters=params, ax=_axes[0])
    params = {'eps': 0.3}
    _dbs = DBSCAN(**params).fit(X_1)
    plot(X_1, _dbs.labels_, parameters=params, ax=_axes[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To properly cluster the two dense clusters, we would need a smaller value of
    epsilon, however at `eps=0.3` we are already fragmenting the sparse clusters,
    which would only become more severe as we decrease epsilon. Indeed it seems
    that DBSCAN is incapable of simultaneously separating the two dense clusters
    while preventing the sparse clusters from fragmenting. Let's compare with
    HDBSCAN.


    """
    )
    return


@app.cell
def _(HDBSCAN, X_1, plot):
    _hdb = HDBSCAN(copy=True).fit(X_1)
    plot(X_1, _hdb.labels_, _hdb.probabilities_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    HDBSCAN is able to adapt to the multi-scale structure of the dataset without
    requiring parameter tuning. While any sufficiently interesting dataset will
    require tuning, this case demonstrates that HDBSCAN can yield qualitatively
    better classes of clusterings without users' intervention which are
    inaccessible via DBSCAN.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Hyperparameter Robustness
    Ultimately tuning will be an important step in any real world application, so
    let's take a look at some of the most important hyperparameters for HDBSCAN.
    While HDBSCAN is free from the `eps` parameter of DBSCAN, it does still have
    some hyperparameters like `min_cluster_size` and `min_samples` which tune its
    results regarding density. We will however see that HDBSCAN is relatively robust
    to various real world examples thanks to those parameters whose clear meaning
    helps tuning them.

    ### `min_cluster_size`
    `min_cluster_size` is the minimum number of samples in a group for that
    group to be considered a cluster.

    Clusters smaller than the ones of this size will be left as noise.
    The default value is 5. This parameter is generally tuned to
    larger values as needed. Smaller values will likely to lead to results with
    fewer points labeled as noise. However values which too small will lead to
    false sub-clusters being picked up and preferred. Larger values tend to be
    more robust with respect to noisy datasets, e.g. high-variance clusters with
    significant overlap.


    """
    )
    return


@app.cell
def _(HDBSCAN, X_1, plot, plt):
    _PARAM = ({'min_cluster_size': 5}, {'min_cluster_size': 3}, {'min_cluster_size': 25})
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 12))
    for _i, _param in enumerate(_PARAM):
        _hdb = HDBSCAN(copy=True, **_param).fit(X_1)
        _labels = _hdb.labels_
        plot(X_1, _labels, _hdb.probabilities_, _param, ax=_axes[_i])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### `min_samples`
    `min_samples` is the number of samples in a neighborhood for a point to
    be considered as a core point, including the point itself.
    `min_samples` defaults to `min_cluster_size`.
    Similarly to `min_cluster_size`, larger values for `min_samples` increase
    the model's robustness to noise, but risks ignoring or discarding
    potentially valid but small clusters.
    `min_samples` better be tuned after finding a good value for `min_cluster_size`.


    """
    )
    return


@app.cell
def _(HDBSCAN, X_1, plot, plt):
    _PARAM = ({'min_cluster_size': 20, 'min_samples': 5}, {'min_cluster_size': 20, 'min_samples': 3}, {'min_cluster_size': 20, 'min_samples': 25})
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 12))
    for _i, _param in enumerate(_PARAM):
        _hdb = HDBSCAN(copy=True, **_param).fit(X_1)
        _labels = _hdb.labels_
        plot(X_1, _labels, _hdb.probabilities_, _param, ax=_axes[_i])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### `dbscan_clustering`
    During `fit`, `HDBSCAN` builds a single-linkage tree which encodes the
    clustering of all points across all values of :class:`~cluster.DBSCAN`'s
    `eps` parameter.
    We can thus plot and evaluate these clusterings efficiently without fully
    recomputing intermediate values such as core-distances, mutual-reachability,
    and the minimum spanning tree. All we need to do is specify the `cut_distance`
    (equivalent to `eps`) we want to cluster with.


    """
    )
    return


@app.cell
def _(HDBSCAN, X_1, plot, plt):
    _PARAM = ({'cut_distance': 0.1}, {'cut_distance': 0.5}, {'cut_distance': 1.0})
    _hdb = HDBSCAN(copy=True)
    _hdb.fit(X_1)
    _fig, _axes = plt.subplots(len(_PARAM), 1, figsize=(10, 12))
    for _i, _param in enumerate(_PARAM):
        _labels = _hdb.dbscan_clustering(**_param)
        plot(X_1, _labels, _hdb.probabilities_, _param, ax=_axes[_i])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
