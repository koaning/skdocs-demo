import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Demonstration of k-means assumptions

    This example is meant to illustrate situations where k-means produces
    unintuitive and possibly undesirable clusters.

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
    ## Data generation

    The function :func:`~sklearn.datasets.make_blobs` generates isotropic
    (spherical) gaussian blobs. To obtain anisotropic (elliptical) gaussian blobs
    one has to define a linear `transformation`.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.datasets import make_blobs

    n_samples = 1500
    random_state = 170
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    X_aniso = np.dot(X, transformation)  # Anisotropic blobs
    X_varied, y_varied = make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )  # Unequal variance
    X_filtered = np.vstack(
        (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10])
    )  # Unevenly sized blobs
    y_filtered = [0] * 500 + [1] * 100 + [2] * 10
    return (
        X,
        X_aniso,
        X_filtered,
        X_varied,
        random_state,
        y,
        y_filtered,
        y_varied,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can visualize the resulting data:


    """
    )
    return


@app.cell
def _(X, X_aniso, X_filtered, X_varied, y, y_filtered, y_varied):
    import matplotlib.pyplot as plt
    _fig, _axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    _axs[0, 0].scatter(X[:, 0], X[:, 1], c=y)
    _axs[0, 0].set_title('Mixture of Gaussian Blobs')
    _axs[0, 1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y)
    _axs[0, 1].set_title('Anisotropically Distributed Blobs')
    _axs[1, 0].scatter(X_varied[:, 0], X_varied[:, 1], c=y_varied)
    _axs[1, 0].set_title('Unequal Variance')
    _axs[1, 1].scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered)
    _axs[1, 1].set_title('Unevenly Sized Blobs')
    plt.suptitle('Ground truth clusters').set_y(0.95)
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fit models and plot results

    The previously generated data is now used to show how
    :class:`~sklearn.cluster.KMeans` behaves in the following scenarios:

    - Non-optimal number of clusters: in a real setting there is no uniquely
      defined **true** number of clusters. An appropriate number of clusters has
      to be decided from data-based criteria and knowledge of the intended goal.
    - Anisotropically distributed blobs: k-means consists of minimizing sample's
      euclidean distances to the centroid of the cluster they are assigned to. As
      a consequence, k-means is more appropriate for clusters that are isotropic
      and normally distributed (i.e. spherical gaussians).
    - Unequal variance: k-means is equivalent to taking the maximum likelihood
      estimator for a "mixture" of k gaussian distributions with the same
      variances but with possibly different means.
    - Unevenly sized blobs: there is no theoretical result about k-means that
      states that it requires similar cluster sizes to perform well, yet
      minimizing euclidean distances does mean that the more sparse and
      high-dimensional the problem is, the higher is the need to run the algorithm
      with different centroid seeds to ensure a global minimal inertia.


    """
    )
    return


@app.cell
def _(X, X_aniso, X_filtered, X_varied, plt, random_state):
    from sklearn.cluster import KMeans
    common_params = {'n_init': 'auto', 'random_state': random_state}
    _fig, _axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    _y_pred = KMeans(n_clusters=2, **common_params).fit_predict(X)
    _axs[0, 0].scatter(X[:, 0], X[:, 1], c=_y_pred)
    _axs[0, 0].set_title('Non-optimal Number of Clusters')
    _y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X_aniso)
    _axs[0, 1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=_y_pred)
    _axs[0, 1].set_title('Anisotropically Distributed Blobs')
    _y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X_varied)
    _axs[1, 0].scatter(X_varied[:, 0], X_varied[:, 1], c=_y_pred)
    _axs[1, 0].set_title('Unequal Variance')
    _y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X_filtered)
    _axs[1, 1].scatter(X_filtered[:, 0], X_filtered[:, 1], c=_y_pred)
    _axs[1, 1].set_title('Unevenly Sized Blobs')
    plt.suptitle('Unexpected KMeans clusters').set_y(0.95)
    plt.show()
    return KMeans, common_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Possible solutions

    For an example on how to find a correct number of blobs, see
    `sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py`.
    In this case it suffices to set `n_clusters=3`.


    """
    )
    return


@app.cell
def _(KMeans, X, common_params, plt):
    _y_pred = KMeans(n_clusters=3, **common_params).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=_y_pred)
    plt.title('Optimal Number of Clusters')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To deal with unevenly sized blobs one can increase the number of random
    initializations. In this case we set `n_init=10` to avoid finding a
    sub-optimal local minimum. For more details see `kmeans_sparse_high_dim`.


    """
    )
    return


@app.cell
def _(KMeans, X_filtered, plt, random_state):
    _y_pred = KMeans(n_clusters=3, n_init=10, random_state=random_state).fit_predict(X_filtered)
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=_y_pred)
    plt.title('Unevenly Sized Blobs \nwith several initializations')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As anisotropic and unequal variances are real limitations of the k-means
    algorithm, here we propose instead the use of
    :class:`~sklearn.mixture.GaussianMixture`, which also assumes gaussian
    clusters but does not impose any constraints on their variances. Notice that
    one still has to find the correct number of blobs (see
    `sphx_glr_auto_examples_mixture_plot_gmm_selection.py`).

    For an example on how other clustering methods deal with anisotropic or
    unequal variance blobs, see the example
    `sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`.


    """
    )
    return


@app.cell
def _(X_aniso, X_varied, plt):
    from sklearn.mixture import GaussianMixture
    _fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    _y_pred = GaussianMixture(n_components=3).fit_predict(X_aniso)
    ax1.scatter(X_aniso[:, 0], X_aniso[:, 1], c=_y_pred)
    ax1.set_title('Anisotropically Distributed Blobs')
    _y_pred = GaussianMixture(n_components=3).fit_predict(X_varied)
    ax2.scatter(X_varied[:, 0], X_varied[:, 1], c=_y_pred)
    ax2.set_title('Unequal Variance')
    plt.suptitle('Gaussian mixture clusters').set_y(0.95)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Final remarks

    In high-dimensional spaces, Euclidean distances tend to become inflated
    (not shown in this example). Running a dimensionality reduction algorithm
    prior to k-means clustering can alleviate this problem and speed up the
    computations (see the example
    `sphx_glr_auto_examples_text_plot_document_clustering.py`).

    In the case where clusters are known to be isotropic, have similar variance
    and are not too sparse, the k-means algorithm is quite effective and is one of
    the fastest clustering algorithms available. This advantage is lost if one has
    to restart it several times to avoid convergence to a local minimum.


    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
