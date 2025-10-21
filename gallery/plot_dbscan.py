import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Demo of DBSCAN clustering algorithm

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds core
    samples in regions of high density and expands clusters from them. This
    algorithm is good for data which contains clusters of similar density.

    See the `sphx_glr_auto_examples_cluster_plot_cluster_comparison.py` example
    for a demo of different clustering algorithms on 2D datasets.

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

    We use :class:`~sklearn.datasets.make_blobs` to create 3 synthetic clusters.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )

    X = StandardScaler().fit_transform(X)
    return X, labels_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can visualize the resulting data:


    """
    )
    return


@app.cell
def _(X):
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Compute DBSCAN

    One can access the labels assigned by :class:`~sklearn.cluster.DBSCAN` using
    the `labels_` attribute. Noisy samples are given the label $-1$.


    """
    )
    return


@app.cell
def _(X):
    import numpy as np

    from sklearn import metrics
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    return db, labels, metrics, n_clusters_, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Clustering algorithms are fundamentally unsupervised learning methods.
    However, since :class:`~sklearn.datasets.make_blobs` gives access to the true
    labels of the synthetic clusters, it is possible to use evaluation metrics
    that leverage this "supervised" ground truth information to quantify the
    quality of the resulting clusters. Examples of such metrics are the
    homogeneity, completeness, V-measure, Rand-Index, Adjusted Rand-Index and
    Adjusted Mutual Information (AMI).

    If the ground truth labels are not known, evaluation can only be performed
    using the model results itself. In that case, the Silhouette Coefficient comes
    in handy.

    For more information, see the
    `sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`
    example or the `clustering_evaluation` module.


    """
    )
    return


@app.cell
def _(X, labels, labels_true, metrics):
    print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
    print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
    print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
    print(
        "Adjusted Mutual Information:"
        f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
    )
    print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot results

    Core samples (large dots) and non-core samples (small dots) are color-coded
    according to the assigned cluster. Samples tagged as noise are represented in
    black.


    """
    )
    return


@app.cell
def _(X, db, labels, n_clusters_, np, plt):
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
