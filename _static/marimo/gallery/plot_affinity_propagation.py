import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        
⚠️ **Note**: This notebook was automatically converted from Jupyter.
Some features may behave differently in marimo.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Demo of affinity propagation clustering algorithm

    Reference:
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import numpy as np

    from sklearn import metrics
    from sklearn.cluster import AffinityPropagation
    from sklearn.datasets import make_blobs
    return AffinityPropagation, make_blobs, metrics, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Generate sample data


    """
    )
    return


@app.cell
def _(make_blobs):
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=0
    )
    return X, labels_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Compute Affinity Propagation


    """
    )
    return


@app.cell
def _(AffinityPropagation, X, labels_true, metrics):
    af = AffinityPropagation(preference=-50, random_state=0).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels)
    )
    print(
        "Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, labels, metric="sqeuclidean")
    )
    return cluster_centers_indices, labels, n_clusters_


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot result


    """
    )
    return


@app.cell
def _(X, cluster_centers_indices, labels, n_clusters_, np):
    import matplotlib.pyplot as plt

    plt.close("all")
    plt.figure(1)
    plt.clf()

    colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 4)))

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.scatter(
            X[class_members, 0], X[class_members, 1], color=col["color"], marker="."
        )
        plt.scatter(
            cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
        )
        for x in X[class_members]:
            plt.plot(
                [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"]
            )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
    return

if __name__ == "__main__":
    app.run()
