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

    # A demo of K-Means clustering on the handwritten digits data

    In this example we compare the various initialization strategies for K-means in
    terms of runtime and quality of the results.

    As the ground truth is known here, we also apply different cluster quality
    metrics to judge the goodness of fit of the cluster labels to the ground truth.

    Cluster quality metrics evaluated (see `clustering_evaluation` for
    definitions and discussions of the metrics):

    =========== ========================================================
    Shorthand    full name
    =========== ========================================================
    homo         homogeneity score
    compl        completeness score
    v-meas       V measure
    ARI          adjusted Rand index
    AMI          adjusted mutual information
    silhouette   silhouette coefficient
    =========== ========================================================

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
    ## Load the dataset

    We will start by loading the `digits` dataset. This dataset contains
    handwritten digits from 0 to 9. In the context of clustering, one would like
    to group images such that the handwritten digits on the image are the same.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.datasets import load_digits

    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    return data, labels, n_digits, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define our evaluation benchmark

    We will first our evaluation benchmark. During this benchmark, we intend to
    compare different initialization methods for KMeans. Our benchmark will:

    * create a pipeline which will scale the data using a
      :class:`~sklearn.preprocessing.StandardScaler`;
    * train and time the pipeline fitting;
    * measure the performance of the clustering obtained via different metrics.


    """
    )
    return


@app.cell
def _():
    from time import time
    from sklearn import metrics
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def bench_k_means(kmeans, name, data, labels):
        """Benchmark to evaluate the KMeans initialization methods.

        Parameters
        ----------
        kmeans : KMeans instance
            A :class:`~sklearn.cluster.KMeans` instance with the initialization
            already set.
        name : str
            Name given to the strategy. It will be used to show the results in a
            table.
        data : ndarray of shape (n_samples, n_features)
            The data to cluster.
        labels : ndarray of shape (n_samples,)
            The labels used to compute the clustering metrics which requires some
            supervision.
        """
        t0 = time()
        estimator = make_pipeline(StandardScaler(), _kmeans).fit(data)
        fit_time = time() - t0
        results = [name, fit_time, estimator[-1].inertia_]
        clustering_metrics = [metrics.homogeneity_score, metrics.completeness_score, metrics.v_measure_score, metrics.adjusted_rand_score, metrics.adjusted_mutual_info_score]
        results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]
        results += [metrics.silhouette_score(data, estimator[-1].labels_, metric='euclidean', sample_size=300)]
        formatter_result = '{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'  # Define the metrics which require only the true labels and estimator
        print(formatter_result.format(*results))  # labels  # The silhouette score requires the full dataset  # Show the results
    return (bench_k_means,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Run the benchmark

    We will compare three approaches:

    * an initialization using `k-means++`. This method is stochastic and we will
      run the initialization 4 times;
    * a random initialization. This method is stochastic as well and we will run
      the initialization 4 times;
    * an initialization based on a :class:`~sklearn.decomposition.PCA`
      projection. Indeed, we will use the components of the
      :class:`~sklearn.decomposition.PCA` to initialize KMeans. This method is
      deterministic and a single initialization suffice.


    """
    )
    return


@app.cell
def _(bench_k_means, data, labels, n_digits):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    _kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=4, random_state=0)
    bench_k_means(kmeans=_kmeans, name='k-means++', data=data, labels=labels)
    _kmeans = KMeans(init='random', n_clusters=n_digits, n_init=4, random_state=0)
    bench_k_means(kmeans=_kmeans, name='random', data=data, labels=labels)
    pca = PCA(n_components=n_digits).fit(data)
    _kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
    bench_k_means(kmeans=_kmeans, name='PCA-based', data=data, labels=labels)
    print(82 * '_')
    return KMeans, PCA


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualize the results on PCA-reduced data

    :class:`~sklearn.decomposition.PCA` allows to project the data from the
    original 64-dimensional space into a lower dimensional space. Subsequently,
    we can use :class:`~sklearn.decomposition.PCA` to project into a
    2-dimensional space and plot the data and the clusters in this new space.


    """
    )
    return


@app.cell
def _(KMeans, PCA, data, n_digits, np):
    import matplotlib.pyplot as plt
    reduced_data = PCA(n_components=2).fit_transform(data)
    _kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=4)
    _kmeans.fit(reduced_data)
    h = 0.02
    x_min, x_max = (reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    y_min, y_max = (reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1)  # point in the mesh [x_min, x_max]x[y_min, y_max].
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Plot the decision boundary. For that, we will assign a color to each
    Z = _kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    # Obtain labels for each point in mesh. Use last trained model.
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Put the result into a color plot
    centroids = _kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\nCentroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    # Plot the centroids as a white X
    plt.show()
    return

if __name__ == "__main__":
    app.run()
