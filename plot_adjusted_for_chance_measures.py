import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Adjustment for chance in clustering performance evaluation
    This notebook explores the impact of uniformly-distributed random labeling on
    the behavior of some clustering evaluation metrics. For such purpose, the
    metrics are computed with a fixed number of samples and as a function of the number
    of clusters assigned by the estimator. The example is divided into two
    experiments:

    - a first experiment with fixed "ground truth labels" (and therefore fixed
      number of classes) and randomly "predicted labels";
    - a second experiment with varying "ground truth labels", randomly "predicted
      labels". The "predicted labels" have the same number of classes and clusters
      as the "ground truth labels".

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
    ## Defining the list of metrics to evaluate

    Clustering algorithms are fundamentally unsupervised learning methods.
    However, since we assign class labels for the synthetic clusters in this
    example, it is possible to use evaluation metrics that leverage this
    "supervised" ground truth information to quantify the quality of the resulting
    clusters. Examples of such metrics are the following:

    - V-measure, the harmonic mean of completeness and homogeneity;

    - Rand index, which measures how frequently pairs of data points are grouped
      consistently according to the result of the clustering algorithm and the
      ground truth class assignment;

    - Adjusted Rand index (ARI), a chance-adjusted Rand index such that a random
      cluster assignment has an ARI of 0.0 in expectation;

    - Mutual Information (MI) is an information theoretic measure that quantifies
      how dependent are the two labelings. Note that the maximum value of MI for
      perfect labelings depends on the number of clusters and samples;

    - Normalized Mutual Information (NMI), a Mutual Information defined between 0
      (no mutual information) in the limit of large number of data points and 1
      (perfectly matching label assignments, up to a permutation of the labels).
      It is not adjusted for chance: then the number of clustered data points is
      not large enough, the expected values of MI or NMI for random labelings can
      be significantly non-zero;

    - Adjusted Mutual Information (AMI), a chance-adjusted Mutual Information.
      Similarly to ARI, random cluster assignment has an AMI of 0.0 in
      expectation.

    For more information, see the `clustering_evaluation` module.


    """
    )
    return


@app.cell
def _():
    from sklearn import metrics

    score_funcs = [
        ("V-measure", metrics.v_measure_score),
        ("Rand index", metrics.rand_score),
        ("ARI", metrics.adjusted_rand_score),
        ("MI", metrics.mutual_info_score),
        ("NMI", metrics.normalized_mutual_info_score),
        ("AMI", metrics.adjusted_mutual_info_score),
    ]
    return (score_funcs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## First experiment: fixed ground truth labels and growing number of clusters

    We first define a function that creates uniformly-distributed random labeling.


    """
    )
    return


@app.cell
def _():
    import numpy as np
    rng = np.random.RandomState(0)

    def random_labels(n_samples, n_classes):
        return rng.randint(low=0, high=n_classes, size=_n_samples)
    return np, random_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Another function will use the `random_labels` function to create a fixed set
    of ground truth labels (`labels_a`) distributed in `n_classes` and then score
    several sets of randomly "predicted" labels (`labels_b`) to assess the
    variability of a given metric at a given `n_clusters`.


    """
    )
    return


@app.cell
def _(np, random_labels):
    def fixed_classes_uniform_labelings_scores(score_func, n_samples, n_clusters_range, n_classes, n_runs=5):
        _scores = np.zeros((len(_n_clusters_range), n_runs))
        labels_a = random_labels(n_samples=_n_samples, n_classes=n_classes)
        for i, n_clusters in enumerate(_n_clusters_range):
            for j in range(n_runs):
                labels_b = random_labels(n_samples=_n_samples, n_classes=n_clusters)
                _scores[i, j] = _score_func(labels_a, labels_b)
        return _scores
    return (fixed_classes_uniform_labelings_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this first example we set the number of classes (true number of clusters) to
    `n_classes=10`. The number of clusters varies over the values provided by
    `n_clusters_range`.


    """
    )
    return


@app.cell
def _(fixed_classes_uniform_labelings_scores, np, score_funcs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    _n_samples = 1000
    n_classes = 10
    _n_clusters_range = np.linspace(2, 100, 10).astype(int)
    _plots = []
    _names = []
    sns.color_palette('colorblind')
    plt.figure(1)
    for _marker, (_score_name, _score_func) in zip('d^vx.,', score_funcs):
        _scores = fixed_classes_uniform_labelings_scores(_score_func, _n_samples, _n_clusters_range, n_classes=n_classes)
        _plots.append(plt.errorbar(_n_clusters_range, _scores.mean(axis=1), _scores.std(axis=1), alpha=0.8, linewidth=1, marker=_marker)[0])
        _names.append(_score_name)
    plt.title(f'Clustering measures for random uniform labeling\nagainst reference assignment with {n_classes} classes')
    plt.xlabel(f'Number of clusters (Number of samples is fixed to {_n_samples})')
    plt.ylabel('Score value')
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(_plots, _names, bbox_to_anchor=(0.5, 0.5))
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The Rand index saturates for `n_clusters` > `n_classes`. Other non-adjusted
    measures such as the V-Measure show a linear dependency between the number of
    clusters and the number of samples.

    Adjusted for chance measure, such as ARI and AMI, display some random
    variations centered around a mean score of 0.0, independently of the number of
    samples and clusters.

    ## Second experiment: varying number of classes and clusters

    In this section we define a similar function that uses several metrics to
    score 2 uniformly-distributed random labelings. In this case, the number of
    classes and assigned number of clusters are matched for each possible value in
    `n_clusters_range`.


    """
    )
    return


@app.cell
def _(np, random_labels):
    def uniform_labelings_scores(score_func, n_samples, n_clusters_range, n_runs=5):
        _scores = np.zeros((len(_n_clusters_range), n_runs))
        for i, n_clusters in enumerate(_n_clusters_range):
            for j in range(n_runs):
                labels_a = random_labels(n_samples=_n_samples, n_classes=n_clusters)
                labels_b = random_labels(n_samples=_n_samples, n_classes=n_clusters)
                _scores[i, j] = _score_func(labels_a, labels_b)
        return _scores
    return (uniform_labelings_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this case, we use `n_samples=100` to show the effect of having a number of
    clusters similar or equal to the number of samples.


    """
    )
    return


@app.cell
def _(np, plt, score_funcs, uniform_labelings_scores):
    _n_samples = 100
    _n_clusters_range = np.linspace(2, _n_samples, 10).astype(int)
    plt.figure(2)
    _plots = []
    _names = []
    for _marker, (_score_name, _score_func) in zip('d^vx.,', score_funcs):
        _scores = uniform_labelings_scores(_score_func, _n_samples, _n_clusters_range)
        _plots.append(plt.errorbar(_n_clusters_range, np.median(_scores, axis=1), _scores.std(axis=1), alpha=0.8, linewidth=2, marker=_marker)[0])
        _names.append(_score_name)
    plt.title('Clustering measures for 2 random uniform labelings\nwith equal number of clusters')
    plt.xlabel(f'Number of clusters (Number of samples is fixed to {_n_samples})')
    plt.ylabel('Score value')
    plt.legend(_plots, _names)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We observe similar results as for the first experiment: adjusted for chance
    metrics stay constantly near zero while other metrics tend to get larger with
    finer-grained labelings. The mean V-measure of random labeling increases
    significantly as the number of clusters is closer to the total number of
    samples used to compute the measure. Furthermore, raw Mutual Information is
    unbounded from above and its scale depends on the dimensions of the clustering
    problem and the cardinality of the ground truth classes. This is why the
    curve goes off the chart.

    Only adjusted measures can hence be safely used as a consensus index to
    evaluate the average stability of clustering algorithms for a given value of k
    on various overlapping sub-samples of the dataset.

    Non-adjusted clustering evaluation metric can therefore be misleading as they
    output large values for fine-grained labelings, one could be lead to think
    that the labeling has captured meaningful groups while they can be totally
    random. In particular, such non-adjusted metrics should not be used to compare
    the results of different clustering algorithms that output a different number
    of clusters.


    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
