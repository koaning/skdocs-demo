import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # IsolationForest example

    An example using :class:`~sklearn.ensemble.IsolationForest` for anomaly
    detection.

    The `isolation_forest` is an ensemble of "Isolation Trees" that "isolate"
    observations by recursive random partitioning, which can be represented by a
    tree structure. The number of splittings required to isolate a sample is lower
    for outliers and higher for inliers.

    In the present example we demo two ways to visualize the decision boundary of an
    Isolation Forest trained on a toy dataset.

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

    We generate two clusters (each one containing `n_samples`) by randomly
    sampling the standard normal distribution as returned by
    :func:`numpy.random.randn`. One of them is spherical and the other one is
    slightly deformed.

    For consistency with the :class:`~sklearn.ensemble.IsolationForest` notation,
    the inliers (i.e. the gaussian clusters) are assigned a ground truth label `1`
    whereas the outliers (created with :func:`numpy.random.uniform`) are assigned
    the label `-1`.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.model_selection import train_test_split

    n_samples, n_outliers = 120, 40
    rng = np.random.RandomState(0)
    covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
    cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
    cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
    outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

    X = np.concatenate([cluster_1, cluster_2, outliers])
    y = np.concatenate(
        [np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    return X, X_train, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can visualize the resulting clusters:


    """
    )
    return


@app.cell
def _(X, y):
    import matplotlib.pyplot as plt

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    handles, labels = scatter.legend_elements()
    plt.axis("square")
    plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
    plt.title("Gaussian inliers with \nuniformly distributed outliers")
    plt.show()
    return handles, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training of the model


    """
    )
    return


@app.cell
def _(X_train):
    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(max_samples=100, random_state=0)
    clf.fit(X_train)
    return (clf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot discrete decision boundary

    We use the class :class:`~sklearn.inspection.DecisionBoundaryDisplay` to
    visualize a discrete decision boundary. The background color represents
    whether a sample in that given area is predicted to be an outlier
    or not. The scatter plot displays the true labels.


    """
    )
    return


@app.cell
def _(X, clf, handles, plt, y):
    from sklearn.inspection import DecisionBoundaryDisplay
    _disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', alpha=0.5)
    _disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    _disp.ax_.set_title('Binary decision boundary \nof IsolationForest')
    plt.axis('square')
    plt.legend(handles=handles, labels=['outliers', 'inliers'], title='true class')
    plt.show()
    return (DecisionBoundaryDisplay,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot path length decision boundary

    By setting the `response_method="decision_function"`, the background of the
    :class:`~sklearn.inspection.DecisionBoundaryDisplay` represents the measure of
    normality of an observation. Such score is given by the path length averaged
    over a forest of random trees, which itself is given by the depth of the leaf
    (or equivalently the number of splits) required to isolate a given sample.

    When a forest of random trees collectively produce short path lengths for
    isolating some particular samples, they are highly likely to be anomalies and
    the measure of normality is close to `0`. Similarly, large paths correspond to
    values close to `1` and are more likely to be inliers.


    """
    )
    return


@app.cell
def _(DecisionBoundaryDisplay, X, clf, handles, plt, y):
    _disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method='decision_function', alpha=0.5)
    _disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    _disp.ax_.set_title('Path length decision boundary \nof IsolationForest')
    plt.axis('square')
    plt.legend(handles=handles, labels=['outliers', 'inliers'], title='true class')
    plt.colorbar(_disp.ax_.collections[1])
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
