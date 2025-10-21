import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Plot the decision surface of decision trees trained on the iris dataset

    Plot the decision surface of a decision tree trained on pairs
    of features of the iris dataset.

    See `decision tree <tree>` for more information on the estimator.

    For each pair of iris features, the decision tree learns decision
    boundaries made of combinations of simple thresholding rules inferred from
    the training samples.

    We also show the tree structure of a model built on all of the features.

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
    First load the copy of the Iris dataset shipped with scikit-learn:


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_iris

    iris = load_iris()
    return (iris,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Display the decision functions of trees trained on all pairs of features.


    """
    )
    return


@app.cell
def _(iris):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.tree import DecisionTreeClassifier
    n_classes = 3
    plot_colors = 'ryb'
    plot_step = 0.02
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
        X = iris.data[:, pair]
        y = iris.target
        _clf = DecisionTreeClassifier().fit(X, y)
        ax = plt.subplot(2, 3, pairidx + 1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        DecisionBoundaryDisplay.from_estimator(_clf, X, cmap=plt.cm.RdYlBu, response_method='predict', ax=ax, xlabel=iris.feature_names[pair[0]], ylabel=iris.feature_names[pair[1]])
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.asarray(y == i).nonzero()
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i], edgecolor='black', s=15)
    plt.suptitle('Decision surface of decision trees trained on pairs of features')
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    _ = plt.axis('tight')
    return DecisionTreeClassifier, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Display the structure of a single decision tree trained on all the features
    together.


    """
    )
    return


@app.cell
def _(DecisionTreeClassifier, iris, plt):
    from sklearn.tree import plot_tree
    plt.figure()
    _clf = DecisionTreeClassifier().fit(iris.data, iris.target)
    plot_tree(_clf, filled=True)
    plt.title('Decision tree trained on all the iris features')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
