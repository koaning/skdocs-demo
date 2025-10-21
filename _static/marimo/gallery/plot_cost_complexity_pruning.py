import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Post pruning decision trees with cost complexity pruning

    .. currentmodule:: sklearn.tree

    The :class:`DecisionTreeClassifier` provides parameters such as
    ``min_samples_leaf`` and ``max_depth`` to prevent a tree from overfiting. Cost
    complexity pruning provides another option to control the size of a tree. In
    :class:`DecisionTreeClassifier`, this pruning technique is parameterized by the
    cost complexity parameter, ``ccp_alpha``. Greater values of ``ccp_alpha``
    increase the number of nodes pruned. Here we only show the effect of
    ``ccp_alpha`` on regularizing the trees and how to choose a ``ccp_alpha``
    based on validation scores.

    See also `minimal_cost_complexity_pruning` for details on pruning.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier, load_breast_cancer, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Total impurity of leaves vs effective alphas of pruned tree
    Minimal cost complexity pruning recursively finds the node with the "weakest
    link". The weakest link is characterized by an effective alpha, where the
    nodes with the smallest effective alpha are pruned first. To get an idea of
    what values of ``ccp_alpha`` could be appropriate, scikit-learn provides
    :func:`DecisionTreeClassifier.cost_complexity_pruning_path` that returns the
    effective alphas and the corresponding total leaf impurities at each step of
    the pruning process. As alpha increases, more of the tree is pruned, which
    increases the total impurity of its leaves.


    """
    )
    return


@app.cell
def _(DecisionTreeClassifier, load_breast_cancer, train_test_split):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    _clf = DecisionTreeClassifier(random_state=0)
    path = _clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = (path.ccp_alphas, path.impurities)
    return X_test, X_train, ccp_alphas, impurities, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the following plot, the maximum effective alpha value is removed, because
    it is the trivial tree with only one node.


    """
    )
    return


@app.cell
def _(ccp_alphas, impurities, plt):
    _fig, _ax = plt.subplots()
    _ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle='steps-post')
    _ax.set_xlabel('effective alpha')
    _ax.set_ylabel('total impurity of leaves')
    _ax.set_title('Total Impurity vs effective alpha for training set')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Next, we train a decision tree using the effective alphas. The last value
    in ``ccp_alphas`` is the alpha value that prunes the whole tree,
    leaving the tree, ``clfs[-1]``, with one node.


    """
    )
    return


@app.cell
def _(DecisionTreeClassifier, X_train, ccp_alphas, y_train):
    clfs = []
    for ccp_alpha in ccp_alphas:
        _clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        _clf.fit(X_train, y_train)
        clfs.append(_clf)
    print('Number of nodes in the last tree is: {} with ccp_alpha: {}'.format(clfs[-1].tree_.node_count, ccp_alphas[-1]))
    return (clfs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For the remainder of this example, we remove the last element in
    ``clfs`` and ``ccp_alphas``, because it is the trivial tree with only one
    node. Here we show that the number of nodes and tree depth decreases as alpha
    increases.


    """
    )
    return


@app.cell
def _(ccp_alphas, clfs, plt):
    clfs_1 = clfs[:-1]
    ccp_alphas_1 = ccp_alphas[:-1]
    node_counts = [_clf.tree_.node_count for _clf in clfs_1]
    depth = [_clf.tree_.max_depth for _clf in clfs_1]
    _fig, _ax = plt.subplots(2, 1)
    _ax[0].plot(ccp_alphas_1, node_counts, marker='o', drawstyle='steps-post')
    _ax[0].set_xlabel('alpha')
    _ax[0].set_ylabel('number of nodes')
    _ax[0].set_title('Number of nodes vs alpha')
    _ax[1].plot(ccp_alphas_1, depth, marker='o', drawstyle='steps-post')
    _ax[1].set_xlabel('alpha')
    _ax[1].set_ylabel('depth of tree')
    _ax[1].set_title('Depth vs alpha')
    _fig.tight_layout()
    return ccp_alphas_1, clfs_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Accuracy vs alpha for training and testing sets
    When ``ccp_alpha`` is set to zero and keeping the other default parameters
    of :class:`DecisionTreeClassifier`, the tree overfits, leading to
    a 100% training accuracy and 88% testing accuracy. As alpha increases, more
    of the tree is pruned, thus creating a decision tree that generalizes better.
    In this example, setting ``ccp_alpha=0.015`` maximizes the testing accuracy.


    """
    )
    return


@app.cell
def _(X_test, X_train, ccp_alphas_1, clfs_1, plt, y_test, y_train):
    train_scores = [_clf.score(X_train, y_train) for _clf in clfs_1]
    test_scores = [_clf.score(X_test, y_test) for _clf in clfs_1]
    _fig, _ax = plt.subplots()
    _ax.set_xlabel('alpha')
    _ax.set_ylabel('accuracy')
    _ax.set_title('Accuracy vs alpha for training and testing sets')
    _ax.plot(ccp_alphas_1, train_scores, marker='o', label='train', drawstyle='steps-post')
    _ax.plot(ccp_alphas_1, test_scores, marker='o', label='test', drawstyle='steps-post')
    _ax.legend()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
