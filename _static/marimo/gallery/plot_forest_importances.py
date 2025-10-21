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

    # Feature importances with a forest of trees

    This example shows the use of a forest of trees to evaluate the importance of
    features on an artificial classification task. The blue bars are the feature
    importances of the forest, along with their inter-trees variability represented
    by the error bars.

    As expected, the plot suggests that 3 features are informative, while the
    remaining are not.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data generation and model fitting
    We generate a synthetic dataset with only 3 informative features. We will
    explicitly not shuffle the dataset to ensure that the informative features
    will correspond to the three first columns of X. In addition, we will split
    our dataset into training and testing subsets.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    return X, X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A random forest classifier will be fitted to compute the feature importances.


    """
    )
    return


@app.cell
def _(X, X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier

    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    return feature_names, forest


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Feature importance based on mean decrease in impurity
    Feature importances are provided by the fitted attribute
    `feature_importances_` and they are computed as the mean and standard
    deviation of accumulation of the impurity decrease within each tree.

    <div class="alert alert-danger"><h4>Warning</h4><p>Impurity-based feature importances can be misleading for **high
        cardinality** features (many unique values). See
        `permutation_importance` as an alternative below.</p></div>


    """
    )
    return


@app.cell
def _(forest):
    import time
    import numpy as np
    _start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    _elapsed_time = time.time() - _start_time
    print(f'Elapsed time to compute the importances: {_elapsed_time:.3f} seconds')
    return importances, std, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's plot the impurity-based importance.


    """
    )
    return


@app.cell
def _(feature_names, importances, plt, std):
    import pandas as pd
    forest_importances = pd.Series(importances, index=feature_names)
    _fig, _ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=_ax)
    _ax.set_title('Feature importances using MDI')
    _ax.set_ylabel('Mean decrease in impurity')
    _fig.tight_layout()
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We observe that, as expected, the three first features are found important.

    ## Feature importance based on feature permutation
    Permutation feature importance overcomes limitations of the impurity-based
    feature importance: they do not have a bias toward high-cardinality features
    and can be computed on a left-out test set.


    """
    )
    return


@app.cell
def _(X_test, feature_names, forest, pd, time, y_test):
    from sklearn.inspection import permutation_importance
    _start_time = time.time()
    result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    _elapsed_time = time.time() - _start_time
    print(f'Elapsed time to compute the importances: {_elapsed_time:.3f} seconds')
    forest_importances_1 = pd.Series(result.importances_mean, index=feature_names)
    return forest_importances_1, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The computation for full permutation importance is more costly. Each feature is
    shuffled n times and the model is used to make predictions on the permuted data to see
    the drop in performance. Please see `permutation_importance` for more details.
    We can now plot the importance ranking.


    """
    )
    return


@app.cell
def _(forest_importances_1, plt, result):
    _fig, _ax = plt.subplots()
    forest_importances_1.plot.bar(yerr=result.importances_std, ax=_ax)
    _ax.set_title('Feature importances using permutation on full model')
    _ax.set_ylabel('Mean accuracy decrease')
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The same features are detected as most important using both methods. Although
    the relative importances vary. As seen on the plots, MDI is less likely than
    permutation importance to fully omit a feature.


    """
    )
    return

if __name__ == "__main__":
    app.run()
