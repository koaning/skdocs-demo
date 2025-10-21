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

    # Regularization path of L1- Logistic Regression


    Train l1-penalized logistic regression models on a binary classification
    problem derived from the Iris dataset.

    The models are ordered from strongest regularized to least regularized. The 4
    coefficients of the models are collected and plotted as a "regularization
    path": on the left-hand side of the figure (strong regularizers), all the
    coefficients are exactly 0. When regularization gets progressively looser,
    coefficients can get non-zero values one after the other.

    Here we choose the liblinear solver because it can efficiently optimize for the
    Logistic Regression loss with a non-smooth, sparsity inducing l1 penalty.

    Also note that we set a low value for the tolerance to make sure that the model
    has converged before collecting the coefficients.

    We also use warm_start=True which means that the coefficients of the models are
    reused to initialize the next model fit to speed-up the computation of the
    full-path.

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
    ## Load data


    """
    )
    return


@app.cell
def _():
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    return X, feature_names, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here we remove the third class to make the problem a binary classification


    """
    )
    return


@app.cell
def _(X, y):
    X_1 = X[y != 2]
    y_1 = y[y != 2]
    return X_1, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Compute regularization path


    """
    )
    return


@app.cell
def _(X_1, y_1):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import l1_min_c
    cs = l1_min_c(X_1, y_1, loss='log') * np.logspace(0, 1, 16)
    return LogisticRegression, StandardScaler, cs, make_pipeline, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Create a pipeline with `StandardScaler` and `LogisticRegression`, to normalize
    the data before fitting a linear model, in order to speed-up convergence and
    make the coefficients comparable. Also, as a side effect, since the data is now
    centered around 0, we don't need to fit an intercept.


    """
    )
    return


@app.cell
def _(LogisticRegression, StandardScaler, X_1, cs, make_pipeline, np, y_1):
    clf = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', tol=1e-06, max_iter=int(1000000.0), warm_start=True, fit_intercept=False))
    coefs_ = []
    for c in cs:
        clf.set_params(logisticregression__C=c)
        clf.fit(X_1, y_1)
        coefs_.append(clf['logisticregression'].coef_.ravel().copy())
    coefs_ = np.array(coefs_)
    return (coefs_,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot regularization path


    """
    )
    return


@app.cell
def _(coefs_, cs, feature_names):
    import matplotlib.pyplot as plt

    # Colorblind-friendly palette (IBM Color Blind Safe palette)
    colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]

    plt.figure(figsize=(10, 6))
    for i in range(coefs_.shape[1]):
        plt.semilogx(cs, coefs_[:, i], marker="o", color=colors[i], label=feature_names[i])

    ymin, ymax = plt.ylim()
    plt.xlabel("C")
    plt.ylabel("Coefficients")
    plt.title("Logistic Regression Path")
    plt.legend()
    plt.axis("tight")
    plt.show()
    return

if __name__ == "__main__":
    app.run()
