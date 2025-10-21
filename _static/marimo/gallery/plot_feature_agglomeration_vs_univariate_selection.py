import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Feature agglomeration vs. univariate selection

    This example compares 2 dimensionality reduction strategies:

    - univariate feature selection with Anova

    - feature agglomeration with Ward hierarchical clustering

    Both methods are compared in a regression problem using
    a BayesianRidge as supervised estimator.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause
    return


@app.cell
def _():
    import shutil
    import tempfile

    import matplotlib.pyplot as plt
    import numpy as np
    from joblib import Memory
    from scipy import linalg, ndimage

    from sklearn import feature_selection
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.feature_extraction.image import grid_to_graph
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    return (
        BayesianRidge,
        FeatureAgglomeration,
        GridSearchCV,
        KFold,
        Memory,
        Pipeline,
        feature_selection,
        grid_to_graph,
        linalg,
        ndimage,
        np,
        plt,
        shutil,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Set parameters


    """
    )
    return


@app.cell
def _(np):
    n_samples = 200
    size = 40  # image size
    roi_size = 15
    snr = 5.0
    np.random.seed(0)
    return n_samples, roi_size, size, snr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Generate data


    """
    )
    return


@app.cell
def _(n_samples, ndimage, np, roi_size, size):
    coef = np.zeros((size, size))
    coef[0:roi_size, 0:roi_size] = -1.0
    coef[-roi_size:, -roi_size:] = 1.0
    X = np.random.randn(n_samples, size ** 2)
    for x in X:
        x[:] = ndimage.gaussian_filter(x.reshape(size, size), sigma=1.0).ravel()  # smooth data
    X = X - X.mean(axis=0)
    X = X / X.std(axis=0)
    y = np.dot(X, coef.ravel())
    return X, coef, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    add noise


    """
    )
    return


@app.cell
def _(linalg, np, snr, y):
    noise = np.random.randn(y.shape[0])
    noise_coef = linalg.norm(y, 2) / np.exp(snr / 20.0) / linalg.norm(noise, 2)
    y_1 = y + noise_coef * noise
    return (y_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Compute the coefs of a Bayesian Ridge with GridSearch


    """
    )
    return


@app.cell
def _(BayesianRidge, KFold, Memory, tempfile):
    cv = KFold(2)  # cross-validation generator for model selection
    ridge = BayesianRidge()
    cachedir = tempfile.mkdtemp()
    mem = Memory(location=cachedir, verbose=1)
    return cachedir, cv, mem, ridge


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Ward agglomeration followed by BayesianRidge


    """
    )
    return


@app.cell
def _(
    FeatureAgglomeration,
    GridSearchCV,
    Pipeline,
    X,
    cv,
    grid_to_graph,
    mem,
    ridge,
    size,
    y_1,
):
    connectivity = grid_to_graph(n_x=size, n_y=size)
    ward = FeatureAgglomeration(n_clusters=10, connectivity=connectivity, memory=mem)
    _clf = Pipeline([('ward', ward), ('ridge', ridge)])
    _clf = GridSearchCV(_clf, {'ward__n_clusters': [10, 20, 30]}, n_jobs=1, cv=cv)
    _clf.fit(X, y_1)
    _coef_ = _clf.best_estimator_.steps[-1][1].coef_
    _coef_ = _clf.best_estimator_.steps[0][1].inverse_transform(_coef_)
    coef_agglomeration_ = _coef_.reshape(size, size)
    return (coef_agglomeration_,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Anova univariate feature selection followed by BayesianRidge


    """
    )
    return


@app.cell
def _(GridSearchCV, Pipeline, X, cv, feature_selection, mem, ridge, size, y_1):
    f_regression = mem.cache(feature_selection.f_regression)
    anova = feature_selection.SelectPercentile(f_regression)
    _clf = Pipeline([('anova', anova), ('ridge', ridge)])
    _clf = GridSearchCV(_clf, {'anova__percentile': [5, 10, 20]}, cv=cv)
    _clf.fit(X, y_1)
    _coef_ = _clf.best_estimator_.steps[-1][1].coef_
    _coef_ = _clf.best_estimator_.steps[0][1].inverse_transform(_coef_.reshape(1, -1))
    coef_selection_ = _coef_.reshape(size, size)
    return (coef_selection_,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Inverse the transformation to plot the results on an image


    """
    )
    return


@app.cell
def _(coef, coef_agglomeration_, coef_selection_, plt):
    plt.close("all")
    plt.figure(figsize=(7.3, 2.7))
    plt.subplot(1, 3, 1)
    plt.imshow(coef, interpolation="nearest", cmap=plt.cm.RdBu_r)
    plt.title("True weights")
    plt.subplot(1, 3, 2)
    plt.imshow(coef_selection_, interpolation="nearest", cmap=plt.cm.RdBu_r)
    plt.title("Feature Selection")
    plt.subplot(1, 3, 3)
    plt.imshow(coef_agglomeration_, interpolation="nearest", cmap=plt.cm.RdBu_r)
    plt.title("Feature Agglomeration")
    plt.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.16, 0.26)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Attempt to remove the temporary cachedir, but don't worry if it fails


    """
    )
    return


@app.cell
def _(cachedir, shutil):
    shutil.rmtree(cachedir, ignore_errors=True)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
