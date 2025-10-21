import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Factor Analysis (with rotation) to visualize patterns

    Investigating the Iris dataset, we see that sepal length, petal
    length and petal width are highly correlated. Sepal width is
    less redundant. Matrix decomposition techniques can uncover
    these latent patterns. Applying rotations to the resulting
    components does not inherently improve the predictive value
    of the derived latent space, but can help visualise their
    structure; here, for example, the varimax rotation, which
    is found by maximizing the squared variances of the weights,
    finds a structure where the second component only loads
    positively on sepal width.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    return FactorAnalysis, PCA, StandardScaler, load_iris, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Load Iris data


    """
    )
    return


@app.cell
def _(StandardScaler, load_iris):
    data = load_iris()
    X = StandardScaler().fit_transform(data["data"])
    feature_names = data["feature_names"]
    return X, feature_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Plot covariance of Iris features


    """
    )
    return


@app.cell
def _(X, feature_names, np, plt):
    _ax = plt.axes()
    im = _ax.imshow(np.corrcoef(X.T), cmap='RdBu_r', vmin=-1, vmax=1)
    _ax.set_xticks([0, 1, 2, 3])
    _ax.set_xticklabels(list(feature_names), rotation=90)
    _ax.set_yticks([0, 1, 2, 3])
    _ax.set_yticklabels(list(feature_names))
    plt.colorbar(im).ax.set_ylabel('$r$', rotation=0)
    _ax.set_title('Iris feature correlation matrix')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Run factor analysis with Varimax rotation


    """
    )
    return


@app.cell
def _(FactorAnalysis, PCA, X, feature_names, np, plt):
    n_comps = 2
    methods = [('PCA', PCA()), ('Unrotated FA', FactorAnalysis()), ('Varimax FA', FactorAnalysis(rotation='varimax'))]
    fig, axes = plt.subplots(ncols=len(methods), figsize=(10, 8), sharey=True)
    for _ax, (method, fa) in zip(axes, methods):
        fa.set_params(n_components=n_comps)
        fa.fit(X)
        components = fa.components_.T
        print('\n\n %s :\n' % method)
        print(components)
        vmax = np.abs(components).max()
        _ax.imshow(components, cmap='RdBu_r', vmax=vmax, vmin=-vmax)
        _ax.set_yticks(np.arange(len(feature_names)))
        _ax.set_yticklabels(feature_names)
        _ax.set_title(str(method))
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(['Comp. 1', 'Comp. 2'])
    fig.suptitle('Factors')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
