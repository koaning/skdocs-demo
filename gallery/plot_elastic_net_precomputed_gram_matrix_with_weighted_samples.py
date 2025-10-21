import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Fitting an Elastic Net with a precomputed Gram Matrix and Weighted Samples

    The following example shows how to precompute the gram matrix
    while using weighted samples with an :class:`~sklearn.linear_model.ElasticNet`.

    If weighted samples are used, the design matrix must be centered and then
    rescaled by the square root of the weight vector before the gram matrix
    is computed.

    <div class="alert alert-info"><h4>Note</h4><p>`sample_weight` vector is also rescaled to sum to `n_samples`, see the
       documentation for the `sample_weight` parameter to
       :meth:`~sklearn.linear_model.ElasticNet.fit`.</p></div>

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
    Let's start by loading the dataset and creating some sample weights.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.datasets import make_regression

    rng = np.random.RandomState(0)

    n_samples = int(1e5)
    X, y = make_regression(n_samples=n_samples, noise=0.5, random_state=rng)

    sample_weight = rng.lognormal(size=n_samples)
    # normalize the sample weights
    normalized_weights = sample_weight * (n_samples / (sample_weight.sum()))
    return X, normalized_weights, np, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To fit the elastic net using the `precompute` option together with the sample
    weights, we must first center the design matrix,  and rescale it by the
    normalized weights prior to computing the gram matrix.


    """
    )
    return


@app.cell
def _(X, normalized_weights, np):
    X_offset = np.average(X, axis=0, weights=normalized_weights)
    X_centered = X - np.average(X, axis=0, weights=normalized_weights)
    X_scaled = X_centered * np.sqrt(normalized_weights)[:, np.newaxis]
    gram = np.dot(X_scaled.T, X_scaled)
    return X_centered, gram


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can now proceed with fitting. We must passed the centered design matrix to
    `fit` otherwise the elastic net estimator will detect that it is uncentered
    and discard the gram matrix we passed. However, if we pass the scaled design
    matrix, the preprocessing code will incorrectly rescale it a second time.


    """
    )
    return


@app.cell
def _(X_centered, gram, normalized_weights, y):
    from sklearn.linear_model import ElasticNet

    lm = ElasticNet(alpha=0.01, precompute=gram)
    lm.fit(X_centered, y, sample_weight=normalized_weights)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
