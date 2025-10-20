import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Lasso on dense and sparse data

    We show that linear_model.Lasso provides the same results for dense and sparse
    data and that in the case of sparse data the speed is improved.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    from time import time

    from scipy import linalg, sparse

    from sklearn.datasets import make_regression
    from sklearn.linear_model import Lasso
    return Lasso, linalg, make_regression, sparse, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Comparing the two Lasso implementations on Dense data

    We create a linear regression problem that is suitable for the Lasso,
    that is to say, with more features than samples. We then store the data
    matrix in both dense (the usual) and sparse format, and train a Lasso on
    each. We compute the runtime of both and check that they learned the
    same model by computing the Euclidean norm of the difference between the
    coefficients they learned. Because the data is dense, we expect better
    runtime with a dense data format.


    """
    )
    return


@app.cell
def _(Lasso, linalg, make_regression, sparse, time):
    X, y = make_regression(n_samples=200, n_features=5000, random_state=0)
    # create a copy of X in sparse format
    X_sp = sparse.coo_matrix(X)
    _alpha = 1
    _sparse_lasso = Lasso(alpha=_alpha, fit_intercept=False, max_iter=1000)
    _dense_lasso = Lasso(alpha=_alpha, fit_intercept=False, max_iter=1000)
    _t0 = time()
    _sparse_lasso.fit(X_sp, y)
    print(f'Sparse Lasso done in {time() - _t0:.3f}s')
    _t0 = time()
    _dense_lasso.fit(X, y)
    print(f'Dense Lasso done in {time() - _t0:.3f}s')
    _coeff_diff = linalg.norm(_sparse_lasso.coef_ - _dense_lasso.coef_)
    # compare the regression coefficients
    #
    print(f'Distance between coefficients : {_coeff_diff:.2e}')
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Comparing the two Lasso implementations on Sparse data

    We make the previous problem sparse by replacing all small values with 0
    and run the same comparisons as above. Because the data is now sparse, we
    expect the implementation that uses the sparse data format to be faster.


    """
    )
    return


@app.cell
def _(Lasso, X, linalg, sparse, time, y):
    # make a copy of the previous data
    Xs = X.copy()
    # make Xs sparse by replacing the values lower than 2.5 with 0s
    Xs[Xs < 2.5] = 0.0
    # create a copy of Xs in sparse format
    Xs_sp = sparse.coo_matrix(Xs)
    Xs_sp = Xs_sp.tocsc()
    print(f'Matrix density : {Xs_sp.nnz / float(X.size) * 100:.3f}%')
    # compute the proportion of non-zero coefficient in the data matrix
    _alpha = 0.1
    _sparse_lasso = Lasso(alpha=_alpha, fit_intercept=False, max_iter=10000)
    _dense_lasso = Lasso(alpha=_alpha, fit_intercept=False, max_iter=10000)
    _t0 = time()
    _sparse_lasso.fit(Xs_sp, y)
    print(f'Sparse Lasso done in {time() - _t0:.3f}s')
    _t0 = time()
    _dense_lasso.fit(Xs, y)
    print(f'Dense Lasso done in  {time() - _t0:.3f}s')
    _coeff_diff = linalg.norm(_sparse_lasso.coef_ - _dense_lasso.coef_)
    # compare the regression coefficients
    print(f'Distance between coefficients : {_coeff_diff:.2e}')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
