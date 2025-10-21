import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Theil-Sen Regression

    Computes a Theil-Sen Regression on a synthetic dataset.

    See `theil_sen_regression` for more information on the regressor.

    Compared to the OLS (ordinary least squares) estimator, the Theil-Sen
    estimator is robust against outliers. It has a breakdown point of about 29.3%
    in case of a simple linear regression which means that it can tolerate
    arbitrary corrupted data (outliers) of up to 29.3% in the two-dimensional
    case.

    The estimation of the model is done by calculating the slopes and intercepts
    of a subpopulation of all possible combinations of p subsample points. If an
    intercept is fitted, p must be greater than or equal to n_features + 1. The
    final slope and intercept is then defined as the spatial median of these
    slopes and intercepts.

    In certain cases Theil-Sen performs better than `RANSAC
    <ransac_regression>` which is also a robust method. This is illustrated in the
    second example below where outliers with respect to the x-axis perturb RANSAC.
    Tuning the ``residual_threshold`` parameter of RANSAC remedies this but in
    general a priori knowledge about the data and the nature of the outliers is
    needed.
    Due to the computational complexity of Theil-Sen it is recommended to use it
    only for small problems in terms of number of samples and features. For larger
    problems the ``max_subpopulation`` parameter restricts the magnitude of all
    possible combinations of p subsample points to a randomly chosen subset and
    therefore also limits the runtime. Therefore, Theil-Sen is applicable to larger
    problems with the drawback of losing some of its mathematical properties since
    it then works on a random subset.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import time

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor

    estimators = [
        ("OLS", LinearRegression()),
        ("Theil-Sen", TheilSenRegressor(random_state=42)),
        ("RANSAC", RANSACRegressor(random_state=42)),
    ]
    colors = {"OLS": "turquoise", "Theil-Sen": "gold", "RANSAC": "lightgreen"}
    lw = 2
    return colors, estimators, lw, np, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Outliers only in the y direction


    """
    )
    return


@app.cell
def _(colors, estimators, lw, np, plt, time):
    np.random.seed(0)
    n_samples = 200
    # Linear model y = 3*x + N(2, 0.1**2)
    _x = np.random.randn(n_samples)
    w = 3.0
    c = 2.0
    _noise = 0.1 * np.random.randn(n_samples)
    _y = w * _x + c + _noise
    # 10% outliers
    _y[-20:] += -20 * _x[-20:]
    _X = _x[:, np.newaxis]
    plt.scatter(_x, _y, color='indigo', marker='x', s=40)
    _line_x = np.array([-3, 3])
    for _name, _estimator in estimators:
        _t0 = time.time()
        _estimator.fit(_X, _y)
        _elapsed_time = time.time() - _t0
        _y_pred = _estimator.predict(_line_x.reshape(2, 1))
        plt.plot(_line_x, _y_pred, color=colors[_name], linewidth=lw, label='%s (fit time: %.2fs)' % (_name, _elapsed_time))
    plt.axis('tight')
    plt.legend(loc='upper right')
    _ = plt.title('Corrupt y')
    return (n_samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Outliers in the X direction


    """
    )
    return


@app.cell
def _(colors, estimators, lw, n_samples, np, plt, time):
    np.random.seed(0)
    # Linear model y = 3*x + N(2, 0.1**2)
    _x = np.random.randn(n_samples)
    _noise = 0.1 * np.random.randn(n_samples)
    _y = 3 * _x + 2 + _noise
    # 10% outliers
    _x[-20:] = 9.9
    _y[-20:] += 22
    _X = _x[:, np.newaxis]
    plt.figure()
    plt.scatter(_x, _y, color='indigo', marker='x', s=40)
    _line_x = np.array([-3, 10])
    for _name, _estimator in estimators:
        _t0 = time.time()
        _estimator.fit(_X, _y)
        _elapsed_time = time.time() - _t0
        _y_pred = _estimator.predict(_line_x.reshape(2, 1))
        plt.plot(_line_x, _y_pred, color=colors[_name], linewidth=lw, label='%s (fit time: %.2fs)' % (_name, _elapsed_time))
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.title('Corrupt x')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
