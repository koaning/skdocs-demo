import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Isotonic Regression

    An illustration of the isotonic regression on generated data (non-linear
    monotonic trend with homoscedastic uniform noise).

    The isotonic regression algorithm finds a non-decreasing approximation of a
    function while minimizing the mean squared error on the training data. The
    benefit of such a non-parametric model is that it does not assume any shape for
    the target function besides monotonicity. For comparison a linear regression is
    also presented.

    The plot on the right-hand side shows the model prediction function that
    results from the linear interpolation of threshold points. The threshold
    points are a subset of the training input observations and their matching
    target values are computed by the isotonic non-parametric fit.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.utils import check_random_state

    n = 100
    x = np.arange(n)
    rs = check_random_state(0)
    y = rs.randint(-50, 50, size=(n,)) + 50.0 * np.log1p(np.arange(n))
    return (
        IsotonicRegression,
        LineCollection,
        LinearRegression,
        n,
        np,
        plt,
        x,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fit IsotonicRegression and LinearRegression models:


    """
    )
    return


@app.cell
def _(IsotonicRegression, LinearRegression, np, x, y):
    ir = IsotonicRegression(out_of_bounds="clip")
    y_ = ir.fit_transform(x, y)

    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression
    return ir, lr, y_


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Plot results:


    """
    )
    return


@app.cell
def _(LineCollection, ir, lr, n, np, plt, x, y, y_):
    segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(len(y)))
    lc.set_linewidths(np.full(n, 0.5))

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))

    ax0.plot(x, y, "C0.", markersize=12)
    ax0.plot(x, y_, "C1.-", markersize=12)
    ax0.plot(x, lr.predict(x[:, np.newaxis]), "C2-")
    ax0.add_collection(lc)
    ax0.legend(("Training data", "Isotonic fit", "Linear fit"), loc="lower right")
    ax0.set_title("Isotonic regression fit on noisy data (n=%d)" % n)

    x_test = np.linspace(-10, 110, 1000)
    ax1.plot(x_test, ir.predict(x_test), "C1-")
    ax1.plot(ir.X_thresholds_, ir.y_thresholds_, "C1.", markersize=12)
    ax1.set_title("Prediction function (%d thresholds)" % len(ir.X_thresholds_))

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that we explicitly passed `out_of_bounds="clip"` to the constructor of
    `IsotonicRegression` to control the way the model extrapolates outside of the
    range of data observed in the training set. This "clipping" extrapolation can
    be seen on the plot of the decision function on the right-hand.


    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
