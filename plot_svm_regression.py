import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Support Vector Regression (SVR) using linear and non-linear kernels

    Toy example of 1D regression using linear, polynomial and RBF kernels.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.svm import SVR
    return SVR, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Generate sample data


    """
    )
    return


@app.cell
def _(np):
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    # add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(8))
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fit regression model


    """
    )
    return


@app.cell
def _(SVR):
    svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svr_lin = SVR(kernel="linear", C=100, gamma="auto")
    svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
    return svr_lin, svr_poly, svr_rbf


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Look at the results


    """
    )
    return


@app.cell
def _(X, np, plt, svr_lin, svr_poly, svr_rbf, y):
    lw = 2

    svrs = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ["RBF", "Linear", "Polynomial"]
    model_color = ["m", "c", "g"]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(
            X,
            svr.fit(X, y).predict(X),
            color=model_color[ix],
            lw=lw,
            label="{} model".format(kernel_label[ix]),
        )
        axes[ix].scatter(
            X[svr.support_],
            y[svr.support_],
            facecolor="none",
            edgecolor=model_color[ix],
            s=50,
            label="{} support vectors".format(kernel_label[ix]),
        )
        axes[ix].scatter(
            X[np.setdiff1d(np.arange(len(X)), svr.support_)],
            y[np.setdiff1d(np.arange(len(X)), svr.support_)],
            facecolor="none",
            edgecolor="k",
            s=50,
            label="other training data",
        )
        axes[ix].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=1,
            fancybox=True,
            shadow=True,
        )

    fig.text(0.5, 0.04, "data", ha="center", va="center")
    fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
