import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Non-negative least squares

    In this example, we fit a linear model with positive constraints on the
    regression coefficients and compare the estimated coefficients to a classic
    linear regression.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.metrics import r2_score
    return np, plt, r2_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Generate some random data


    """
    )
    return


@app.cell
def _(np):
    np.random.seed(42)

    n_samples, n_features = 200, 50
    X = np.random.randn(n_samples, n_features)
    true_coef = 3 * np.random.randn(n_features)
    # Threshold coefficients to render them non-negative
    true_coef[true_coef < 0] = 0
    y = np.dot(X, true_coef)

    # Add some noise
    y += 5 * np.random.normal(size=(n_samples,))
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Split the data in train set and test set


    """
    )
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fit the Non-Negative least squares.


    """
    )
    return


@app.cell
def _(X_test, X_train, r2_score, y_test, y_train):
    from sklearn.linear_model import LinearRegression

    reg_nnls = LinearRegression(positive=True)
    y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
    r2_score_nnls = r2_score(y_test, y_pred_nnls)
    print("NNLS R2 score", r2_score_nnls)
    return LinearRegression, reg_nnls


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fit an OLS.


    """
    )
    return


@app.cell
def _(LinearRegression, X_test, X_train, r2_score, y_test, y_train):
    reg_ols = LinearRegression()
    y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
    r2_score_ols = r2_score(y_test, y_pred_ols)
    print("OLS R2 score", r2_score_ols)
    return (reg_ols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Comparing the regression coefficients between OLS and NNLS, we can observe
    they are highly correlated (the dashed line is the identity relation),
    but the non-negative constraint shrinks some to 0.
    The Non-Negative Least squares inherently yield sparse results.


    """
    )
    return


@app.cell
def _(plt, reg_nnls, reg_ols):
    fig, ax = plt.subplots()
    ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth=0, marker=".")

    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    low = max(low_x, low_y)
    high = min(high_x, high_y)
    ax.plot([low, high], [low, high], ls="--", c=".3", alpha=0.5)
    ax.set_xlabel("OLS regression coefficients", fontweight="bold")
    ax.set_ylabel("NNLS regression coefficients", fontweight="bold")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
