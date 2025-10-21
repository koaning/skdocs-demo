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

    # Decision Tree Regression
    In this example, we demonstrate the effect of changing the maximum depth of a
    decision tree on how it fits to the data. We perform this once on a 1D regression
    task and once on a multi-output regression task.

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
    ## Decision Tree on a 1D Regression Task

    Here we fit a tree on a 1D regression task.

    The `decision trees <tree>` is
    used to fit a sine curve with addition noisy observation. As a result, it
    learns local linear regressions approximating the sine curve.

    We can see that if the maximum depth of the tree (controlled by the
    `max_depth` parameter) is set too high, the decision trees learn too fine
    details of the training data and learn from the noise, i.e. they overfit.

    ### Create a random 1D dataset


    """
    )
    return


@app.cell
def _():
    import numpy as np
    _rng = np.random.RandomState(1)
    X = np.sort(5 * _rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] = y[::5] + 3 * (0.5 - _rng.rand(16))
    return X, np, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Fit regression model
    Here we fit two models with different maximum depths


    """
    )
    return


@app.cell
def _(X, y):
    from sklearn.tree import DecisionTreeRegressor

    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    return DecisionTreeRegressor, regr_1, regr_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Predict
    Get predictions on the test set


    """
    )
    return


@app.cell
def _(np, regr_1, regr_2):
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    return X_test, y_1, y_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Plot the results


    """
    )
    return


@app.cell
def _(X, X_test, y, y_1, y_2):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see, the model with a depth of 5 (yellow) learns the details of the
    training data to the point that it overfits to the noise. On the other hand,
    the model with a depth of 2 (blue) learns the major tendencies in the data well
    and does not overfit. In real use cases, you need to make sure that the tree
    is not overfitting the training data, which can be done using cross-validation.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Decision Tree Regression with Multi-Output Targets

    Here the `decision trees <tree>`
    is used to predict simultaneously the noisy `x` and `y` observations of a circle
    given a single underlying feature. As a result, it learns local linear
    regressions approximating the circle.

    We can see that if the maximum depth of the tree (controlled by the
    `max_depth` parameter) is set too high, the decision trees learn too fine
    details of the training data and learn from the noise, i.e. they overfit.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Create a random dataset


    """
    )
    return


@app.cell
def _(np):
    _rng = np.random.RandomState(1)
    X_1 = np.sort(200 * _rng.rand(100, 1) - 100, axis=0)
    y_4 = np.array([np.pi * np.sin(X_1).ravel(), np.pi * np.cos(X_1).ravel()]).T
    y_4[::5, :] = y_4[::5, :] + (0.5 - _rng.rand(20, 2))
    return X_1, y_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Fit regression model


    """
    )
    return


@app.cell
def _(DecisionTreeRegressor, X_1, y_4):
    regr_1_1 = DecisionTreeRegressor(max_depth=2)
    regr_2_1 = DecisionTreeRegressor(max_depth=5)
    regr_3 = DecisionTreeRegressor(max_depth=8)
    regr_1_1.fit(X_1, y_4)
    regr_2_1.fit(X_1, y_4)
    regr_3.fit(X_1, y_4)
    return regr_1_1, regr_2_1, regr_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Predict
    Get predictions on the test set


    """
    )
    return


@app.cell
def _(np, regr_1_1, regr_2_1, regr_3):
    X_test_1 = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
    y_1_1 = regr_1_1.predict(X_test_1)
    y_2_1 = regr_2_1.predict(X_test_1)
    y_3 = regr_3.predict(X_test_1)
    return y_1_1, y_2_1, y_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Plot the results


    """
    )
    return


@app.cell
def _(plt, y_1_1, y_2_1, y_3, y_4):
    plt.figure()
    s = 25
    plt.scatter(y_4[:, 0], y_4[:, 1], c='yellow', s=s, edgecolor='black', label='data')
    plt.scatter(y_1_1[:, 0], y_1_1[:, 1], c='cornflowerblue', s=s, edgecolor='black', label='max_depth=2')
    plt.scatter(y_2_1[:, 0], y_2_1[:, 1], c='red', s=s, edgecolor='black', label='max_depth=5')
    plt.scatter(y_3[:, 0], y_3[:, 1], c='blue', s=s, edgecolor='black', label='max_depth=8')
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.xlabel('target 1')
    plt.ylabel('target 2')
    plt.title('Multi-output Decision Tree Regression')
    plt.legend(loc='best')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see, the higher the value of `max_depth`, the more details of the data
    are caught by the model. However, the model also overfits to the data and is
    influenced by the noise.


    """
    )
    return

if __name__ == "__main__":
    app.run()
