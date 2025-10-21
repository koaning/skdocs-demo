import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Nearest Neighbors regression

    Demonstrate the resolution of a regression problem
    using a k-Nearest Neighbor and the interpolation of the
    target using both barycenter and constant weights.

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
    ## Generate sample data
    Here we generate a few data points to use to train the model. We also generate
    data in the whole range of the training data to visualize how the model would
    react in that whole region.


    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn import neighbors

    rng = np.random.RandomState(0)
    X_train = np.sort(5 * rng.rand(40, 1), axis=0)
    X_test = np.linspace(0, 5, 500)[:, np.newaxis]
    y = np.sin(X_train).ravel()

    # Add noise to targets
    y[::5] += 1 * (0.5 - np.random.rand(8))
    return X_test, X_train, neighbors, plt, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fit regression model
    Here we train a model and visualize how `uniform` and `distance`
    weights in prediction effect predicted values.


    """
    )
    return


@app.cell
def _(X_test, X_train, neighbors, plt, y):
    n_neighbors = 5

    for i, weights in enumerate(["uniform", "distance"]):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X_train, y).predict(X_test)

        plt.subplot(2, 1, i + 1)
        plt.scatter(X_train, y, color="darkorange", label="data")
        plt.plot(X_test, y_, color="navy", label="prediction")
        plt.axis("tight")
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
