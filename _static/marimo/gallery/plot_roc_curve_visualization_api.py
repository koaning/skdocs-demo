import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # ROC Curve with Visualization API
    Scikit-learn defines a simple API for creating visualizations for machine
    learning. The key features of this API is to allow for quick plotting and
    visual adjustments without recalculation. In this example, we will demonstrate
    how to use the visualization API by comparing ROC curves.

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
    ## Load Data and Train a SVC
    First, we load the wine dataset and convert it to a binary classification
    problem. Then, we train a support vector classifier on a training dataset.


    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import RocCurveDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    X, y = load_wine(return_X_y=True)
    y = y == 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    svc = SVC(random_state=42)
    svc.fit(X_train, y_train)
    return (
        RandomForestClassifier,
        RocCurveDisplay,
        X_test,
        X_train,
        plt,
        svc,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting the ROC Curve
    Next, we plot the ROC curve with a single call to
    :func:`sklearn.metrics.RocCurveDisplay.from_estimator`. The returned
    `svc_disp` object allows us to continue using the already computed ROC curve
    for the SVC in future plots.


    """
    )
    return


@app.cell
def _(RocCurveDisplay, X_test, plt, svc, y_test):
    svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
    plt.show()
    return (svc_disp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training a Random Forest and Plotting the ROC Curve
    We train a random forest classifier and create a plot comparing it to the SVC
    ROC curve. Notice how `svc_disp` uses
    :func:`~sklearn.metrics.RocCurveDisplay.plot` to plot the SVC ROC curve
    without recomputing the values of the roc curve itself. Furthermore, we
    pass `alpha=0.8` to the plot functions to adjust the alpha values of the
    curves.


    """
    )
    return


@app.cell
def _(
    RandomForestClassifier,
    RocCurveDisplay,
    X_test,
    X_train,
    plt,
    svc_disp,
    y_test,
    y_train,
):
    rfc = RandomForestClassifier(n_estimators=10, random_state=42)
    rfc.fit(X_train, y_train)
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        rfc, X_test, y_test, ax=ax, curve_kwargs=dict(alpha=0.8)
    )
    svc_disp.plot(ax=ax, curve_kwargs=dict(alpha=0.8))
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
