import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Label Propagation circles: Learning a complex structure

    Example of LabelPropagation learning a complex internal structure
    to demonstrate "manifold learning". The outer circle should be
    labeled "red" and the inner circle "blue". Because both label groups
    lie inside their own distinct shape, we can see that the labels
    propagate correctly around the circle.

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
    We generate a dataset with two concentric circles. In addition, a label
    is associated with each sample of the dataset that is: 0 (belonging to
    the outer circle), 1 (belonging to the inner circle), and -1 (unknown).
    Here, all labels but two are tagged as unknown.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.datasets import make_circles

    n_samples = 200
    X, y = make_circles(n_samples=n_samples, shuffle=False)
    outer, inner = 0, 1
    labels = np.full(n_samples, -1.0)
    labels[0] = outer
    labels[-1] = inner
    return X, inner, labels, np, outer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Plot raw data


    """
    )
    return


@app.cell
def _(X, inner, labels, outer):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.scatter(
        X[labels == outer, 0],
        X[labels == outer, 1],
        color="navy",
        marker="s",
        lw=0,
        label="outer labeled",
        s=10,
    )
    plt.scatter(
        X[labels == inner, 0],
        X[labels == inner, 1],
        color="c",
        marker="s",
        lw=0,
        label="inner labeled",
        s=10,
    )
    plt.scatter(
        X[labels == -1, 0],
        X[labels == -1, 1],
        color="darkorange",
        marker=".",
        label="unlabeled",
    )
    plt.legend(scatterpoints=1, shadow=False, loc="center")
    _ = plt.title("Raw data (2 classes=outer and inner)")
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The aim of :class:`~sklearn.semi_supervised.LabelSpreading` is to associate
    a label to sample where the label is initially unknown.


    """
    )
    return


@app.cell
def _(X, labels):
    from sklearn.semi_supervised import LabelSpreading

    label_spread = LabelSpreading(kernel="knn", alpha=0.8)
    label_spread.fit(X, labels)
    return (label_spread,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, we can check which labels have been associated with each sample
    when the label was unknown.


    """
    )
    return


@app.cell
def _(X, inner, label_spread, np, outer, plt):
    output_labels = label_spread.transduction_
    output_label_array = np.asarray(output_labels)
    outer_numbers = (output_label_array == outer).nonzero()[0]
    inner_numbers = (output_label_array == inner).nonzero()[0]

    plt.figure(figsize=(4, 4))
    plt.scatter(
        X[outer_numbers, 0],
        X[outer_numbers, 1],
        color="navy",
        marker="s",
        lw=0,
        s=10,
        label="outer learned",
    )
    plt.scatter(
        X[inner_numbers, 0],
        X[inner_numbers, 1],
        color="c",
        marker="s",
        lw=0,
        s=10,
        label="inner learned",
    )
    plt.legend(scatterpoints=1, shadow=False, loc="center")
    plt.title("Labels learned with Label Spreading (KNN)")
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
