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

    # SVM with custom kernel

    Simple usage of Support Vector Machines to classify a sample. It will
    plot the decision surface and the support vectors.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn import datasets, svm
    from sklearn.inspection import DecisionBoundaryDisplay

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    Y = iris.target


    def my_kernel(X, Y):
        """
        We create a custom kernel:

                     (2  0)
        k(X, Y) = X  (    ) Y.T
                     (0  1)
        """
        M = np.array([[2, 0], [0, 1.0]])
        return np.dot(np.dot(X, M), Y.T)


    h = 0.02  # step size in the mesh

    # we create an instance of SVM and fit out data.
    clf = svm.SVC(kernel=my_kernel)
    clf.fit(X, Y)

    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.Paired,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
    )

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
    plt.title("3-Class classification using Support Vector Machine with custom kernel")
    plt.axis("tight")
    plt.show()
    return

if __name__ == "__main__":
    app.run()
