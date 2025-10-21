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

    # A demo of the Spectral Biclustering algorithm

    This example demonstrates how to generate a checkerboard dataset and bicluster
    it using the :class:`~sklearn.cluster.SpectralBiclustering` algorithm. The
    spectral biclustering algorithm is specifically designed to cluster data by
    simultaneously considering both the rows (samples) and columns (features) of a
    matrix. It aims to identify patterns not only between samples but also within
    subsets of samples, allowing for the detection of localized structure within the
    data. This makes spectral biclustering particularly well-suited for datasets
    where the order or arrangement of features is fixed, such as in images, time
    series, or genomes.

    The data is generated, then shuffled and passed to the spectral biclustering
    algorithm. The rows and columns of the shuffled matrix are then rearranged to
    plot the biclusters found.

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
    We generate the sample data using the
    :func:`~sklearn.datasets.make_checkerboard` function. Each pixel within
    `shape=(300, 300)` represents with its color a value from a uniform
    distribution. The noise is added from a normal distribution, where the value
    chosen for `noise` is the standard deviation.

    As you can see, the data is distributed over 12 cluster cells and is
    relatively well distinguishable.


    """
    )
    return


@app.cell
def _():
    from matplotlib import pyplot as plt

    from sklearn.datasets import make_checkerboard

    n_clusters = (4, 3)
    data, rows, columns = make_checkerboard(
        shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=42
    )

    plt.matshow(data, cmap=plt.cm.Blues)
    plt.title("Original dataset")
    plt.show()
    return columns, data, n_clusters, plt, rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We shuffle the data and the goal is to reconstruct it afterwards using
    :class:`~sklearn.cluster.SpectralBiclustering`.


    """
    )
    return


@app.cell
def _(data):
    import numpy as np

    # Creating lists of shuffled row and column indices
    rng = np.random.RandomState(0)
    row_idx_shuffled = rng.permutation(data.shape[0])
    col_idx_shuffled = rng.permutation(data.shape[1])
    return col_idx_shuffled, np, row_idx_shuffled


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We redefine the shuffled data and plot it. We observe that we lost the
    structure of original data matrix.


    """
    )
    return


@app.cell
def _(col_idx_shuffled, data, plt, row_idx_shuffled):
    data_1 = data[row_idx_shuffled][:, col_idx_shuffled]
    plt.matshow(data_1, cmap=plt.cm.Blues)
    plt.title('Shuffled dataset')
    plt.show()
    return (data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fitting `SpectralBiclustering`
    We fit the model and compare the obtained clusters with the ground truth. Note
    that when creating the model we specify the same number of clusters that we
    used to create the dataset (`n_clusters = (4, 3)`), which will contribute to
    obtain a good result.


    """
    )
    return


@app.cell
def _(col_idx_shuffled, columns, data_1, n_clusters, row_idx_shuffled, rows):
    from sklearn.cluster import SpectralBiclustering
    from sklearn.metrics import consensus_score
    model = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=0)
    model.fit(data_1)
    score = consensus_score(model.biclusters_, (rows[:, row_idx_shuffled], columns[:, col_idx_shuffled]))
    # Compute the similarity of two sets of biclusters
    print(f'consensus score: {score:.1f}')
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The score is between 0 and 1, where 1 corresponds to a perfect matching. It
    shows the quality of the biclustering.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting results
    Now, we rearrange the data based on the row and column labels assigned by the
    :class:`~sklearn.cluster.SpectralBiclustering` model in ascending order and
    plot again. The `row_labels_` range from 0 to 3, while the `column_labels_`
    range from 0 to 2, representing a total of 4 clusters per row and 3 clusters
    per column.


    """
    )
    return


@app.cell
def _(data_1, model, np, plt):
    # Reordering first the rows and then the columns.
    reordered_rows = data_1[np.argsort(model.row_labels_)]
    reordered_data = reordered_rows[:, np.argsort(model.column_labels_)]
    plt.matshow(reordered_data, cmap=plt.cm.Blues)
    plt.title('After biclustering; rearranged to show biclusters')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As a last step, we want to demonstrate the relationships between the row
    and column labels assigned by the model. Therefore, we create a grid with
    :func:`numpy.outer`, which takes the sorted `row_labels_` and `column_labels_`
    and adds 1 to each to ensure that the labels start from 1 instead of 0 for
    better visualization.


    """
    )
    return


@app.cell
def _(model, np, plt):
    plt.matshow(
        np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1),
        cmap=plt.cm.Blues,
    )
    plt.title("Checkerboard structure of rearranged data")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The outer product of the row and column label vectors shows a representation
    of the checkerboard structure, where different combinations of row and column
    labels are represented by different shades of blue.


    """
    )
    return

if __name__ == "__main__":
    app.run()
