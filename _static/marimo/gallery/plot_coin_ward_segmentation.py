import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # A demo of structured Ward hierarchical clustering on an image of coins

    Compute the segmentation of a 2D image with Ward hierarchical
    clustering. The clustering is spatially constrained in order
    for each segmented region to be in one piece.

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
    ## Generate data


    """
    )
    return


@app.cell
def _():
    from skimage.data import coins

    orig_coins = coins()
    return (orig_coins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Resize it to 20% of the original size to speed up the processing
    Applying a Gaussian filter for smoothing prior to down-scaling
    reduces aliasing artifacts.


    """
    )
    return


@app.cell
def _(orig_coins):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from skimage.transform import rescale

    smoothened_coins = gaussian_filter(orig_coins, sigma=2)
    rescaled_coins = rescale(
        smoothened_coins,
        0.2,
        mode="reflect",
        anti_aliasing=False,
    )

    X = np.reshape(rescaled_coins, (-1, 1))
    return X, np, rescaled_coins


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define structure of the data

    Pixels are connected to their neighbors.


    """
    )
    return


@app.cell
def _(rescaled_coins):
    from sklearn.feature_extraction.image import grid_to_graph

    connectivity = grid_to_graph(*rescaled_coins.shape)
    return (connectivity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Compute clustering


    """
    )
    return


@app.cell
def _(X, connectivity, np, rescaled_coins):
    import time as time

    from sklearn.cluster import AgglomerativeClustering

    print("Compute structured hierarchical clustering...")
    st = time.time()
    n_clusters = 27  # number of regions
    ward = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward", connectivity=connectivity
    )
    ward.fit(X)
    label = np.reshape(ward.labels_, rescaled_coins.shape)
    print(f"Elapsed time: {time.time() - st:.3f}s")
    print(f"Number of pixels: {label.size}")
    print(f"Number of clusters: {np.unique(label).size}")
    return label, n_clusters


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot the results on an image

    Agglomerative clustering is able to segment each coin however, we have had to
    use a ``n_cluster`` larger than the number of coins because the segmentation
    is finding a large in the background.


    """
    )
    return


@app.cell
def _(label, n_clusters, rescaled_coins):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.imshow(rescaled_coins, cmap=plt.cm.gray)
    for l in range(n_clusters):
        plt.contour(
            label == l,
            colors=[
                plt.cm.nipy_spectral(l / float(n_clusters)),
            ],
        )
    plt.axis("off")
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
