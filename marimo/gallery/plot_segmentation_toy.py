import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Spectral clustering for image segmentation

    In this example, an image with connected circles is generated and
    spectral clustering is used to separate the circles.

    In these settings, the `spectral_clustering` approach solves the problem
    know as 'normalized graph cuts': the image is seen as a graph of
    connected voxels, and the spectral clustering algorithm amounts to
    choosing graph cuts defining regions while minimizing the ratio of the
    gradient along the cut, and the volume of the region.

    As the algorithm tries to balance the volume (ie balance the region
    sizes), if we take circles with different sizes, the segmentation fails.

    In addition, as there is no useful information in the intensity of the image,
    or its gradient, we choose to perform the spectral clustering on a graph
    that is only weakly informed by the gradient. This is close to performing
    a Voronoi partition of the graph.

    In addition, we use the mask of the objects to restrict the graph to the
    outline of the objects. In this example, we are interested in
    separating the objects one from the other, and not from the background.

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
    ## Generate the data


    """
    )
    return


@app.cell
def _():
    import numpy as np

    l = 100
    x, y = np.indices((l, l))

    center1 = (28, 24)
    center2 = (40, 50)
    center3 = (67, 58)
    center4 = (24, 70)

    radius1, radius2, radius3, radius4 = 16, 14, 15, 14

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1**2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2**2
    circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3**2
    circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4**2
    return circle1, circle2, circle3, circle4, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting four circles


    """
    )
    return


@app.cell
def _(circle1, circle2, circle3, circle4, np):
    img = circle1 + circle2 + circle3 + circle4
    mask = img.astype(bool)
    # We use a mask that limits to the foreground: the problem that we are
    # interested in here is not separating the objects from the background,
    # but separating them one from the other.
    img = img.astype(float)
    img = img + (1 + 0.2 * np.random.randn(*img.shape))
    return img, mask


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Convert the image into a graph with the value of the gradient on the
    edges.


    """
    )
    return


@app.cell
def _(img, mask):
    from sklearn.feature_extraction import image

    graph = image.img_to_graph(img, mask=mask)
    return graph, image


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Take a decreasing function of the gradient resulting in a segmentation
    that is close to a Voronoi partition


    """
    )
    return


@app.cell
def _(graph, np):
    graph.data = np.exp(-graph.data / graph.data.std())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here we perform spectral clustering using the arpack solver since amg is
    numerically unstable on this example. We then plot the results.


    """
    )
    return


@app.cell
def _(graph, img, mask, np):
    import matplotlib.pyplot as plt
    from sklearn.cluster import spectral_clustering
    _labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
    _label_im = np.full(mask.shape, -1.0)
    _label_im[mask] = _labels
    _fig, _axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    _axs[0].matshow(img)
    _axs[1].matshow(_label_im)
    plt.show()
    return plt, spectral_clustering


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting two circles
    Here we repeat the above process but only consider the first two circles
    we generated. Note that this results in a cleaner separation between the
    circles as the region sizes are easier to balance in this case.


    """
    )
    return


@app.cell
def _(circle1, circle2, image, np, plt, spectral_clustering):
    img_1 = circle1 + circle2
    mask_1 = img_1.astype(bool)
    img_1 = img_1.astype(float)
    img_1 = img_1 + (1 + 0.2 * np.random.randn(*img_1.shape))
    graph_1 = image.img_to_graph(img_1, mask=mask_1)
    graph_1.data = np.exp(-graph_1.data / graph_1.data.std())
    _labels = spectral_clustering(graph_1, n_clusters=2, eigen_solver='arpack')
    _label_im = np.full(mask_1.shape, -1.0)
    _label_im[mask_1] = _labels
    _fig, _axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    _axs[0].matshow(img_1)
    _axs[1].matshow(_label_im)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
