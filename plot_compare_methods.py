import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Comparison of Manifold Learning methods

    An illustration of dimensionality reduction on the S-curve dataset
    with various manifold learning methods.

    For a discussion and comparison of these algorithms, see the
    `manifold module page <manifold>`

    For a similar example, where the methods are applied to a
    sphere dataset, see `sphx_glr_auto_examples_manifold_plot_manifold_sphere.py`

    Note that the purpose of the MDS is to find a low-dimensional
    representation of the data (here 2D) in which the distances respect well
    the distances in the original high-dimensional space, unlike other
    manifold-learning algorithms, it does not seeks an isotropic
    representation of the data in the low-dimensional space.

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
    ## Dataset preparation

    We start by generating the S-curve dataset.


    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    # unused but required import for doing 3d projections with matplotlib < 3.2
    import mpl_toolkits.mplot3d  # noqa: F401
    from matplotlib import ticker

    from sklearn import datasets, manifold

    n_samples = 1500
    S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
    return S_color, S_points, manifold, plt, ticker


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's look at the original data. Also define some helping
    functions, which we will use further on.


    """
    )
    return


@app.cell
def _(S_color, S_points, plt, ticker):
    def plot_3d(points, points_color, title):
        x, y, z = _points.T
        _fig, _ax = plt.subplots(figsize=(6, 6), facecolor='white', tight_layout=True, subplot_kw={'projection': '3d'})
        _fig.suptitle(title, size=16)
        col = _ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
        _ax.view_init(azim=-60, elev=9)
        _ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        _ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        _ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
        _fig.colorbar(col, ax=_ax, orientation='horizontal', shrink=0.6, aspect=60, pad=0.01)
        plt.show()

    def plot_2d(points, points_color, title):
        _fig, _ax = plt.subplots(figsize=(3, 3), facecolor='white', constrained_layout=True)
        _fig.suptitle(title, size=16)
        add_2d_scatter(_ax, _points, points_color)
        plt.show()

    def add_2d_scatter(ax, points, points_color, title=None):
        x, y = _points.T
        _ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
        _ax.set_title(title)
        _ax.xaxis.set_major_formatter(ticker.NullFormatter())
        _ax.yaxis.set_major_formatter(ticker.NullFormatter())
    plot_3d(S_points, S_color, 'Original S-curve samples')
    return add_2d_scatter, plot_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define algorithms for the manifold learning

    Manifold learning is an approach to non-linear dimensionality reduction.
    Algorithms for this task are based on the idea that the dimensionality of
    many data sets is only artificially high.

    Read more in the `User Guide <manifold>`.


    """
    )
    return


@app.cell
def _():
    n_neighbors = 12  # neighborhood which is used to recover the locally linear structure
    n_components = 2  # number of coordinates for the manifold
    return n_components, n_neighbors


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Locally Linear Embeddings

    Locally linear embedding (LLE) can be thought of as a series of local
    Principal Component Analyses which are globally compared to find the
    best non-linear embedding.
    Read more in the `User Guide <locally_linear_embedding>`.


    """
    )
    return


@app.cell
def _(S_points, manifold, n_components, n_neighbors):
    params = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "eigen_solver": "auto",
        "random_state": 0,
    }

    lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
    S_standard = lle_standard.fit_transform(S_points)

    lle_ltsa = manifold.LocallyLinearEmbedding(method="ltsa", **params)
    S_ltsa = lle_ltsa.fit_transform(S_points)

    lle_hessian = manifold.LocallyLinearEmbedding(method="hessian", **params)
    S_hessian = lle_hessian.fit_transform(S_points)

    lle_mod = manifold.LocallyLinearEmbedding(method="modified", **params)
    S_mod = lle_mod.fit_transform(S_points)
    return S_hessian, S_ltsa, S_mod, S_standard


@app.cell
def _(S_color, S_hessian, S_ltsa, S_mod, S_standard, add_2d_scatter, plt):
    _fig, _axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), facecolor='white', constrained_layout=True)
    _fig.suptitle('Locally Linear Embeddings', size=16)
    lle_methods = [('Standard locally linear embedding', S_standard), ('Local tangent space alignment', S_ltsa), ('Hessian eigenmap', S_hessian), ('Modified locally linear embedding', S_mod)]
    for _ax, _method in zip(_axs.flat, lle_methods):
        _name, _points = _method
        add_2d_scatter(_ax, _points, S_color, _name)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Isomap Embedding

    Non-linear dimensionality reduction through Isometric Mapping.
    Isomap seeks a lower-dimensional embedding which maintains geodesic
    distances between all points. Read more in the `User Guide <isomap>`.


    """
    )
    return


@app.cell
def _(S_color, S_points, manifold, n_components, n_neighbors, plot_2d):
    isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
    S_isomap = isomap.fit_transform(S_points)

    plot_2d(S_isomap, S_color, "Isomap Embedding")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Multidimensional scaling

    Multidimensional scaling (MDS) seeks a low-dimensional representation
    of the data in which the distances respect well the distances in the
    original high-dimensional space.
    Read more in the `User Guide <multidimensional_scaling>`.


    """
    )
    return


@app.cell
def _(S_points, manifold, n_components):
    md_scaling = manifold.MDS(
        n_components=n_components,
        max_iter=50,
        n_init=1,
        random_state=0,
        normalized_stress=False,
    )
    S_scaling_metric = md_scaling.fit_transform(S_points)

    md_scaling_nonmetric = manifold.MDS(
        n_components=n_components,
        max_iter=50,
        n_init=1,
        random_state=0,
        normalized_stress=False,
        metric=False,
    )
    S_scaling_nonmetric = md_scaling_nonmetric.fit_transform(S_points)

    md_scaling_classical = manifold.ClassicalMDS(n_components=n_components)
    S_scaling_classical = md_scaling_classical.fit_transform(S_points)
    return S_scaling_classical, S_scaling_metric, S_scaling_nonmetric


@app.cell
def _(
    S_color,
    S_scaling_classical,
    S_scaling_metric,
    S_scaling_nonmetric,
    add_2d_scatter,
    plt,
):
    _fig, _axs = plt.subplots(nrows=1, ncols=3, figsize=(7, 3.5), facecolor='white', constrained_layout=True)
    _fig.suptitle('Multidimensional scaling', size=16)
    mds_methods = [('Metric MDS', S_scaling_metric), ('Non-metric MDS', S_scaling_nonmetric), ('Classical MDS', S_scaling_classical)]
    for _ax, _method in zip(_axs.flat, mds_methods):
        _name, _points = _method
        add_2d_scatter(_ax, _points, S_color, _name)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Spectral embedding for non-linear dimensionality reduction

    This implementation uses Laplacian Eigenmaps, which finds a low dimensional
    representation of the data using a spectral decomposition of the graph Laplacian.
    Read more in the `User Guide <spectral_embedding>`.


    """
    )
    return


@app.cell
def _(S_color, S_points, manifold, n_components, n_neighbors, plot_2d):
    spectral = manifold.SpectralEmbedding(
        n_components=n_components, n_neighbors=n_neighbors, random_state=42
    )
    S_spectral = spectral.fit_transform(S_points)

    plot_2d(S_spectral, S_color, "Spectral Embedding")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### T-distributed Stochastic Neighbor Embedding

    It converts similarities between data points to joint probabilities and
    tries to minimize the Kullback-Leibler divergence between the joint probabilities
    of the low-dimensional embedding and the high-dimensional data. t-SNE has a cost
    function that is not convex, i.e. with different initializations we can get
    different results. Read more in the `User Guide <t_sne>`.


    """
    )
    return


@app.cell
def _(S_color, S_points, manifold, n_components, plot_2d):
    t_sne = manifold.TSNE(
        n_components=n_components,
        perplexity=30,
        init="random",
        max_iter=250,
        random_state=0,
    )
    S_t_sne = t_sne.fit_transform(S_points)

    plot_2d(S_t_sne, S_color, "T-distributed Stochastic  \n Neighbor Embedding")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
