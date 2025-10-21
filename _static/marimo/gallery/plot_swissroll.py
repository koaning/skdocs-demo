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

    # Swiss Roll And Swiss-Hole Reduction
    This notebook seeks to compare two popular non-linear dimensionality
    techniques, T-distributed Stochastic Neighbor Embedding (t-SNE) and
    Locally Linear Embedding (LLE), on the classic Swiss Roll dataset.
    Then, we will explore how they both deal with the addition of a hole
    in the data.

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
    ## Swiss Roll

    We start by generating the Swiss Roll dataset.


    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    from sklearn import datasets, manifold

    sr_points, sr_color = datasets.make_swiss_roll(n_samples=1500, random_state=0)
    return datasets, manifold, plt, sr_color, sr_points


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, let's take a look at our data:


    """
    )
    return


@app.cell
def _(plt, sr_color, sr_points):
    _fig = plt.figure(figsize=(8, 6))
    _ax = _fig.add_subplot(111, projection='3d')
    _fig.add_axes(_ax)
    _ax.scatter(sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=50, alpha=0.8)
    _ax.set_title('Swiss Roll in Ambient Space')
    _ax.view_init(azim=-66, elev=12)
    _ = _ax.text2D(0.8, 0.05, s='n_samples=1500', transform=_ax.transAxes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Computing the LLE and t-SNE embeddings, we find that LLE seems to unroll the
    Swiss Roll pretty effectively. t-SNE on the other hand, is able
    to preserve the general structure of the data, but, poorly represents the
    continuous nature of our original data. Instead, it seems to unnecessarily
    clump sections of points together.


    """
    )
    return


@app.cell
def _(manifold, plt, sr_color, sr_points):
    sr_lle, sr_err = manifold.locally_linear_embedding(sr_points, n_neighbors=12, n_components=2)
    sr_tsne = manifold.TSNE(n_components=2, perplexity=40, random_state=0).fit_transform(sr_points)
    _fig, _axs = plt.subplots(figsize=(8, 8), nrows=2)
    _axs[0].scatter(sr_lle[:, 0], sr_lle[:, 1], c=sr_color)
    _axs[0].set_title('LLE Embedding of Swiss Roll')
    _axs[1].scatter(sr_tsne[:, 0], sr_tsne[:, 1], c=sr_color)
    _ = _axs[1].set_title('t-SNE Embedding of Swiss Roll')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <div class="alert alert-info"><h4>Note</h4><p>LLE seems to be stretching the points from the center (purple)
        of the swiss roll. However, we observe that this is simply a byproduct
        of how the data was generated. There is a higher density of points near the
        center of the roll, which ultimately affects how LLE reconstructs the
        data in a lower dimension.</p></div>


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Swiss-Hole

    Now let's take a look at how both algorithms deal with us adding a hole to
    the data. First, we generate the Swiss-Hole dataset and plot it:


    """
    )
    return


@app.cell
def _(datasets, plt):
    sh_points, sh_color = datasets.make_swiss_roll(n_samples=1500, hole=True, random_state=0)
    _fig = plt.figure(figsize=(8, 6))
    _ax = _fig.add_subplot(111, projection='3d')
    _fig.add_axes(_ax)
    _ax.scatter(sh_points[:, 0], sh_points[:, 1], sh_points[:, 2], c=sh_color, s=50, alpha=0.8)
    _ax.set_title('Swiss-Hole in Ambient Space')
    _ax.view_init(azim=-66, elev=12)
    _ = _ax.text2D(0.8, 0.05, s='n_samples=1500', transform=_ax.transAxes)
    return sh_color, sh_points


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Computing the LLE and t-SNE embeddings, we obtain similar results to the
    Swiss Roll. LLE very capably unrolls the data and even preserves
    the hole. t-SNE, again seems to clump sections of points together, but, we
    note that it preserves the general topology of the original data.


    """
    )
    return


@app.cell
def _(manifold, plt, sh_color, sh_points):
    sh_lle, sh_err = manifold.locally_linear_embedding(sh_points, n_neighbors=12, n_components=2)
    sh_tsne = manifold.TSNE(n_components=2, perplexity=40, init='random', random_state=0).fit_transform(sh_points)
    _fig, _axs = plt.subplots(figsize=(8, 8), nrows=2)
    _axs[0].scatter(sh_lle[:, 0], sh_lle[:, 1], c=sh_color)
    _axs[0].set_title('LLE Embedding of Swiss-Hole')
    _axs[1].scatter(sh_tsne[:, 0], sh_tsne[:, 1], c=sh_color)
    _ = _axs[1].set_title('t-SNE Embedding of Swiss-Hole')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Concluding remarks

    We note that t-SNE benefits from testing more combinations of parameters.
    Better results could probably have been obtained by better tuning these
    parameters.

    We observe that, as seen in the "Manifold learning on
    handwritten digits" example, t-SNE generally performs better than LLE
    on real world data.


    """
    )
    return

if __name__ == "__main__":
    app.run()
