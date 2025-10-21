import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Polynomial and Spline interpolation

    This example demonstrates how to approximate a function with polynomials up to
    degree ``degree`` by using ridge regression. We show two different ways given
    ``n_samples`` of 1d points ``x_i``:

    - :class:`~sklearn.preprocessing.PolynomialFeatures` generates all monomials
      up to ``degree``. This gives us the so called Vandermonde matrix with
      ``n_samples`` rows and ``degree + 1`` columns::

        [[1, x_0, x_0 ** 2, x_0 ** 3, ..., x_0 ** degree],
         [1, x_1, x_1 ** 2, x_1 ** 3, ..., x_1 ** degree],
         ...]

      Intuitively, this matrix can be interpreted as a matrix of pseudo features
      (the points raised to some power). The matrix is akin to (but different from)
      the matrix induced by a polynomial kernel.

    - :class:`~sklearn.preprocessing.SplineTransformer` generates B-spline basis
      functions. A basis function of a B-spline is a piece-wise polynomial function
      of degree ``degree`` that is non-zero only between ``degree+1`` consecutive
      knots. Given ``n_knots`` number of knots, this results in matrix of
      ``n_samples`` rows and ``n_knots + degree - 1`` columns::

        [[basis_1(x_0), basis_2(x_0), ...],
         [basis_1(x_1), basis_2(x_1), ...],
         ...]

    This example shows that these two transformers are well suited to model
    non-linear effects with a linear model, using a pipeline to add non-linear
    features. Kernel methods extend this idea and can induce very high (even
    infinite) dimensional feature spaces.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
    return PolynomialFeatures, Ridge, SplineTransformer, make_pipeline, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We start by defining a function that we intend to approximate and prepare
    plotting it.


    """
    )
    return


@app.cell
def _(np):
    def f(x):
        """Function to be approximated by polynomial interpolation."""
        return x * np.sin(x)


    # whole range we want to plot
    x_plot = np.linspace(-1, 11, 100)
    return f, x_plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To make it interesting, we only give a small subset of points to train on.


    """
    )
    return


@app.cell
def _(f, np, x_plot):
    x_train = np.linspace(0, 10, 100)
    rng = np.random.RandomState(0)
    x_train = np.sort(rng.choice(x_train, size=20, replace=False))
    y_train = f(x_train)

    # create 2D-array versions of these arrays to feed to transformers
    X_train = x_train[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]
    return X_plot, X_train, x_train, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we are ready to create polynomial features and splines, fit on the
    training points and show how well they interpolate.


    """
    )
    return


@app.cell
def _(
    PolynomialFeatures,
    Ridge,
    SplineTransformer,
    X_plot,
    X_train,
    f,
    make_pipeline,
    plt,
    x_plot,
    x_train,
    y_train,
):
    # plot function
    _lw = 2
    _fig, _ax = plt.subplots()
    _ax.set_prop_cycle(color=['black', 'teal', 'yellowgreen', 'gold', 'darkorange', 'tomato'])
    _ax.plot(x_plot, f(x_plot), linewidth=_lw, label='ground truth')
    _ax.scatter(x_train, y_train, label='training points')
    for degree in [3, 4, 5]:
        _model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.001))
    # plot training points
        _model.fit(X_train, y_train)
        y_plot = _model.predict(X_plot)
    # polynomial features
        _ax.plot(x_plot, y_plot, label=f'degree {degree}')
    _model = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=0.001))
    _model.fit(X_train, y_train)
    y_plot = _model.predict(X_plot)
    _ax.plot(x_plot, y_plot, label='B-spline')
    _ax.legend(loc='lower center')
    # B-spline with 4 + 3 - 1 = 6 basis functions
    _ax.set_ylim(-20, 10)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This shows nicely that higher degree polynomials can fit the data better. But
    at the same time, too high powers can show unwanted oscillatory behaviour
    and are particularly dangerous for extrapolation beyond the range of fitted
    data. This is an advantage of B-splines. They usually fit the data as well as
    polynomials and show very nice and smooth behaviour. They have also good
    options to control the extrapolation, which defaults to continue with a
    constant. Note that most often, you would rather increase the number of knots
    but keep ``degree=3``.

    In order to give more insights into the generated feature bases, we plot all
    columns of both transformers separately.


    """
    )
    return


@app.cell
def _(PolynomialFeatures, SplineTransformer, X_plot, X_train, plt, x_plot):
    _fig, axes = plt.subplots(ncols=2, figsize=(16, 5))
    pft = PolynomialFeatures(degree=3).fit(X_train)
    axes[0].plot(x_plot, pft.transform(X_plot))
    axes[0].legend(axes[0].lines, [f'degree {n}' for n in range(4)])
    axes[0].set_title('PolynomialFeatures')
    _splt = SplineTransformer(n_knots=4, degree=3).fit(X_train)
    axes[1].plot(x_plot, _splt.transform(X_plot))
    axes[1].legend(axes[1].lines, [f'spline {n}' for n in range(6)])
    axes[1].set_title('SplineTransformer')
    _knots = _splt.bsplines_[0].t
    axes[1].vlines(_knots[3:-3], ymin=0, ymax=0.8, linestyles='dashed')
    # plot knots of spline
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the left plot, we recognize the lines corresponding to simple monomials
    from ``x**0`` to ``x**3``. In the right figure, we see the six B-spline
    basis functions of ``degree=3`` and also the four knot positions that were
    chosen during ``fit``. Note that there are ``degree`` number of additional
    knots each to the left and to the right of the fitted interval. These are
    there for technical reasons, so we refrain from showing them. Every basis
    function has local support and is continued as a constant beyond the fitted
    range. This extrapolating behaviour could be changed by the argument
    ``extrapolation``.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Periodic Splines
    In the previous example we saw the limitations of polynomials and splines for
    extrapolation beyond the range of the training observations. In some
    settings, e.g. with seasonal effects, we expect a periodic continuation of
    the underlying signal. Such effects can be modelled using periodic splines,
    which have equal function value and equal derivatives at the first and last
    knot. In the following case we show how periodic splines provide a better fit
    both within and outside of the range of training data given the additional
    information of periodicity. The splines period is the distance between
    the first and last knot, which we specify manually.

    Periodic splines can also be useful for naturally periodic features (such as
    day of the year), as the smoothness at the boundary knots prevents a jump in
    the transformed values (e.g. from Dec 31st to Jan 1st). For such naturally
    periodic features or more generally features where the period is known, it is
    advised to explicitly pass this information to the `SplineTransformer` by
    setting the knots manually.


    """
    )
    return


@app.cell
def _(Ridge, SplineTransformer, X_train, make_pipeline, np, plt, x_train):
    def g(x):
        """Function to be approximated by periodic spline interpolation."""
        return np.sin(x) - 0.7 * np.cos(x * 3)
    y_train_1 = g(x_train)
    x_plot_ext = np.linspace(-1, 21, 200)
    X_plot_ext = x_plot_ext[:, np.newaxis]
    _lw = 2
    _fig, _ax = plt.subplots()
    _ax.set_prop_cycle(color=['black', 'tomato', 'teal'])
    _ax.plot(x_plot_ext, g(x_plot_ext), linewidth=_lw, label='ground truth')
    _ax.scatter(x_train, y_train_1, label='training points')
    for transformer, label in [(SplineTransformer(degree=3, n_knots=10), 'spline'), (SplineTransformer(degree=3, knots=np.linspace(0, 2 * np.pi, 10)[:, None], extrapolation='periodic'), 'periodic spline')]:
        _model = make_pipeline(transformer, Ridge(alpha=0.001))
        _model.fit(X_train, y_train_1)
        y_plot_ext = _model.predict(X_plot_ext)
        _ax.plot(x_plot_ext, y_plot_ext, label=label)
    _ax.legend()
    _fig.show()
    return X_plot_ext, x_plot_ext


@app.cell
def _(SplineTransformer, X_plot_ext, X_train, np, plt, x_plot_ext):
    _fig, _ax = plt.subplots()
    _knots = np.linspace(0, 2 * np.pi, 4)
    _splt = SplineTransformer(knots=_knots[:, None], degree=3, extrapolation='periodic').fit(X_train)
    _ax.plot(x_plot_ext, _splt.transform(X_plot_ext))
    _ax.legend(_ax.lines, [f'spline {n}' for n in range(3)])
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
