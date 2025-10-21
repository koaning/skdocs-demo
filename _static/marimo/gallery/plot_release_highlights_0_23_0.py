import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Release Highlights for scikit-learn 0.23

    .. currentmodule:: sklearn

    We are pleased to announce the release of scikit-learn 0.23! Many bug fixes
    and improvements were added, as well as some new key features. We detail
    below a few of the major features of this release. **For an exhaustive list of
    all the changes**, please refer to the `release notes <release_notes_0_23>`.

    To install the latest version (with pip)::

        pip install --upgrade scikit-learn

    or with conda::

        conda install -c conda-forge scikit-learn

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Generalized Linear Models, and Poisson loss for gradient boosting
    Long-awaited Generalized Linear Models with non-normal loss functions are now
    available. In particular, three new regressors were implemented:
    :class:`~sklearn.linear_model.PoissonRegressor`,
    :class:`~sklearn.linear_model.GammaRegressor`, and
    :class:`~sklearn.linear_model.TweedieRegressor`. The Poisson regressor can be
    used to model positive integer counts, or relative frequencies. Read more in
    the `User Guide <Generalized_linear_regression>`. Additionally,
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor` supports a new
    'poisson' loss as well.


    """
    )
    return


@app.cell
def _():
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import train_test_split
    _n_samples, _n_features = (1000, 20)
    _rng = np.random.RandomState(0)
    _X = _rng.randn(_n_samples, _n_features)
    _y = _rng.poisson(lam=np.exp(_X[:, 5]) / 2)
    _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, random_state=_rng)
    # positive integer target correlated with X[:, 5] with many zeros:
    glm = PoissonRegressor()
    gbdt = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.01)
    glm.fit(_X_train, _y_train)
    gbdt.fit(_X_train, _y_train)
    print(glm.score(_X_test, _y_test))
    print(gbdt.score(_X_test, _y_test))
    return HistGradientBoostingRegressor, np, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Rich visual representation of estimators
    Estimators can now be visualized in notebooks by enabling the
    `display='diagram'` option. This is particularly useful to summarise the
    structure of pipelines and other composite estimators, with interactivity to
    provide detail.  Click on the example image below to expand Pipeline
    elements.  See `visualizing_composite_estimators` for how you can use
    this feature.


    """
    )
    return


@app.cell
def _():
    from sklearn import set_config
    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    set_config(display="diagram")

    num_proc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_proc = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = make_column_transformer(
        (num_proc, ("feat1", "feat3")), (cat_proc, ("feat0", "feat2"))
    )

    clf = make_pipeline(preprocessor, LogisticRegression())
    clf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Scalability and stability improvements to KMeans
    The :class:`~sklearn.cluster.KMeans` estimator was entirely re-worked, and it
    is now significantly faster and more stable. In addition, the Elkan algorithm
    is now compatible with sparse matrices. The estimator uses OpenMP based
    parallelism instead of relying on joblib, so the `n_jobs` parameter has no
    effect anymore. For more details on how to control the number of threads,
    please refer to our `parallelism` notes.


    """
    )
    return


@app.cell
def _(np, train_test_split):
    import scipy
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import completeness_score
    _rng = np.random.RandomState(0)
    _X, _y = make_blobs(random_state=_rng)
    _X = scipy.sparse.csr_matrix(_X)
    _X_train, _X_test, _, _y_test = train_test_split(_X, _y, random_state=_rng)
    kmeans = KMeans(n_init='auto').fit(_X_train)
    print(completeness_score(kmeans.predict(_X_test), _y_test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Improvements to the histogram-based Gradient Boosting estimators
    Various improvements were made to
    :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`. On top of the
    Poisson loss mentioned above, these estimators now support `sample
    weights <sw_hgbdt>`. Also, an automatic early-stopping criterion was added:
    early-stopping is enabled by default when the number of samples exceeds 10k.
    Finally, users can now define `monotonic constraints
    <monotonic_cst_gbdt>` to constrain the predictions based on the variations of
    specific features. In the following example, we construct a target that is
    generally positively correlated with the first feature, with some noise.
    Applying monotoinc constraints allows the prediction to capture the global
    effect of the first feature, instead of fitting the noise. For a usecase
    example, see `sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py`.


    """
    )
    return


@app.cell
def _(HistGradientBoostingRegressor, np):
    from matplotlib import pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    _n_samples = 500
    _rng = np.random.RandomState(0)
    _X = _rng.randn(_n_samples, 2)
    noise = _rng.normal(loc=0.0, scale=0.01, size=_n_samples)
    _y = 5 * _X[:, 0] + np.sin(10 * np.pi * _X[:, 0]) - noise
    gbdt_no_cst = HistGradientBoostingRegressor().fit(_X, _y)
    gbdt_cst = HistGradientBoostingRegressor(monotonic_cst=[1, 0]).fit(_X, _y)
    disp = PartialDependenceDisplay.from_estimator(gbdt_no_cst, _X, features=[0], feature_names=['feature 0'], line_kw={'linewidth': 4, 'label': 'unconstrained', 'color': 'tab:blue'})
    PartialDependenceDisplay.from_estimator(gbdt_cst, _X, features=[0], line_kw={'linewidth': 4, 'label': 'constrained', 'color': 'tab:orange'}, ax=disp.axes_)
    disp.axes_[0, 0].plot(_X[:, 0], _y, 'o', alpha=0.5, zorder=-1, label='samples', color='tab:green')
    disp.axes_[0, 0].set_ylim(-3, 3)
    disp.axes_[0, 0].set_xlim(-1, 1)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Sample-weight support for Lasso and ElasticNet
    The two linear regressors :class:`~sklearn.linear_model.Lasso` and
    :class:`~sklearn.linear_model.ElasticNet` now support sample weights.


    """
    )
    return


@app.cell
def _(np, train_test_split):
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Lasso
    _n_samples, _n_features = (1000, 20)
    _rng = np.random.RandomState(0)
    _X, _y = make_regression(_n_samples, _n_features, random_state=_rng)
    sample_weight = _rng.rand(_n_samples)
    _X_train, _X_test, _y_train, _y_test, sw_train, sw_test = train_test_split(_X, _y, sample_weight, random_state=_rng)
    reg = Lasso()
    reg.fit(_X_train, _y_train, sample_weight=sw_train)
    print(reg.score(_X_test, _y_test, sw_test))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
