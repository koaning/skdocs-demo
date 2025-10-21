import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Release Highlights for scikit-learn 1.2

    .. currentmodule:: sklearn

    We are pleased to announce the release of scikit-learn 1.2! Many bug fixes
    and improvements were added, as well as some new key features. We detail
    below a few of the major features of this release. **For an exhaustive list of
    all the changes**, please refer to the `release notes <release_notes_1_2>`.

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
    ## Pandas output with `set_output` API
    scikit-learn's transformers now support pandas output with the `set_output` API.
    To learn more about the `set_output` API see the example:
    `sphx_glr_auto_examples_miscellaneous_plot_set_output.py` and
    # this [video, pandas DataFrame output for scikit-learn transformers
    (some examples)](https://youtu.be/5bCg8VfX2x8)_.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

    X, y = load_iris(as_frame=True, return_X_y=True)
    sepal_cols = ["sepal length (cm)", "sepal width (cm)"]
    petal_cols = ["petal length (cm)", "petal width (cm)"]

    preprocessor = ColumnTransformer(
        [
            ("scaler", StandardScaler(), sepal_cols),
            (
                "kbin",
                KBinsDiscretizer(encode="ordinal", quantile_method="averaged_inverted_cdf"),
                petal_cols,
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    X_out = preprocessor.fit_transform(X)
    X_out.sample(n=5, random_state=0)
    return ColumnTransformer, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Interaction constraints in Histogram-based Gradient Boosting Trees
    :class:`~ensemble.HistGradientBoostingRegressor` and
    :class:`~ensemble.HistGradientBoostingClassifier` now supports interaction constraints
    with the `interaction_cst` parameter. For details, see the
    `User Guide <interaction_cst_hgbt>`. In the following example, features are not
    allowed to interact.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import HistGradientBoostingRegressor
    X_1, y_1 = load_diabetes(return_X_y=True, as_frame=True)
    hist_no_interact = HistGradientBoostingRegressor(interaction_cst=[[i] for i in range(X_1.shape[1])], random_state=0)
    hist_no_interact.fit(X_1, y_1)
    return HistGradientBoostingRegressor, X_1, hist_no_interact, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## New and enhanced displays
    :class:`~metrics.PredictionErrorDisplay` provides a way to analyze regression
    models in a qualitative manner.


    """
    )
    return


@app.cell
def _(X_1, hist_no_interact, y_1):
    import matplotlib.pyplot as plt
    from sklearn.metrics import PredictionErrorDisplay
    _fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    _ = PredictionErrorDisplay.from_estimator(hist_no_interact, X_1, y_1, kind='actual_vs_predicted', ax=axs[0])
    _ = PredictionErrorDisplay.from_estimator(hist_no_interact, X_1, y_1, kind='residual_vs_predicted', ax=axs[1])
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    :class:`~model_selection.LearningCurveDisplay` is now available to plot
    results from :func:`~model_selection.learning_curve`.


    """
    )
    return


@app.cell
def _(X_1, hist_no_interact, np, y_1):
    from sklearn.model_selection import LearningCurveDisplay
    _ = LearningCurveDisplay.from_estimator(hist_no_interact, X_1, y_1, cv=5, n_jobs=2, train_sizes=np.linspace(0.1, 1, 5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    :class:`~inspection.PartialDependenceDisplay` exposes a new parameter
    `categorical_features` to display partial dependence for categorical features
    using bar plots and heatmaps.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml
    X_2, y_2 = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True, parser='pandas')
    X_2 = X_2.select_dtypes(['number', 'category']).drop(columns=['body'])
    return X_2, fetch_openml, y_2


@app.cell
def _(ColumnTransformer, HistGradientBoostingRegressor, X_2, y_2):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder
    categorical_features = ['pclass', 'sex', 'embarked']
    model = make_pipeline(ColumnTransformer(transformers=[('cat', OrdinalEncoder(), categorical_features)], remainder='passthrough'), HistGradientBoostingRegressor(random_state=0)).fit(X_2, y_2)
    return categorical_features, model


@app.cell
def _(X_2, categorical_features, model, plt):
    from sklearn.inspection import PartialDependenceDisplay
    _fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    _ = PartialDependenceDisplay.from_estimator(model, X_2, features=['age', 'sex', ('pclass', 'sex')], categorical_features=categorical_features, ax=ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Faster parser in :func:`~datasets.fetch_openml`
    :func:`~datasets.fetch_openml` now supports a new `"pandas"` parser that is
    more memory and CPU efficient. In v1.4, the default will change to
    `parser="auto"` which will automatically use the `"pandas"` parser for dense
    data and `"liac-arff"` for sparse data.


    """
    )
    return


@app.cell
def _(fetch_openml):
    X_3, y_3 = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True, parser='pandas')
    X_3.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Experimental Array API support in :class:`~discriminant_analysis.LinearDiscriminantAnalysis`
    Experimental support for the [Array API](https://data-apis.org/array-api/latest/)
    specification was added to :class:`~discriminant_analysis.LinearDiscriminantAnalysis`.
    The estimator can now run on any Array API compliant libraries such as
    [CuPy](https://docs.cupy.dev/en/stable/overview.html)_, a GPU-accelerated array
    library. For details, see the `User Guide <array_api>`.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Improved efficiency of many estimators
    In version 1.1 the efficiency of many estimators relying on the computation of
    pairwise distances (essentially estimators related to clustering, manifold
    learning and neighbors search algorithms) was greatly improved for float64
    dense input. Efficiency improvement especially were a reduced memory footprint
    and a much better scalability on multi-core machines.
    In version 1.2, the efficiency of these estimators was further improved for all
    combinations of dense and sparse inputs on float32 and float64 datasets, except
    the sparse-dense and dense-sparse combinations for the Euclidean and Squared
    Euclidean Distance metrics.
    A detailed list of the impacted estimators can be found in the
    `changelog <release_notes_1_2>`.


    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
