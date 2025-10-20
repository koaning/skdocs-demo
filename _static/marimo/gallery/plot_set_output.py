import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Introducing the `set_output` API

    .. currentmodule:: sklearn

    This example will demonstrate the `set_output` API to configure transformers to
    output pandas DataFrames. `set_output` can be configured per estimator by calling
    the `set_output` method or globally by setting `set_config(transform_output="pandas")`.
    For details, see
    [SLEP018](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep018/proposal.html)_.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    First, we load the iris dataset as a DataFrame to demonstrate the `set_output` API.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    _X, _y = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, stratify=_y, random_state=0)
    X_train.head()
    return X_test, X_train, train_test_split, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To configure an estimator such as :class:`preprocessing.StandardScaler` to return
    DataFrames, call `set_output`. This feature requires pandas to be installed.


    """
    )
    return


@app.cell
def _(X_test, X_train):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().set_output(transform='pandas')
    scaler.fit(X_train)
    _X_test_scaled = scaler.transform(X_test)
    _X_test_scaled.head()
    return (StandardScaler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `set_output` can be called after `fit` to configure `transform` after the fact.


    """
    )
    return


@app.cell
def _(StandardScaler, X_test, X_train):
    scaler2 = StandardScaler()

    scaler2.fit(X_train)
    X_test_np = scaler2.transform(X_test)
    print(f"Default output type: {type(X_test_np).__name__}")

    scaler2.set_output(transform="pandas")
    X_test_df = scaler2.transform(X_test)
    print(f"Configured pandas output type: {type(X_test_df).__name__}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In a :class:`pipeline.Pipeline`, `set_output` configures all steps to output
    DataFrames.


    """
    )
    return


@app.cell
def _(StandardScaler, X_train, y_train):
    from sklearn.feature_selection import SelectPercentile
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(
        StandardScaler(), SelectPercentile(percentile=75), LogisticRegression()
    )
    clf.set_output(transform="pandas")
    clf.fit(X_train, y_train)
    return LogisticRegression, SelectPercentile, clf, make_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Each transformer in the pipeline is configured to return DataFrames. This
    means that the final logistic regression step contains the feature names of the input.


    """
    )
    return


@app.cell
def _(clf):
    clf[-1].feature_names_in_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <div class="alert alert-info"><h4>Note</h4><p>If one uses the method `set_params`, the transformer will be
       replaced by a new one with the default output format.</p></div>


    """
    )
    return


@app.cell
def _(StandardScaler, X_train, clf, y_train):
    clf.set_params(standardscaler=StandardScaler())
    clf.fit(X_train, y_train)
    clf[-1].feature_names_in_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To keep the intended behavior, use `set_output` on the new transformer
    beforehand


    """
    )
    return


@app.cell
def _(StandardScaler, X_train, clf, y_train):
    scaler_1 = StandardScaler().set_output(transform='pandas')
    clf.set_params(standardscaler=scaler_1)
    clf.fit(X_train, y_train)
    clf[-1].feature_names_in_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Next we load the titanic dataset to demonstrate `set_output` with
    :class:`compose.ColumnTransformer` and heterogeneous data.


    """
    )
    return


@app.cell
def _(train_test_split):
    from sklearn.datasets import fetch_openml
    _X, _y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(_X, _y, stratify=_y)
    return X_test_1, X_train_1, y_test_1, y_train_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The `set_output` API can be configured globally by using :func:`set_config` and
    setting `transform_output` to `"pandas"`.


    """
    )
    return


@app.cell
def _(
    LogisticRegression,
    SelectPercentile,
    StandardScaler,
    X_test_1,
    X_train_1,
    make_pipeline,
    y_test_1,
    y_train_1,
):
    from sklearn import set_config
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    set_config(transform_output='pandas')
    num_pipe = make_pipeline(SimpleImputer(), StandardScaler())
    num_cols = ['age', 'fare']
    ct = ColumnTransformer((('numerical', num_pipe, num_cols), ('categorical', OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore'), ['embarked', 'sex', 'pclass'])), verbose_feature_names_out=False)
    clf_1 = make_pipeline(ct, SelectPercentile(percentile=50), LogisticRegression())
    clf_1.fit(X_train_1, y_train_1)
    clf_1.score(X_test_1, y_test_1)
    return clf_1, num_cols, set_config


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    With the global configuration, all transformers output DataFrames. This allows us to
    easily plot the logistic regression coefficients with the corresponding feature names.


    """
    )
    return


@app.cell
def _(clf_1):
    import pandas as pd
    log_reg = clf_1[-1]
    coef = pd.Series(log_reg.coef_.ravel(), index=log_reg.feature_names_in_)
    _ = coef.sort_values().plot.barh()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In order to demonstrate the :func:`config_context` functionality below, let
    us first reset `transform_output` to its default value.


    """
    )
    return


@app.cell
def _(set_config):
    set_config(transform_output="default")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    When configuring the output type with :func:`config_context` the
    configuration at the time when `transform` or `fit_transform` are
    called is what counts. Setting these only when you construct or fit
    the transformer has no effect.


    """
    )
    return


@app.cell
def _(StandardScaler, X_train_1, num_cols):
    from sklearn import config_context
    scaler_2 = StandardScaler()
    scaler_2.fit(X_train_1[num_cols])
    return config_context, scaler_2


@app.cell
def _(X_test_1, config_context, num_cols, scaler_2):
    with config_context(transform_output='pandas'):
        _X_test_scaled = scaler_2.transform(X_test_1[num_cols])
    _X_test_scaled.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    outside of the context manager, the output will be a NumPy array


    """
    )
    return


@app.cell
def _(X_test_1, num_cols, scaler_2):
    _X_test_scaled = scaler_2.transform(X_test_1[num_cols])
    _X_test_scaled[:5]
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
