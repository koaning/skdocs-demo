import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Displaying estimators and complex pipelines

    This example illustrates different ways estimators and pipelines can be
    displayed.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return (
        LogisticRegression,
        OneHotEncoder,
        SimpleImputer,
        StandardScaler,
        make_column_transformer,
        make_pipeline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Compact text representation

    Estimators will only show the parameters that have been set to non-default
    values when displayed as a string. This reduces the visual noise and makes it
    easier to spot what the differences are when comparing instances.


    """
    )
    return


@app.cell
def _(LogisticRegression):
    lr = LogisticRegression(penalty="l1")
    print(lr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Rich HTML representation
    In notebooks estimators and pipelines will use a rich HTML representation.
    This is particularly useful to summarise the
    structure of pipelines and other composite estimators, with interactivity to
    provide detail.  Click on the example image below to expand Pipeline
    elements.  See `visualizing_composite_estimators` for how you can use
    this feature.


    """
    )
    return


@app.cell
def _(
    LogisticRegression,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
    make_column_transformer,
    make_pipeline,
):
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


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
