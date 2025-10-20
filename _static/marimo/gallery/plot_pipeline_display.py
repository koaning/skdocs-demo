import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Displaying Pipelines

    The default configuration for displaying a pipeline in a Jupyter Notebook is
    `'diagram'` where `set_config(display='diagram')`. To deactivate HTML representation,
    use `set_config(display='text')`.

    To see more detailed steps in the visualization of the pipeline, click on the
    steps in the pipeline.

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
    ## Displaying a Pipeline with a Preprocessing Step and Classifier
    This section constructs a :class:`~sklearn.pipeline.Pipeline` with a preprocessing
    step, :class:`~sklearn.preprocessing.StandardScaler`, and classifier,
    :class:`~sklearn.linear_model.LogisticRegression`, and displays its visual
    representation.


    """
    )
    return


@app.cell
def _():
    from sklearn import set_config
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    _steps = [('preprocessing', StandardScaler()), ('classifier', LogisticRegression())]
    pipe = Pipeline(_steps)
    return LogisticRegression, Pipeline, StandardScaler, pipe, set_config


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To visualize the diagram, the default is `display='diagram'`.


    """
    )
    return


@app.cell
def _(pipe, set_config):
    set_config(display="diagram")
    pipe  # click on the diagram below to see the details of each step
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To view the text pipeline, change to `display='text'`.


    """
    )
    return


@app.cell
def _(pipe, set_config):
    set_config(display="text")
    pipe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Put back the default display


    """
    )
    return


@app.cell
def _(set_config):
    set_config(display="diagram")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Displaying a Pipeline Chaining Multiple Preprocessing Steps & Classifier
    This section constructs a :class:`~sklearn.pipeline.Pipeline` with multiple
    preprocessing steps, :class:`~sklearn.preprocessing.PolynomialFeatures` and
    :class:`~sklearn.preprocessing.StandardScaler`, and a classifier step,
    :class:`~sklearn.linear_model.LogisticRegression`, and displays its visual
    representation.


    """
    )
    return


@app.cell
def _(LogisticRegression, Pipeline, StandardScaler):
    from sklearn.preprocessing import PolynomialFeatures
    _steps = [('standard_scaler', StandardScaler()), ('polynomial', PolynomialFeatures(degree=3)), ('classifier', LogisticRegression(C=2.0))]
    pipe_1 = Pipeline(_steps)
    pipe_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Displaying a Pipeline and Dimensionality Reduction and Classifier
    This section constructs a :class:`~sklearn.pipeline.Pipeline` with a
    dimensionality reduction step, :class:`~sklearn.decomposition.PCA`,
    a classifier, :class:`~sklearn.svm.SVC`, and displays its visual
    representation.


    """
    )
    return


@app.cell
def _(Pipeline):
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    _steps = [('reduce_dim', PCA(n_components=4)), ('classifier', SVC(kernel='linear'))]
    pipe_2 = Pipeline(_steps)
    pipe_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Displaying a Complex Pipeline Chaining a Column Transformer
    This section constructs a complex :class:`~sklearn.pipeline.Pipeline` with a
    :class:`~sklearn.compose.ColumnTransformer` and a classifier,
    :class:`~sklearn.linear_model.LogisticRegression`, and displays its visual
    representation.


    """
    )
    return


@app.cell
def _(LogisticRegression, Pipeline, StandardScaler):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    _numeric_preprocessor = Pipeline(steps=[('imputation_mean', SimpleImputer(missing_values=np.nan, strategy='mean')), ('scaler', StandardScaler())])
    _categorical_preprocessor = Pipeline(steps=[('imputation_constant', SimpleImputer(fill_value='missing', strategy='constant')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    _preprocessor = ColumnTransformer([('categorical', _categorical_preprocessor, ['state', 'gender']), ('numerical', _numeric_preprocessor, ['age', 'weight'])])
    pipe_3 = make_pipeline(_preprocessor, LogisticRegression(max_iter=500))
    pipe_3
    return ColumnTransformer, OneHotEncoder, SimpleImputer, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Displaying a Grid Search over a Pipeline with a Classifier
    This section constructs a :class:`~sklearn.model_selection.GridSearchCV`
    over a :class:`~sklearn.pipeline.Pipeline` with
    :class:`~sklearn.ensemble.RandomForestClassifier` and displays its visual
    representation.


    """
    )
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    np,
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    _numeric_preprocessor = Pipeline(steps=[('imputation_mean', SimpleImputer(missing_values=np.nan, strategy='mean')), ('scaler', StandardScaler())])
    _categorical_preprocessor = Pipeline(steps=[('imputation_constant', SimpleImputer(fill_value='missing', strategy='constant')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    _preprocessor = ColumnTransformer([('categorical', _categorical_preprocessor, ['state', 'gender']), ('numerical', _numeric_preprocessor, ['age', 'weight'])])
    pipe_4 = Pipeline(steps=[('preprocessor', _preprocessor), ('classifier', RandomForestClassifier())])
    param_grid = {'classifier__n_estimators': [200, 500], 'classifier__max_features': ['auto', 'sqrt', 'log2'], 'classifier__max_depth': [4, 5, 6, 7, 8], 'classifier__criterion': ['gini', 'entropy']}
    grid_search = GridSearchCV(pipe_4, param_grid=param_grid, n_jobs=1)
    grid_search
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
