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

    # Column Transformer with Mixed Types

    .. currentmodule:: sklearn

    This example illustrates how to apply different preprocessing and feature
    extraction pipelines to different subsets of features, using
    :class:`~compose.ColumnTransformer`. This is particularly handy for the
    case of datasets that contain heterogeneous data types, since we may want to
    scale the numeric features and one-hot encode the categorical ones.

    In this example, the numeric data is standard-scaled after mean-imputation. The
    categorical data is one-hot encoded via ``OneHotEncoder``, which
    creates a new category for missing values. We further reduce the dimensionality
    by selecting categories using a chi-squared test.

    In addition, we show two different ways to dispatch the columns to the
    particular pre-processor: by column names and by column data types.

    Finally, the preprocessing pipeline is integrated in a full prediction pipeline
    using :class:`~pipeline.Pipeline`, together with a simple classification
    model.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause
    return


@app.cell
def _():
    import numpy as np

    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import fetch_openml
    from sklearn.feature_selection import SelectPercentile, chi2
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    np.random.seed(0)
    return (
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        RandomizedSearchCV,
        SelectPercentile,
        SimpleImputer,
        StandardScaler,
        chi2,
        fetch_openml,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Load data from https://www.openml.org/d/40945


    """
    )
    return


@app.cell
def _(fetch_openml):
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    # Alternatively X and y can be obtained directly from the frame attribute:
    # X = titanic.frame.drop('survived', axis=1)
    # y = titanic.frame['survived']
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Use ``ColumnTransformer`` by selecting column by names

    We will train our classifier with the following features:

    Numeric Features:

    * ``age``: float;
    * ``fare``: float.

    Categorical Features:

    * ``embarked``: categories encoded as strings ``{'C', 'S', 'Q'}``;
    * ``sex``: categories encoded as strings ``{'female', 'male'}``;
    * ``pclass``: ordinal integers ``{1, 2, 3}``.

    We create the preprocessing pipelines for both numeric and categorical data.
    Note that ``pclass`` could either be treated as a categorical or numeric
    feature.


    """
    )
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SelectPercentile,
    SimpleImputer,
    StandardScaler,
    chi2,
):
    numeric_features = ["age", "fare"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["embarked", "sex", "pclass"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return categorical_transformer, numeric_transformer, preprocessor


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Append classifier to preprocessing pipeline.
    Now we have a full prediction pipeline.


    """
    )
    return


@app.cell
def _(LogisticRegression, Pipeline, X, preprocessor, train_test_split, y):
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf.fit(X_train, y_train)
    print("model score: %.3f" % clf.score(X_test, y_test))
    return X_test, X_train, clf, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    HTML representation of ``Pipeline`` (display diagram)

    When the ``Pipeline`` is printed out in a jupyter notebook an HTML
    representation of the estimator is displayed:


    """
    )
    return


@app.cell
def _(clf):
    clf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Use ``ColumnTransformer`` by selecting column by data types

    When dealing with a cleaned dataset, the preprocessing can be automatic by
    using the data types of the column to decide whether to treat a column as a
    numerical or categorical feature.
    :func:`sklearn.compose.make_column_selector` gives this possibility.
    First, let's only select a subset of columns to simplify our
    example.


    """
    )
    return


@app.cell
def _(X_test, X_train):
    subset_feature = ['embarked', 'sex', 'pclass', 'age', 'fare']
    X_train_1, X_test_1 = (X_train[subset_feature], X_test[subset_feature])
    return X_test_1, X_train_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Then, we introspect the information regarding each column data type.


    """
    )
    return


@app.cell
def _(X_train_1):
    X_train_1.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can observe that the `embarked` and `sex` columns were tagged as
    `category` columns when loading the data with ``fetch_openml``. Therefore, we
    can use this information to dispatch the categorical columns to the
    ``categorical_transformer`` and the remaining columns to the
    ``numerical_transformer``.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <div class="alert alert-info"><h4>Note</h4><p>In practice, you will have to handle yourself the column data type.
       If you want some columns to be considered as `category`, you will have to
       convert them into categorical columns. If you are using pandas, you can
       refer to their documentation regarding [Categorical data](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html).</p></div>


    """
    )
    return


@app.cell
def _(
    ColumnTransformer,
    LogisticRegression,
    Pipeline,
    X_test_1,
    X_train_1,
    categorical_transformer,
    numeric_transformer,
    y_test,
    y_train,
):
    from sklearn.compose import make_column_selector as selector
    preprocessor_1 = ColumnTransformer(transformers=[('num', numeric_transformer, selector(dtype_exclude='category')), ('cat', categorical_transformer, selector(dtype_include='category'))])
    clf_1 = Pipeline(steps=[('preprocessor', preprocessor_1), ('classifier', LogisticRegression())])
    clf_1.fit(X_train_1, y_train)
    print('model score: %.3f' % clf_1.score(X_test_1, y_test))
    clf_1
    return clf_1, selector


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The resulting score is not exactly the same as the one from the previous
    pipeline because the dtype-based selector treats the ``pclass`` column as
    a numeric feature instead of a categorical feature as previously:


    """
    )
    return


@app.cell
def _(X_train_1, selector):
    selector(dtype_exclude='category')(X_train_1)
    return


@app.cell
def _(X_train_1, selector):
    selector(dtype_include='category')(X_train_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Using the prediction pipeline in a grid search

    Grid search can also be performed on the different preprocessing steps
    defined in the ``ColumnTransformer`` object, together with the classifier's
    hyperparameters as part of the ``Pipeline``.
    We will search for both the imputer strategy of the numeric preprocessing
    and the regularization parameter of the logistic regression using
    :class:`~sklearn.model_selection.RandomizedSearchCV`. This
    hyperparameter search randomly selects a fixed number of parameter
    settings configured by `n_iter`. Alternatively, one can use
    :class:`~sklearn.model_selection.GridSearchCV` but the cartesian product of
    the parameter space will be evaluated.


    """
    )
    return


@app.cell
def _(RandomizedSearchCV, clf_1):
    param_grid = {'preprocessor__num__imputer__strategy': ['mean', 'median'], 'preprocessor__cat__selector__percentile': [10, 30, 50, 70], 'classifier__C': [0.1, 1.0, 10, 100]}
    search_cv = RandomizedSearchCV(clf_1, param_grid, n_iter=10, random_state=0)
    search_cv
    return (search_cv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Calling 'fit' triggers the cross-validated search for the best
    hyper-parameters combination:



    """
    )
    return


@app.cell
def _(X_train_1, search_cv, y_train):
    search_cv.fit(X_train_1, y_train)
    print('Best params:')
    print(search_cv.best_params_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The internal cross-validation scores obtained by those parameters is:


    """
    )
    return


@app.cell
def _(search_cv):
    print(f"Internal CV score: {search_cv.best_score_:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can also introspect the top grid search results as a pandas dataframe:


    """
    )
    return


@app.cell
def _(search_cv):
    import pandas as pd

    cv_results = pd.DataFrame(search_cv.cv_results_)
    cv_results = cv_results.sort_values("mean_test_score", ascending=False)
    cv_results[
        [
            "mean_test_score",
            "std_test_score",
            "param_preprocessor__num__imputer__strategy",
            "param_preprocessor__cat__selector__percentile",
            "param_classifier__C",
        ]
    ].head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The best hyper-parameters have be used to re-fit a final model on the full
    training set. We can evaluate that final model on held out test data that was
    not used for hyperparameter tuning.



    """
    )
    return


@app.cell
def _(X_test_1, search_cv, y_test):
    print(f'accuracy of the best model from randomized search: {search_cv.score(X_test_1, y_test):.3f}')
    return

if __name__ == "__main__":
    app.run()
