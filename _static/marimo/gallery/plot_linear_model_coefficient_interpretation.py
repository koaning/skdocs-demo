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

    # Common pitfalls in the interpretation of coefficients of linear models

    In linear models, the target value is modeled as a linear combination of the
    features (see the `linear_model` User Guide section for a description of a
    set of linear models available in scikit-learn). Coefficients in multiple linear
    models represent the relationship between the given feature, $X_i$ and the
    target, $y$, assuming that all the other features remain constant
    ([conditional dependence](https://en.wikipedia.org/wiki/Conditional_dependence)). This is different
    from plotting $X_i$ versus $y$ and fitting a linear relationship: in
    that case all possible values of the other features are taken into account in
    the estimation (marginal dependence).

    This example will provide some hints in interpreting coefficient in linear
    models, pointing at problems that arise when either the linear model is not
    appropriate to describe the dataset, or when features are correlated.

    <div class="alert alert-info"><h4>Note</h4><p>Keep in mind that the features $X$ and the outcome $y$ are in
        general the result of a data generating process that is unknown to us.
        Machine learning models are trained to approximate the unobserved
        mathematical function that links $X$ to $y$ from sample data. As
        a result, any interpretation made about a model may not necessarily
        generalize to the true data generating process. This is especially true when
        the model is of bad quality or when the sample data is not representative of
        the population.</p></div>

    We will use data from the ["Current Population Survey"](https://www.openml.org/d/534) from 1985 to predict wage as a function of
    various features such as experience, age, or education.

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
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy as sp
    import seaborn as sns
    return np, pd, plt, sns, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The dataset: wages

    We fetch the data from [OpenML](http://openml.org/).
    Note that setting the parameter `as_frame` to True will retrieve the data
    as a pandas dataframe.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    survey = fetch_openml(data_id=534, as_frame=True)
    return (survey,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Then, we identify features `X` and target `y`: the column WAGE is our
    target variable (i.e. the variable which we want to predict).


    """
    )
    return


@app.cell
def _(survey):
    X = survey.data[survey.feature_names]
    X.describe(include="all")
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that the dataset contains categorical and numerical variables.
    We will need to take this into account when preprocessing the dataset
    thereafter.


    """
    )
    return


@app.cell
def _(X):
    X.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Our target for prediction: the wage.
    Wages are described as floating-point number in dollars per hour.


    """
    )
    return


@app.cell
def _(survey):
    y = survey.target.values.ravel()
    survey.target.head()
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We split the sample into a train and a test dataset.
    Only the train dataset will be used in the following exploratory analysis.
    This is a way to emulate a real situation where predictions are performed on
    an unknown target, and we don't want our analysis and decisions to be biased
    by our knowledge of the test data.


    """
    )
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    First, let's get some insights by looking at the variables' distributions and
    at the pairwise relationships between them. Only numerical
    variables will be used. In the following plot, each dot represents a sample.



    """
    )
    return


@app.cell
def _(X_train, sns, y_train):
    train_dataset = X_train.copy()
    train_dataset.insert(0, "WAGE", y_train)
    _ = sns.pairplot(train_dataset, kind="reg", diag_kind="kde")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Looking closely at the WAGE distribution reveals that it has a
    long tail. For this reason, we should take its logarithm
    to turn it approximately into a normal distribution (linear models such
    as ridge or lasso work best for a normal distribution of error).

    The WAGE is increasing when EDUCATION is increasing.
    Note that the dependence between WAGE and EDUCATION
    represented here is a marginal dependence, i.e. it describes the behavior
    of a specific variable without keeping the others fixed.

    Also, the EXPERIENCE and AGE are strongly linearly correlated.


    ## The machine-learning pipeline

    To design our machine-learning pipeline, we first manually
    check the type of data that we are dealing with:


    """
    )
    return


@app.cell
def _(survey):
    survey.data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As seen previously, the dataset contains columns with different data types
    and we need to apply a specific preprocessing for each data types.
    In particular categorical variables cannot be included in linear model if not
    coded as integers first. In addition, to avoid categorical features to be
    treated as ordered values, we need to one-hot-encode them.
    Our pre-processor will:

    - one-hot encode (i.e., generate a column by category) the categorical
      columns, only for non-binary categorical variables;
    - as a first approach (we will see after how the normalisation of numerical
      values will affect our discussion), keep numerical values as they are.


    """
    )
    return


@app.cell
def _():
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder

    categorical_columns = ["RACE", "OCCUPATION", "SECTOR", "MARR", "UNION", "SEX", "SOUTH"]
    numerical_columns = ["EDUCATION", "EXPERIENCE", "AGE"]

    preprocessor = make_column_transformer(
        (OneHotEncoder(drop="if_binary"), categorical_columns),
        remainder="passthrough",
        verbose_feature_names_out=False,  # avoid to prepend the preprocessor names
    )
    return (
        OneHotEncoder,
        categorical_columns,
        make_column_transformer,
        numerical_columns,
        preprocessor,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We use a ridge regressor
    with a very small regularization to model the logarithm of the WAGE.


    """
    )
    return


@app.cell
def _(np, preprocessor, sp):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        preprocessor,
        TransformedTargetRegressor(
            regressor=Ridge(alpha=1e-10), func=np.log10, inverse_func=sp.special.exp10
        ),
    )
    return Ridge, TransformedTargetRegressor, make_pipeline, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Processing the dataset

    First, we fit the model.


    """
    )
    return


@app.cell
def _(X_train, model, y_train):
    model.fit(X_train, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Then we check the performance of the computed model by plotting its predictions
    against the actual values on the test set, and by computing
    the median absolute error.


    """
    )
    return


@app.cell
def _(X_test, X_train, model, y_test, y_train):
    from sklearn.metrics import PredictionErrorDisplay, median_absolute_error
    _mae_train = median_absolute_error(y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    _mae_test = median_absolute_error(y_test, y_pred)
    scores = {'MedAE on training set': f'{_mae_train:.2f} $/hour', 'MedAE on testing set': f'{_mae_test:.2f} $/hour'}
    return PredictionErrorDisplay, median_absolute_error, scores, y_pred


@app.cell
def _(PredictionErrorDisplay, plt, scores, y_pred, y_test):
    _, _ax = plt.subplots(figsize=(5, 5))
    _display = PredictionErrorDisplay.from_predictions(y_test, y_pred, kind='actual_vs_predicted', ax=_ax, scatter_kwargs={'alpha': 0.5})
    _ax.set_title('Ridge model, small regularization')
    for _name, _score in scores.items():
        _ax.plot([], [], ' ', label=f'{_name}: {_score}')
    _ax.legend(loc='upper left')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The model learnt is far from being a good model making accurate predictions:
    this is obvious when looking at the plot above, where good predictions
    should lie on the black dashed line.

    In the following section, we will interpret the coefficients of the model.
    While we do so, we should keep in mind that any conclusion we draw is
    about the model that we build, rather than about the true (real-world)
    generative process of the data.

    ## Interpreting coefficients: scale matters

    First of all, we can take a look to the values of the coefficients of the
    regressor we have fitted.


    """
    )
    return


@app.cell
def _(model, pd):
    feature_names = model[:-1].get_feature_names_out()

    coefs = pd.DataFrame(
        model[-1].regressor_.coef_,
        columns=["Coefficients"],
        index=feature_names,
    )

    coefs
    return coefs, feature_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The AGE coefficient is expressed in "dollars/hour per living years" while the
    EDUCATION one is expressed in "dollars/hour per years of education". This
    representation of the coefficients has the benefit of making clear the
    practical predictions of the model: an increase of $1$ year in AGE
    means a decrease of $0.030867$ dollars/hour, while an increase of
    $1$ year in EDUCATION means an increase of $0.054699$
    dollars/hour. On the other hand, categorical variables (as UNION or SEX) are
    adimensional numbers taking either the value 0 or 1. Their coefficients
    are expressed in dollars/hour. Then, we cannot compare the magnitude of
    different coefficients since the features have different natural scales, and
    hence value ranges, because of their different unit of measure. This is more
    visible if we plot the coefficients.


    """
    )
    return


@app.cell
def _(coefs, plt):
    coefs.plot.barh(figsize=(9, 7))
    plt.title("Ridge model, small regularization")
    plt.axvline(x=0, color=".5")
    plt.xlabel("Raw coefficient values")
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Indeed, from the plot above the most important factor in determining WAGE
    appears to be the
    variable UNION, even if our intuition might tell us that variables
    like EXPERIENCE should have more impact.

    Looking at the coefficient plot to gauge feature importance can be
    misleading as some of them vary on a small scale, while others, like AGE,
    varies a lot more, several decades.

    This is visible if we compare the standard deviations of different
    features.


    """
    )
    return


@app.cell
def _(X_train, feature_names, model, pd, plt):
    X_train_preprocessed = pd.DataFrame(
        model[:-1].transform(X_train), columns=feature_names
    )

    X_train_preprocessed.std(axis=0).plot.barh(figsize=(9, 7))
    plt.title("Feature ranges")
    plt.xlabel("Std. dev. of feature values")
    plt.subplots_adjust(left=0.3)
    return (X_train_preprocessed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Multiplying the coefficients by the standard deviation of the related
    feature would reduce all the coefficients to the same unit of measure.
    As we will see `after<scaling_num>` this is equivalent to normalize
    numerical variables to their standard deviation,
    as $y = \sum{coef_i \times X_i} =
    \sum{(coef_i \times std_i) \times (X_i / std_i)}$.

    In that way, we emphasize that the
    greater the variance of a feature, the larger the weight of the corresponding
    coefficient on the output, all else being equal.


    """
    )
    return


@app.cell
def _(X_train_preprocessed, feature_names, model, pd, plt):
    coefs_1 = pd.DataFrame(model[-1].regressor_.coef_ * X_train_preprocessed.std(axis=0), columns=['Coefficient importance'], index=feature_names)
    coefs_1.plot(kind='barh', figsize=(9, 7))
    plt.xlabel("Coefficient values corrected by the feature's std. dev.")
    plt.title('Ridge model, small regularization')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now that the coefficients have been scaled, we can safely compare them.

    <div class="alert alert-info"><h4>Note</h4><p>Why does the plot above suggest that an increase in age leads to a
      decrease in wage? Why is the `initial pairplot
      <marginal_dependencies>` telling the opposite?
      This difference is the difference between marginal and conditional dependence.</p></div>

    The plot above tells us about dependencies between a specific feature and
    the target when all other features remain constant, i.e., **conditional
    dependencies**. An increase of the AGE will induce a decrease
    of the WAGE when all other features remain constant. On the contrary, an
    increase of the EXPERIENCE will induce an increase of the WAGE when all
    other features remain constant.
    Also, AGE, EXPERIENCE and EDUCATION are the three variables that most
    influence the model.

    ## Interpreting coefficients: being cautious about causality

    Linear models are a great tool for measuring statistical association, but we
    should be cautious when making statements about causality, after all
    correlation doesn't always imply causation. This is particularly difficult in
    the social sciences because the variables we observe only function as proxies
    for the underlying causal process.

    In our particular case we can think of the EDUCATION of an individual as a
    proxy for their professional aptitude, the real variable we're interested in
    but can't observe. We'd certainly like to think that staying in school for
    longer would increase technical competency, but it's also quite possible that
    causality goes the other way too. That is, those who are technically
    competent tend to stay in school for longer.

    An employer is unlikely to care which case it is (or if it's a mix of both),
    as long as they remain convinced that a person with more EDUCATION is better
    suited for the job, they will be happy to pay out a higher WAGE.

    This confounding of effects becomes problematic when thinking about some
    form of intervention e.g. government subsidies of university degrees or
    promotional material encouraging individuals to take up higher education.
    The usefulness of these measures could end up being overstated, especially if
    the degree of confounding is strong. Our model predicts a $0.054699$
    increase in hourly wage for each year of education. The actual causal effect
    might be lower because of this confounding.

    ## Checking the variability of the coefficients

    We can check the coefficient variability through cross-validation:
    it is a form of data perturbation (related to
    [resampling](https://en.wikipedia.org/wiki/Resampling_(statistics))).

    If coefficients vary significantly when changing the input dataset
    their robustness is not guaranteed, and they should probably be interpreted
    with caution.


    """
    )
    return


@app.cell
def _(X, feature_names, model, pd, y):
    from sklearn.model_selection import RepeatedKFold, cross_validate
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    _cv_model = cross_validate(model, X, y, cv=cv, return_estimator=True, n_jobs=2)
    coefs_2 = pd.DataFrame([est[-1].regressor_.coef_ * est[:-1].transform(X.iloc[train_idx]).std(axis=0) for est, (train_idx, _) in zip(_cv_model['estimator'], cv.split(X, y))], columns=feature_names)
    return coefs_2, cross_validate, cv


@app.cell
def _(coefs_2, plt, sns):
    plt.figure(figsize=(9, 7))
    sns.stripplot(data=coefs_2, orient='h', palette='dark:k', alpha=0.5)
    sns.boxplot(data=coefs_2, orient='h', color='cyan', saturation=0.5, whis=10)
    plt.axvline(x=0, color='.5')
    plt.xlabel('Coefficient importance')
    plt.title('Coefficient importance and its variability')
    plt.suptitle('Ridge model, small regularization')
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The problem of correlated variables

    The AGE and EXPERIENCE coefficients are affected by strong variability which
    might be due to the collinearity between the 2 features: as AGE and
    EXPERIENCE vary together in the data, their effect is difficult to tease
    apart.

    To verify this interpretation we plot the variability of the AGE and
    EXPERIENCE coefficient.



    """
    )
    return


@app.cell
def _(coefs_2, plt):
    plt.ylabel('Age coefficient')
    plt.xlabel('Experience coefficient')
    plt.grid(True)
    plt.xlim(-0.4, 0.5)
    plt.ylim(-0.4, 0.5)
    plt.scatter(coefs_2['AGE'], coefs_2['EXPERIENCE'])
    _ = plt.title('Co-variations of coefficients for AGE and EXPERIENCE across folds')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Two regions are populated: when the EXPERIENCE coefficient is
    positive the AGE one is negative and vice-versa.

    To go further we remove one of the two features, AGE, and check what is the impact
    on the model stability.


    """
    )
    return


@app.cell
def _(X, cross_validate, cv, feature_names, model, pd, y):
    column_to_drop = ['AGE']
    _cv_model = cross_validate(model, X.drop(columns=column_to_drop), y, cv=cv, return_estimator=True, n_jobs=2)
    coefs_3 = pd.DataFrame([est[-1].regressor_.coef_ * est[:-1].transform(X.drop(columns=column_to_drop).iloc[train_idx]).std(axis=0) for est, (train_idx, _) in zip(_cv_model['estimator'], cv.split(X, y))], columns=feature_names[:-1])
    return (coefs_3,)


@app.cell
def _(coefs_3, plt, sns):
    plt.figure(figsize=(9, 7))
    sns.stripplot(data=coefs_3, orient='h', palette='dark:k', alpha=0.5)
    sns.boxplot(data=coefs_3, orient='h', color='cyan', saturation=0.5)
    plt.axvline(x=0, color='.5')
    plt.title('Coefficient importance and its variability')
    plt.xlabel('Coefficient importance')
    plt.suptitle('Ridge model, small regularization, AGE dropped')
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The estimation of the EXPERIENCE coefficient now shows a much reduced
    variability. EXPERIENCE remains important for all models trained during
    cross-validation.


    ## Preprocessing numerical variables

    As said above (see "`the-pipeline`"), we could also choose to scale
    numerical values before training the model.
    This can be useful when we apply a similar amount of regularization to all of them
    in the ridge.
    The preprocessor is redefined in order to subtract the mean and scale
    variables to unit variance.


    """
    )
    return


@app.cell
def _(
    OneHotEncoder,
    categorical_columns,
    make_column_transformer,
    numerical_columns,
):
    from sklearn.preprocessing import StandardScaler
    preprocessor_1 = make_column_transformer((OneHotEncoder(drop='if_binary'), categorical_columns), (StandardScaler(), numerical_columns))
    return (preprocessor_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The model will stay unchanged.


    """
    )
    return


@app.cell
def _(
    Ridge,
    TransformedTargetRegressor,
    X_train,
    make_pipeline,
    np,
    preprocessor_1,
    sp,
    y_train,
):
    model_1 = make_pipeline(preprocessor_1, TransformedTargetRegressor(regressor=Ridge(alpha=1e-10), func=np.log10, inverse_func=sp.special.exp10))
    model_1.fit(X_train, y_train)
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Again, we check the performance of the computed
    model using the median absolute error.


    """
    )
    return


@app.cell
def _(
    PredictionErrorDisplay,
    X_test,
    X_train,
    median_absolute_error,
    model_1,
    plt,
    y_test,
    y_train,
):
    _mae_train = median_absolute_error(y_train, model_1.predict(X_train))
    y_pred_1 = model_1.predict(X_test)
    _mae_test = median_absolute_error(y_test, y_pred_1)
    scores_1 = {'MedAE on training set': f'{_mae_train:.2f} $/hour', 'MedAE on testing set': f'{_mae_test:.2f} $/hour'}
    _, _ax = plt.subplots(figsize=(5, 5))
    _display = PredictionErrorDisplay.from_predictions(y_test, y_pred_1, kind='actual_vs_predicted', ax=_ax, scatter_kwargs={'alpha': 0.5})
    _ax.set_title('Ridge model, small regularization')
    for _name, _score in scores_1.items():
        _ax.plot([], [], ' ', label=f'{_name}: {_score}')
    _ax.legend(loc='upper left')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For the coefficient analysis, scaling is not needed this time because it
    was performed during the preprocessing step.


    """
    )
    return


@app.cell
def _(feature_names, model_1, pd, plt):
    coefs_4 = pd.DataFrame(model_1[-1].regressor_.coef_, columns=['Coefficients importance'], index=feature_names)
    coefs_4.plot.barh(figsize=(9, 7))
    plt.title('Ridge model, small regularization, normalized variables')
    plt.xlabel('Raw coefficient values')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We now inspect the coefficients across several cross-validation folds.


    """
    )
    return


@app.cell
def _(X, cross_validate, cv, feature_names, model_1, pd, y):
    _cv_model = cross_validate(model_1, X, y, cv=cv, return_estimator=True, n_jobs=2)
    coefs_5 = pd.DataFrame([est[-1].regressor_.coef_ for est in _cv_model['estimator']], columns=feature_names)
    return (coefs_5,)


@app.cell
def _(coefs_5, plt, sns):
    plt.figure(figsize=(9, 7))
    sns.stripplot(data=coefs_5, orient='h', palette='dark:k', alpha=0.5)
    sns.boxplot(data=coefs_5, orient='h', color='cyan', saturation=0.5, whis=10)
    plt.axvline(x=0, color='.5')
    plt.title('Coefficient variability')
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The result is quite similar to the non-normalized case.

    ## Linear models with regularization

    In machine-learning practice, ridge regression is more often used with
    non-negligible regularization.

    Above, we limited this regularization to a very little amount. Regularization
    improves the conditioning of the problem and reduces the variance of the
    estimates. :class:`~sklearn.linear_model.RidgeCV` applies cross validation
    in order to determine which value of the regularization parameter (`alpha`)
    is best suited for prediction.


    """
    )
    return


@app.cell
def _(
    TransformedTargetRegressor,
    X_train,
    make_pipeline,
    np,
    preprocessor_1,
    sp,
    y_train,
):
    from sklearn.linear_model import RidgeCV
    _alphas = np.logspace(-10, 10, 21)
    model_2 = make_pipeline(preprocessor_1, TransformedTargetRegressor(regressor=RidgeCV(alphas=_alphas), func=np.log10, inverse_func=sp.special.exp10))
    model_2.fit(X_train, y_train)
    return (model_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    First we check which value of $\alpha$ has been selected.


    """
    )
    return


@app.cell
def _(model_2):
    model_2[-1].regressor_.alpha_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Then we check the quality of the predictions.


    """
    )
    return


@app.cell
def _(
    PredictionErrorDisplay,
    X_test,
    X_train,
    median_absolute_error,
    model_2,
    plt,
    y_test,
    y_train,
):
    _mae_train = median_absolute_error(y_train, model_2.predict(X_train))
    y_pred_2 = model_2.predict(X_test)
    _mae_test = median_absolute_error(y_test, y_pred_2)
    scores_2 = {'MedAE on training set': f'{_mae_train:.2f} $/hour', 'MedAE on testing set': f'{_mae_test:.2f} $/hour'}
    _, _ax = plt.subplots(figsize=(5, 5))
    _display = PredictionErrorDisplay.from_predictions(y_test, y_pred_2, kind='actual_vs_predicted', ax=_ax, scatter_kwargs={'alpha': 0.5})
    _ax.set_title('Ridge model, optimum regularization')
    for _name, _score in scores_2.items():
        _ax.plot([], [], ' ', label=f'{_name}: {_score}')
    _ax.legend(loc='upper left')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The ability to reproduce the data of the regularized model is similar to
    the one of the non-regularized model.


    """
    )
    return


@app.cell
def _(feature_names, model_2, pd, plt):
    coefs_6 = pd.DataFrame(model_2[-1].regressor_.coef_, columns=['Coefficients importance'], index=feature_names)
    coefs_6.plot.barh(figsize=(9, 7))
    plt.title('Ridge model, with regularization, normalized variables')
    plt.xlabel('Raw coefficient values')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The coefficients are significantly different.
    AGE and EXPERIENCE coefficients are both positive but they now have less
    influence on the prediction.

    The regularization reduces the influence of correlated
    variables on the model because the weight is shared between the two
    predictive variables, so neither alone would have strong weights.

    On the other hand, the weights obtained with regularization are more
    stable (see the `ridge_regression` User Guide section). This
    increased stability is visible from the plot, obtained from data
    perturbations, in a cross-validation. This plot can be compared with
    the `previous one<covariation>`.


    """
    )
    return


@app.cell
def _(X, cross_validate, cv, feature_names, model_2, pd, y):
    _cv_model = cross_validate(model_2, X, y, cv=cv, return_estimator=True, n_jobs=2)
    coefs_7 = pd.DataFrame([est[-1].regressor_.coef_ for est in _cv_model['estimator']], columns=feature_names)
    return (coefs_7,)


@app.cell
def _(coefs_7, plt):
    plt.ylabel('Age coefficient')
    plt.xlabel('Experience coefficient')
    plt.grid(True)
    plt.xlim(-0.4, 0.5)
    plt.ylim(-0.4, 0.5)
    plt.scatter(coefs_7['AGE'], coefs_7['EXPERIENCE'])
    _ = plt.title('Co-variations of coefficients for AGE and EXPERIENCE across folds')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Linear models with sparse coefficients

    Another possibility to take into account correlated variables in the dataset,
    is to estimate sparse coefficients. In some way we already did it manually
    when we dropped the AGE column in a previous ridge estimation.

    Lasso models (see the `lasso` User Guide section) estimates sparse
    coefficients. :class:`~sklearn.linear_model.LassoCV` applies cross
    validation in order to determine which value of the regularization parameter
    (`alpha`) is best suited for the model estimation.


    """
    )
    return


@app.cell
def _(
    TransformedTargetRegressor,
    X_train,
    make_pipeline,
    np,
    preprocessor_1,
    sp,
    y_train,
):
    from sklearn.linear_model import LassoCV
    _alphas = np.logspace(-10, 10, 21)
    model_3 = make_pipeline(preprocessor_1, TransformedTargetRegressor(regressor=LassoCV(alphas=_alphas, max_iter=100000), func=np.log10, inverse_func=sp.special.exp10))
    _ = model_3.fit(X_train, y_train)
    return (model_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    First we verify which value of $\alpha$ has been selected.


    """
    )
    return


@app.cell
def _(model_3):
    model_3[-1].regressor_.alpha_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Then we check the quality of the predictions.


    """
    )
    return


@app.cell
def _(
    PredictionErrorDisplay,
    X_test,
    X_train,
    median_absolute_error,
    model_3,
    plt,
    y_test,
    y_train,
):
    _mae_train = median_absolute_error(y_train, model_3.predict(X_train))
    y_pred_3 = model_3.predict(X_test)
    _mae_test = median_absolute_error(y_test, y_pred_3)
    scores_3 = {'MedAE on training set': f'{_mae_train:.2f} $/hour', 'MedAE on testing set': f'{_mae_test:.2f} $/hour'}
    _, _ax = plt.subplots(figsize=(6, 6))
    _display = PredictionErrorDisplay.from_predictions(y_test, y_pred_3, kind='actual_vs_predicted', ax=_ax, scatter_kwargs={'alpha': 0.5})
    _ax.set_title('Lasso model, optimum regularization')
    for _name, _score in scores_3.items():
        _ax.plot([], [], ' ', label=f'{_name}: {_score}')
    _ax.legend(loc='upper left')
    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For our dataset, again the model is not very predictive.


    """
    )
    return


@app.cell
def _(feature_names, model_3, pd, plt):
    coefs_8 = pd.DataFrame(model_3[-1].regressor_.coef_, columns=['Coefficients importance'], index=feature_names)
    coefs_8.plot(kind='barh', figsize=(9, 7))
    plt.title('Lasso model, optimum regularization, normalized variables')
    plt.axvline(x=0, color='.5')
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A Lasso model identifies the correlation between
    AGE and EXPERIENCE and suppresses one of them for the sake of the prediction.

    It is important to keep in mind that the coefficients that have been
    dropped may still be related to the outcome by themselves: the model
    chose to suppress them because they bring little or no additional
    information on top of the other features. Additionally, this selection
    is unstable for correlated features, and should be interpreted with
    caution.

    Indeed, we can check the variability of the coefficients across folds.


    """
    )
    return


@app.cell
def _(X, cross_validate, cv, feature_names, model_3, pd, y):
    _cv_model = cross_validate(model_3, X, y, cv=cv, return_estimator=True, n_jobs=2)
    coefs_9 = pd.DataFrame([est[-1].regressor_.coef_ for est in _cv_model['estimator']], columns=feature_names)
    return (coefs_9,)


@app.cell
def _(coefs_9, plt, sns):
    plt.figure(figsize=(9, 7))
    sns.stripplot(data=coefs_9, orient='h', palette='dark:k', alpha=0.5)
    sns.boxplot(data=coefs_9, orient='h', color='cyan', saturation=0.5, whis=100)
    plt.axvline(x=0, color='.5')
    plt.title('Coefficient variability')
    plt.subplots_adjust(left=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We observe that the AGE and EXPERIENCE coefficients are varying a lot
    depending of the fold.

    ## Wrong causal interpretation

    Policy makers might want to know the effect of education on wage to assess
    whether or not a certain policy designed to entice people to pursue more
    education would make economic sense. While Machine Learning models are great
    for measuring statistical associations, they are generally unable to infer
    causal effects.

    It might be tempting to look at the coefficient of education on wage from our
    last model (or any model for that matter) and conclude that it captures the
    true effect of a change in the standardized education variable on wages.

    Unfortunately there are likely unobserved confounding variables that either
    inflate or deflate that coefficient. A confounding variable is a variable that
    causes both EDUCATION and WAGE. One example of such variable is ability.
    Presumably, more able people are more likely to pursue education while at the
    same time being more likely to earn a higher hourly wage at any level of
    education. In this case, ability induces a positive [Omitted Variable Bias](https://en.wikipedia.org/wiki/Omitted-variable_bias) (OVB) on the EDUCATION
    coefficient, thereby exaggerating the effect of education on wages.

    See the `sphx_glr_auto_examples_inspection_plot_causal_interpretation.py`
    for a simulated case of ability OVB.

    ## Lessons learned

    * Coefficients must be scaled to the same unit of measure to retrieve
      feature importance. Scaling them with the standard-deviation of the
      feature is a useful proxy.
    * Coefficients in multivariate linear models represent the dependency
      between a given feature and the target, **conditional** on the other
      features.
    * Correlated features induce instabilities in the coefficients of linear
      models and their effects cannot be well teased apart.
    * Different linear models respond differently to feature correlation and
      coefficients could significantly vary from one another.
    * Inspecting coefficients across the folds of a cross-validation loop
      gives an idea of their stability.
    * Interpreting causality is difficult when there are confounding effects. If
      the relationship between two variables is also affected by something
      unobserved, we should be careful when making conclusions about causality.


    """
    )
    return

if __name__ == "__main__":
    app.run()
