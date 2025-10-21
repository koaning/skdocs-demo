import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Features in Histogram Gradient Boosting Trees

    `histogram_based_gradient_boosting` (HGBT) models may be one of the most
    useful supervised learning models in scikit-learn. They are based on a modern
    gradient boosting implementation comparable to LightGBM and XGBoost. As such,
    HGBT models are more feature rich than and often outperform alternative models
    like random forests, especially when the number of samples is larger than some
    ten thousands (see
    `sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py`).

    The top usability features of HGBT models are:

    1. Several available loss functions for mean and quantile regression tasks, see
       `Quantile loss <quantile_support_hgbdt>`.
    2. `categorical_support_gbdt`, see
       `sphx_glr_auto_examples_ensemble_plot_gradient_boosting_categorical.py`.
    3. Early stopping.
    4. `nan_support_hgbt`, which avoids the need for an imputer.
    5. `monotonic_cst_gbdt`.
    6. `interaction_cst_hgbt`.

    This example aims at showcasing all points except 2 and 6 in a real life
    setting.

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
    ## Preparing the data
    The [electricity dataset](http://www.openml.org/d/151) consists of data
    collected from the Australian New South Wales Electricity Market. In this
    market, prices are not fixed and are affected by supply and demand. They are
    set every five minutes. Electricity transfers to/from the neighboring state of
    Victoria were done to alleviate fluctuations.

    The dataset, originally named ELEC2, contains 45,312 instances dated from 7
    May 1996 to 5 December 1998. Each sample of the dataset refers to a period of
    30 minutes, i.e. there are 48 instances for each time period of one day. Each
    sample on the dataset has 7 columns:

    - date: between 7 May 1996 to 5 December 1998. Normalized between 0 and 1;
    - day: day of week (1-7);
    - period: half hour intervals over 24 hours. Normalized between 0 and 1;
    - nswprice/nswdemand: electricity price/demand of New South Wales;
    - vicprice/vicdemand: electricity price/demand of Victoria.

    Originally, it is a classification task, but here we use it for the regression
    task to predict the scheduled electricity transfer between states.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    electricity = fetch_openml(
        name="electricity", version=1, as_frame=True, parser="pandas"
    )
    df = electricity.frame
    return df, electricity


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This particular dataset has a stepwise constant target for the first 17,760
    samples:


    """
    )
    return


@app.cell
def _(df):
    df["transfer"][:17_760].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let us drop those entries and explore the hourly electricity transfer over
    different days of the week:


    """
    )
    return


@app.cell
def _(electricity):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_1 = electricity.frame.iloc[17760:]
    X = df_1.drop(columns=['transfer', 'class'])
    y = df_1['transfer']
    _fig, _ax = plt.subplots(figsize=(15, 10))
    pointplot = sns.lineplot(x=df_1['period'], y=df_1['transfer'], hue=df_1['day'], ax=_ax)
    handles, labels = _ax.get_legend_handles_labels()
    _ax.set(title='Hourly energy transfer for different days of the week', xlabel='Normalized time of the day', ylabel='Normalized energy transfer')
    _ = _ax.legend(handles, ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
    return X, df_1, plt, sns, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Notice that energy transfer increases systematically during weekends.

    ## Effect of number of trees and early stopping
    For the sake of illustrating the effect of the (maximum) number of trees, we
    train a :class:`~sklearn.ensemble.HistGradientBoostingRegressor` over the
    daily electricity transfer using the whole dataset. Then we visualize its
    predictions depending on the `max_iter` parameter. Here we don't try to
    evaluate the performance of the model and its capacity to generalize but
    rather its capability to learn from the training data.


    """
    )
    return


@app.cell
def _(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

    print(f"Training sample size: {X_train.shape[0]}")
    print(f"Test sample size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    return HistGradientBoostingRegressor, X_test, X_train, y_test, y_train


@app.cell
def _(HistGradientBoostingRegressor, X_test, X_train, df_1, plt, sns, y_train):
    max_iter_list = [5, 50]
    average_week_demand = df_1.loc[X_test.index].groupby(['day', 'period'], observed=False)['transfer'].mean()
    colors = sns.color_palette('colorblind')
    _fig, _ax = plt.subplots(figsize=(10, 5))
    average_week_demand.plot(color=colors[0], label='recorded average', linewidth=2, ax=_ax)
    for idx, max_iter in enumerate(max_iter_list):
        hgbt = HistGradientBoostingRegressor(max_iter=max_iter, categorical_features=None, random_state=42)
        hgbt.fit(X_train, y_train)
        _y_pred = hgbt.predict(X_test)
        prediction_df = df_1.loc[X_test.index].copy()
        prediction_df['y_pred'] = _y_pred
        average_pred = prediction_df.groupby(['day', 'period'], observed=False)['y_pred'].mean()
        average_pred.plot(color=colors[idx + 1], label=f'max_iter={max_iter}', linewidth=2, ax=_ax)
    _ax.set(title='Predicted average energy transfer during the week', xticks=[(i + 0.2) * 48 for i in range(7)], xticklabels=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], xlabel='Time of the week', ylabel='Normalized energy transfer')
    _ = _ax.legend()
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    With just a few iterations, HGBT models can achieve convergence (see
    `sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py`),
    meaning that adding more trees does not improve the model anymore. In the
    figure above, 5 iterations are not enough to get good predictions. With 50
    iterations, we are already able to do a good job.

    Setting `max_iter` too high might degrade the prediction quality and cost a lot of
    avoidable computing resources. Therefore, the HGBT implementation in scikit-learn
    provides an automatic **early stopping** strategy. With it, the model
    uses a fraction of the training data as internal validation set
    (`validation_fraction`) and stops training if the validation score does not
    improve (or degrades) after `n_iter_no_change` iterations up to a certain
    tolerance (`tol`).

    Notice that there is a trade-off between `learning_rate` and `max_iter`:
    Generally, smaller learning rates are preferable but require more iterations
    to converge to the minimum loss, while larger learning rates converge faster
    (less iterations/trees needed) but at the cost of a larger minimum loss.

    Because of this high correlation between the learning rate the number of iterations,
    a good practice is to tune the learning rate along with all (important) other
    hyperparameters, fit the HBGT on the training set with a large enough value
    for `max_iter` and determine the best `max_iter` via early stopping and some
    explicit `validation_fraction`.


    """
    )
    return


@app.cell
def _(HistGradientBoostingRegressor, X_train, plt, y_train):
    common_params = {'max_iter': 1000, 'learning_rate': 0.3, 'validation_fraction': 0.2, 'random_state': 42, 'categorical_features': None, 'scoring': 'neg_root_mean_squared_error'}
    hgbt_1 = HistGradientBoostingRegressor(early_stopping=True, **common_params)
    hgbt_1.fit(X_train, y_train)
    _, _ax = plt.subplots()
    plt.plot(-hgbt_1.validation_score_)
    _ = _ax.set(xlabel='number of iterations', ylabel='root mean squared error', title=f'Loss of hgbt with early stopping (n_iter={hgbt_1.n_iter_})')
    return common_params, hgbt_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can then overwrite the value for `max_iter` to a reasonable value and avoid
    the extra computational cost of the inner validation. Rounding up the number
    of iterations may account for variability of the training set:


    """
    )
    return


@app.cell
def _(HistGradientBoostingRegressor, common_params, hgbt_1):
    import math
    common_params['max_iter'] = math.ceil(hgbt_1.n_iter_ / 100) * 100
    common_params['early_stopping'] = False
    hgbt_2 = HistGradientBoostingRegressor(**common_params)
    return (hgbt_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <div class="alert alert-info"><h4>Note</h4><p>The inner validation done during early stopping is not optimal for
       time series.</p></div>

    ## Support for missing values
    HGBT models have native support of missing values. During training, the tree
    grower decides where samples with missing values should go (left or right
    child) at each split, based on the potential gain. When predicting, these
    samples are sent to the learnt child accordingly. If a feature had no missing
    values during training, then for prediction, samples with missing values for that
    feature are sent to the child with the most samples (as seen during fit).

    The present example shows how HGBT regressions deal with values missing
    completely at random (MCAR), i.e. the missingness does not depend on the
    observed data or the unobserved data. We can simulate such scenario by
    randomly replacing values from randomly selected features with `nan` values.


    """
    )
    return


@app.cell
def _(X_test, X_train, hgbt_2, plt, y_test, y_train):
    import numpy as np
    from sklearn.metrics import root_mean_squared_error
    rng = np.random.RandomState(42)
    first_week = slice(0, 336)
    missing_fraction_list = [0, 0.01, 0.03]

    def generate_missing_values(X, missing_fraction):
        total_cells = X.shape[0] * X.shape[1]
        num_missing_cells = int(total_cells * missing_fraction)
        row_indices = rng.choice(X.shape[0], num_missing_cells, replace=True)
        col_indices = rng.choice(X.shape[1], num_missing_cells, replace=True)
        X_missing = X.copy()
        X_missing.iloc[row_indices, col_indices] = np.nan
        return X_missing
    _fig, _ax = plt.subplots(figsize=(12, 6))
    _ax.plot(y_test.values[first_week], label='Actual transfer')
    for missing_fraction in missing_fraction_list:
        X_train_missing = generate_missing_values(X_train, missing_fraction)
        X_test_missing = generate_missing_values(X_test, missing_fraction)
        hgbt_2.fit(X_train_missing, y_train)
        _y_pred = hgbt_2.predict(X_test_missing[first_week])
        _rmse = root_mean_squared_error(y_test[first_week], _y_pred)
        _ax.plot(_y_pred[first_week], label=f'missing_fraction={missing_fraction}, RMSE={_rmse:.3f}', alpha=0.5)
    _ax.set(title='Daily energy transfer predictions on data with MCAR values', xticks=[(i + 0.2) * 48 for i in range(7)], xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], xlabel='Time of the week', ylabel='Normalized energy transfer')
    _ = _ax.legend(loc='lower right')
    return first_week, root_mean_squared_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As expected, the model degrades as the proportion of missing values increases.

    ## Support for quantile loss

    The quantile loss in regression enables a view of the variability or
    uncertainty of the target variable. For instance, predicting the 5th and 95th
    percentiles can provide a 90% prediction interval, i.e. the range within which
    we expect a new observed value to fall with 90% probability.


    """
    )
    return


@app.cell
def _(
    HistGradientBoostingRegressor,
    X_test,
    X_train,
    colors,
    common_params,
    first_week,
    plt,
    y_test,
    y_train,
):
    from sklearn.metrics import mean_pinball_loss
    quantiles = [0.95, 0.05]
    predictions = []
    _fig, _ax = plt.subplots(figsize=(12, 6))
    _ax.plot(y_test.values[first_week], label='Actual transfer')
    for quantile in quantiles:
        hgbt_quantile = HistGradientBoostingRegressor(loss='quantile', quantile=quantile, **common_params)
        hgbt_quantile.fit(X_train, y_train)
        _y_pred = hgbt_quantile.predict(X_test[first_week])
        predictions.append(_y_pred)
        score = mean_pinball_loss(y_test[first_week], _y_pred)
        _ax.plot(_y_pred[first_week], label=f'quantile={quantile}, pinball loss={score:.2f}', alpha=0.5)
    _ax.fill_between(range(len(predictions[0][first_week])), predictions[0][first_week], predictions[1][first_week], color=colors[0], alpha=0.1)
    _ax.set(title='Daily energy transfer predictions with quantile loss', xticks=[(i + 0.2) * 48 for i in range(7)], xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], xlabel='Time of the week', ylabel='Normalized energy transfer')
    _ = _ax.legend(loc='lower right')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We observe a tendence to over-estimate the energy transfer. This could be be
    quantitatively confirmed by computing empirical coverage numbers as done in
    the `calibration of confidence intervals section <calibration-section>`.
    Keep in mind that those predicted percentiles are just estimations from a
    model. One can still improve the quality of such estimations by:

    - collecting more data-points;
    - better tuning of the model hyperparameters, see
      `sphx_glr_auto_examples_ensemble_plot_gradient_boosting_quantile.py`;
    - engineering more predictive features from the same data, see
      `sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`.

    ## Monotonic constraints

    Given specific domain knowledge that requires the relationship between a
    feature and the target to be monotonically increasing or decreasing, one can
    enforce such behaviour in the predictions of a HGBT model using monotonic
    constraints. This makes the model more interpretable and can reduce its
    variance (and potentially mitigate overfitting) at the risk of increasing
    bias. Monotonic constraints can also be used to enforce specific regulatory
    requirements, ensure compliance and align with ethical considerations.

    In the present example, the policy of transferring energy from Victoria to New
    South Wales is meant to alleviate price fluctuations, meaning that the model
    predictions have to enforce such goal, i.e. transfer should increase with
    price and demand in New South Wales, but also decrease with price and demand
    in Victoria, in order to benefit both populations.

    If the training data has feature names, it’s possible to specify the monotonic
    constraints by passing a dictionary with the convention:

    - 1: monotonic increase
    - 0: no constraint
    - -1: monotonic decrease

    Alternatively, one can pass an array-like object encoding the above convention by
    position.


    """
    )
    return


@app.cell
def _(HistGradientBoostingRegressor, X, plt, y):
    from sklearn.inspection import PartialDependenceDisplay
    monotonic_cst = {'date': 0, 'day': 0, 'period': 0, 'nswdemand': 1, 'nswprice': 1, 'vicdemand': -1, 'vicprice': -1}
    hgbt_no_cst = HistGradientBoostingRegressor(categorical_features=None, random_state=42).fit(X, y)
    hgbt_cst = HistGradientBoostingRegressor(monotonic_cst=monotonic_cst, categorical_features=None, random_state=42).fit(X, y)
    _fig, _ax = plt.subplots(nrows=2, figsize=(15, 10))
    disp = PartialDependenceDisplay.from_estimator(hgbt_no_cst, X, features=['nswdemand', 'nswprice'], line_kw={'linewidth': 2, 'label': 'unconstrained', 'color': 'tab:blue'}, ax=_ax[0])
    PartialDependenceDisplay.from_estimator(hgbt_cst, X, features=['nswdemand', 'nswprice'], line_kw={'linewidth': 2, 'label': 'constrained', 'color': 'tab:orange'}, ax=disp.axes_)
    disp = PartialDependenceDisplay.from_estimator(hgbt_no_cst, X, features=['vicdemand', 'vicprice'], line_kw={'linewidth': 2, 'label': 'unconstrained', 'color': 'tab:blue'}, ax=_ax[1])
    PartialDependenceDisplay.from_estimator(hgbt_cst, X, features=['vicdemand', 'vicprice'], line_kw={'linewidth': 2, 'label': 'constrained', 'color': 'tab:orange'}, ax=disp.axes_)
    _ = plt.legend()
    return hgbt_cst, hgbt_no_cst


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Observe that `nswdemand` and `vicdemand` seem already monotonic without constraint.
    This is a good example to show that the model with monotonicity constraints is
    "overconstraining".

    Additionally, we can verify that the predictive quality of the model is not
    significantly degraded by introducing the monotonic constraints. For such
    purpose we use :class:`~sklearn.model_selection.TimeSeriesSplit`
    cross-validation to estimate the variance of the test score. By doing so we
    guarantee that the training data does not succeed the testing data, which is
    crucial when dealing with data that have a temporal relationship.


    """
    )
    return


@app.cell
def _(X, hgbt_cst, hgbt_no_cst, root_mean_squared_error, y):
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import TimeSeriesSplit, cross_validate
    ts_cv = TimeSeriesSplit(n_splits=5, gap=48, test_size=336)
    scorer = make_scorer(root_mean_squared_error)
    cv_results = cross_validate(hgbt_no_cst, X, y, cv=ts_cv, scoring=scorer)
    _rmse = cv_results['test_score']
    print(f'RMSE without constraints = {_rmse.mean():.3f} +/- {_rmse.std():.3f}')
    cv_results = cross_validate(hgbt_cst, X, y, cv=ts_cv, scoring=scorer)
    _rmse = cv_results['test_score']
    print(f'RMSE with constraints    = {_rmse.mean():.3f} +/- {_rmse.std():.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    That being said, notice the comparison is between two different models that
    may be optimized by a different combination of hyperparameters. That is the
    reason why we do no use the `common_params` in this section as done before.


    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
