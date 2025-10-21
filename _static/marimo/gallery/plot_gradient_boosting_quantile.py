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

    # Prediction Intervals for Gradient Boosting Regression

    This example shows how quantile regression can be used to create prediction
    intervals. See `sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py`
    for an example showcasing some other features of
    :class:`~ensemble.HistGradientBoostingRegressor`.

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
    Generate some data for a synthetic regression problem by applying the
    function f to uniformly sampled random inputs.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.model_selection import train_test_split


    def f(x):
        """The function to predict."""
        return x * np.sin(x)


    rng = np.random.RandomState(42)
    X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    expected_y = f(X).ravel()
    return X, expected_y, f, np, rng, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To make the problem interesting, we generate observations of the target y as
    the sum of a deterministic term computed by the function f and a random noise
    term that follows a centered [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution). To make this even
    more interesting we consider the case where the amplitude of the noise
    depends on the input variable x (heteroscedastic noise).

    The lognormal distribution is non-symmetric and long tailed: observing large
    outliers is likely but it is impossible to observe small outliers.


    """
    )
    return


@app.cell
def _(X, expected_y, np, rng):
    sigma = 0.5 + X.ravel() / 10
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = expected_y + noise
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Split into train, test datasets:


    """
    )
    return


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fitting non-linear quantile and least squares regressors

    Fit gradient boosting models trained with the quantile loss and
    alpha=0.05, 0.5, 0.95.

    The models obtained for alpha=0.05 and alpha=0.95 produce a 90% confidence
    interval (95% - 5% = 90%).

    The model trained with alpha=0.5 produces a regression of the median: on
    average, there should be the same number of target observations above and
    below the predicted values.


    """
    )
    return


@app.cell
def _(X_train, y_train):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_pinball_loss, mean_squared_error
    all_models = {}
    common_params = dict(learning_rate=0.05, n_estimators=200, max_depth=2, min_samples_leaf=9, min_samples_split=9)
    for _alpha in [0.05, 0.5, 0.95]:
        _gbr = GradientBoostingRegressor(loss='quantile', alpha=_alpha, **common_params)
        all_models['q %1.2f' % _alpha] = _gbr.fit(X_train, y_train)
    return (
        GradientBoostingRegressor,
        all_models,
        common_params,
        mean_pinball_loss,
        mean_squared_error,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Notice that :class:`~sklearn.ensemble.HistGradientBoostingRegressor` is much
    faster than :class:`~sklearn.ensemble.GradientBoostingRegressor` starting with
    intermediate datasets (`n_samples >= 10_000`), which is not the case of the
    present example.

    For the sake of comparison, we also fit a baseline model trained with the
    usual (mean) squared error (MSE).


    """
    )
    return


@app.cell
def _(GradientBoostingRegressor, X_train, all_models, common_params, y_train):
    gbr_ls = GradientBoostingRegressor(loss="squared_error", **common_params)
    all_models["mse"] = gbr_ls.fit(X_train, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Create an evenly spaced evaluation set of input values spanning the [0, 10]
    range.


    """
    )
    return


@app.cell
def _(np):
    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    return (xx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Plot the true conditional mean function f, the predictions of the conditional
    mean (loss equals squared error), the conditional median and the conditional
    90% interval (from 5th to 95th conditional percentiles).


    """
    )
    return


@app.cell
def _(X_test, all_models, f, xx, y_test):
    import matplotlib.pyplot as plt
    _y_pred = all_models['mse'].predict(xx)
    _y_lower = all_models['q 0.05'].predict(xx)
    _y_upper = all_models['q 0.95'].predict(xx)
    y_med = all_models['q 0.50'].predict(xx)
    _fig = plt.figure(figsize=(10, 10))
    plt.plot(xx, f(xx), 'black', linewidth=3, label='$f(x) = x\\,\\sin(x)$')
    plt.plot(X_test, y_test, 'b.', markersize=10, label='Test observations')
    plt.plot(xx, y_med, 'tab:orange', linewidth=3, label='Predicted median')
    plt.plot(xx, _y_pred, 'tab:green', linewidth=3, label='Predicted mean')
    plt.fill_between(xx.ravel(), _y_lower, _y_upper, alpha=0.4, label='Predicted 90% interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 25)
    plt.legend(loc='upper left')
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Comparing the predicted median with the predicted mean, we note that the
    median is on average below the mean as the noise is skewed towards high
    values (large outliers). The median estimate also seems to be smoother
    because of its natural robustness to outliers.

    Also observe that the inductive bias of gradient boosting trees is
    unfortunately preventing our 0.05 quantile to fully capture the sinoisoidal
    shape of the signal, in particular around x=8. Tuning hyper-parameters can
    reduce this effect as shown in the last part of this notebook.

    ## Analysis of the error metrics

    Measure the models with :func:`~sklearn.metrics.mean_squared_error` and
    :func:`~sklearn.metrics.mean_pinball_loss` metrics on the training dataset.


    """
    )
    return


@app.cell
def _(X_train, all_models, mean_pinball_loss, mean_squared_error, y_train):
    import pandas as pd

    def highlight_min(x):
        x_min = x.min()
        return ['font-weight: bold' if v == x_min else '' for v in x]
    _results = []
    for _name, _gbr in sorted(all_models.items()):
        _metrics = {'model': _name}
        _y_pred = _gbr.predict(X_train)
        for _alpha in [0.05, 0.5, 0.95]:
            _metrics['pbl=%1.2f' % _alpha] = mean_pinball_loss(y_train, _y_pred, alpha=_alpha)
        _metrics['MSE'] = mean_squared_error(y_train, _y_pred)
        _results.append(_metrics)
    pd.DataFrame(_results).set_index('model').style.apply(highlight_min)
    return highlight_min, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    One column shows all models evaluated by the same metric. The minimum number
    on a column should be obtained when the model is trained and measured with
    the same metric. This should be always the case on the training set if the
    training converged.

    Note that because the target distribution is asymmetric, the expected
    conditional mean and conditional median are significantly different and
    therefore one could not use the squared error model get a good estimation of
    the conditional median nor the converse.

    If the target distribution were symmetric and had no outliers (e.g. with a
    Gaussian noise), then median estimator and the least squares estimator would
    have yielded similar predictions.

    We then do the same on the test set.


    """
    )
    return


@app.cell
def _(
    X_test,
    all_models,
    highlight_min,
    mean_pinball_loss,
    mean_squared_error,
    pd,
    y_test,
):
    _results = []
    for _name, _gbr in sorted(all_models.items()):
        _metrics = {'model': _name}
        _y_pred = _gbr.predict(X_test)
        for _alpha in [0.05, 0.5, 0.95]:
            _metrics['pbl=%1.2f' % _alpha] = mean_pinball_loss(y_test, _y_pred, alpha=_alpha)
        _metrics['MSE'] = mean_squared_error(y_test, _y_pred)
        _results.append(_metrics)
    pd.DataFrame(_results).set_index('model').style.apply(highlight_min)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Errors are higher meaning the models slightly overfitted the data. It still
    shows that the best test metric is obtained when the model is trained by
    minimizing this same metric.

    Note that the conditional median estimator is competitive with the squared
    error estimator in terms of MSE on the test set: this can be explained by
    the fact the squared error estimator is very sensitive to large outliers
    which can cause significant overfitting. This can be seen on the right hand
    side of the previous plot. The conditional median estimator is biased
    (underestimation for this asymmetric noise) but is also naturally robust to
    outliers and overfits less.


    ## Calibration of the confidence interval

    We can also evaluate the ability of the two extreme quantile estimators at
    producing a well-calibrated conditional 90%-confidence interval.

    To do this we can compute the fraction of observations that fall between the
    predictions:


    """
    )
    return


@app.cell
def _(X_train, all_models, np, y_train):
    def coverage_fraction(y, y_low, y_high):
        return np.mean(np.logical_and(y >= y_low, y <= y_high))


    coverage_fraction(
        y_train,
        all_models["q 0.05"].predict(X_train),
        all_models["q 0.95"].predict(X_train),
    )
    return (coverage_fraction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On the training set the calibration is very close to the expected coverage
    value for a 90% confidence interval.


    """
    )
    return


@app.cell
def _(X_test, all_models, coverage_fraction, y_test):
    coverage_fraction(
        y_test, all_models["q 0.05"].predict(X_test), all_models["q 0.95"].predict(X_test)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On the test set, the estimated confidence interval is slightly too narrow.
    Note, however, that we would need to wrap those metrics in a cross-validation
    loop to assess their variability under data resampling.

    ## Tuning the hyper-parameters of the quantile regressors

    In the plot above, we observed that the 5th percentile regressor seems to
    underfit and could not adapt to sinusoidal shape of the signal.

    The hyper-parameters of the model were approximately hand-tuned for the
    median regressor and there is no reason that the same hyper-parameters are
    suitable for the 5th percentile regressor.

    To confirm this hypothesis, we tune the hyper-parameters of a new regressor
    of the 5th percentile by selecting the best model parameters by
    cross-validation on the pinball loss with alpha=0.05:


    """
    )
    return


@app.cell
def _(GradientBoostingRegressor, X_train, mean_pinball_loss, y_train):
    from pprint import pprint
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.metrics import make_scorer  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV
    param_grid = dict(learning_rate=[0.05, 0.1, 0.2], max_depth=[2, 5, 10], min_samples_leaf=[1, 5, 10, 20], min_samples_split=[5, 10, 20, 30, 50])
    _alpha = 0.05
    neg_mean_pinball_loss_05p_scorer = make_scorer(mean_pinball_loss, alpha=_alpha, greater_is_better=False)
    _gbr = GradientBoostingRegressor(loss='quantile', alpha=_alpha, random_state=0)
    search_05p = HalvingRandomSearchCV(_gbr, param_grid, resource='n_estimators', max_resources=250, min_resources=50, scoring=neg_mean_pinball_loss_05p_scorer, n_jobs=2, random_state=0).fit(X_train, y_train)
    pprint(search_05p.best_params_)  # maximize the negative loss
    return make_scorer, pprint, search_05p


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We observe that the hyper-parameters that were hand-tuned for the median
    regressor are in the same range as the hyper-parameters suitable for the 5th
    percentile regressor.

    Let's now tune the hyper-parameters for the 95th percentile regressor. We
    need to redefine the `scoring` metric used to select the best model, along
    with adjusting the alpha parameter of the inner gradient boosting estimator
    itself:


    """
    )
    return


@app.cell
def _(X_train, make_scorer, mean_pinball_loss, pprint, search_05p, y_train):
    from sklearn.base import clone
    _alpha = 0.95
    neg_mean_pinball_loss_95p_scorer = make_scorer(mean_pinball_loss, alpha=_alpha, greater_is_better=False)
    search_95p = clone(search_05p).set_params(estimator__alpha=_alpha, scoring=neg_mean_pinball_loss_95p_scorer)
    search_95p.fit(X_train, y_train)
    pprint(search_95p.best_params_)  # maximize the negative loss
    return (search_95p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The result shows that the hyper-parameters for the 95th percentile regressor
    identified by the search procedure are roughly in the same range as the hand-tuned
    hyper-parameters for the median regressor and the hyper-parameters
    identified by the search procedure for the 5th percentile regressor. However,
    the hyper-parameter searches did lead to an improved 90% confidence interval
    that is comprised by the predictions of those two tuned quantile regressors.
    Note that the prediction of the upper 95th percentile has a much coarser shape
    than the prediction of the lower 5th percentile because of the outliers:


    """
    )
    return


@app.cell
def _(X_test, f, plt, search_05p, search_95p, xx, y_test):
    _y_lower = search_05p.predict(xx)
    _y_upper = search_95p.predict(xx)
    _fig = plt.figure(figsize=(10, 10))
    plt.plot(xx, f(xx), 'black', linewidth=3, label='$f(x) = x\\,\\sin(x)$')
    plt.plot(X_test, y_test, 'b.', markersize=10, label='Test observations')
    plt.fill_between(xx.ravel(), _y_lower, _y_upper, alpha=0.4, label='Predicted 90% interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 25)
    plt.legend(loc='upper left')
    plt.title('Prediction with tuned hyper-parameters')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The plot looks qualitatively better than for the untuned models, especially
    for the shape of the of lower quantile.

    We now quantitatively evaluate the joint-calibration of the pair of
    estimators:


    """
    )
    return


@app.cell
def _(X_train, coverage_fraction, search_05p, search_95p, y_train):
    coverage_fraction(y_train, search_05p.predict(X_train), search_95p.predict(X_train))
    return


@app.cell
def _(X_test, coverage_fraction, search_05p, search_95p, y_test):
    coverage_fraction(y_test, search_05p.predict(X_test), search_95p.predict(X_test))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The calibration of the tuned pair is sadly not better on the test set: the
    width of the estimated confidence interval is still too narrow.

    Again, we would need to wrap this study in a cross-validation loop to
    better assess the variability of those estimates.


    """
    )
    return

if __name__ == "__main__":
    app.run()
