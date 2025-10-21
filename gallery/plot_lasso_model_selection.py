import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Lasso model selection: AIC-BIC / cross-validation

    This example focuses on model selection for Lasso models that are
    linear models with an L1 penalty for regression problems.

    Indeed, several strategies can be used to select the value of the
    regularization parameter: via cross-validation or using an information
    criterion, namely AIC or BIC.

    In what follows, we will discuss in details the different strategies.

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
    ## Dataset
    In this example, we will use the diabetes dataset.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X.head()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In addition, we add some random features to the original data to
    better illustrate the feature selection performed by the Lasso model.


    """
    )
    return


@app.cell
def _(X):
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(42)
    n_random_features = 14
    X_random = pd.DataFrame(rng.randn(X.shape[0], n_random_features), columns=[f'random_{i:02d}' for i in range(n_random_features)])
    X_1 = pd.concat([X, X_random], axis=1)
    # Show only a subset of the columns
    X_1[X_1.columns[::3]].head()
    return X_1, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Selecting Lasso via an information criterion
    :class:`~sklearn.linear_model.LassoLarsIC` provides a Lasso estimator that
    uses the Akaike information criterion (AIC) or the Bayes information
    criterion (BIC) to select the optimal value of the regularization
    parameter alpha.

    Before fitting the model, we will standardize the data with a
    :class:`~sklearn.preprocessing.StandardScaler`. In addition, we will
    measure the time to fit and tune the hyperparameter alpha in order to
    compare with the cross-validation strategy.

    We will first fit a Lasso model with the AIC criterion.


    """
    )
    return


@app.cell
def _(X_1, y):
    import time
    from sklearn.linear_model import LassoLarsIC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _start_time = time.time()
    lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion='aic')).fit(X_1, y)
    fit_time = time.time() - _start_time
    return StandardScaler, fit_time, lasso_lars_ic, make_pipeline, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We store the AIC metric for each value of alpha used during `fit`.


    """
    )
    return


@app.cell
def _(lasso_lars_ic, pd):
    results = pd.DataFrame(
        {
            "alphas": lasso_lars_ic[-1].alphas_,
            "AIC criterion": lasso_lars_ic[-1].criterion_,
        }
    ).set_index("alphas")
    alpha_aic = lasso_lars_ic[-1].alpha_
    return alpha_aic, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, we perform the same analysis using the BIC criterion.


    """
    )
    return


@app.cell
def _(X_1, lasso_lars_ic, results, y):
    lasso_lars_ic.set_params(lassolarsic__criterion='bic').fit(X_1, y)
    results['BIC criterion'] = lasso_lars_ic[-1].criterion_
    alpha_bic = lasso_lars_ic[-1].alpha_
    return (alpha_bic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can check which value of `alpha` leads to the minimum AIC and BIC.


    """
    )
    return


@app.cell
def _(results):
    def highlight_min(x):
        x_min = x.min()
        return ["font-weight: bold" if v == x_min else "" for v in x]


    results.style.apply(highlight_min)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Finally, we can plot the AIC and BIC values for the different alpha values.
    The vertical lines in the plot correspond to the alpha chosen for each
    criterion. The selected alpha corresponds to the minimum of the AIC or BIC
    criterion.


    """
    )
    return


@app.cell
def _(alpha_aic, alpha_bic, fit_time, results):
    ax = results.plot()
    ax.vlines(
        alpha_aic,
        results["AIC criterion"].min(),
        results["AIC criterion"].max(),
        label="alpha: AIC estimate",
        linestyles="--",
        color="tab:blue",
    )
    ax.vlines(
        alpha_bic,
        results["BIC criterion"].min(),
        results["BIC criterion"].max(),
        label="alpha: BIC estimate",
        linestyle="--",
        color="tab:orange",
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("criterion")
    ax.set_xscale("log")
    ax.legend()
    _ = ax.set_title(
        f"Information-criterion for model selection (training time {fit_time:.2f}s)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Model selection with an information-criterion is very fast. It relies on
    computing the criterion on the in-sample set provided to `fit`. Both criteria
    estimate the model generalization error based on the training set error and
    penalize this overly optimistic error. However, this penalty relies on a
    proper estimation of the degrees of freedom and the noise variance. Both are
    derived for large samples (asymptotic results) and assume the model is
    correct, i.e. that the data are actually generated by this model.

    These models also tend to break when the problem is badly conditioned (more
    features than samples). It is then required to provide an estimate of the
    noise variance.

    ## Selecting Lasso via cross-validation
    The Lasso estimator can be implemented with different solvers: coordinate
    descent and least angle regression. They differ with regards to their
    execution speed and sources of numerical errors.

    In scikit-learn, two different estimators are available with integrated
    cross-validation: :class:`~sklearn.linear_model.LassoCV` and
    :class:`~sklearn.linear_model.LassoLarsCV` that respectively solve the
    problem with coordinate descent and least angle regression.

    In the remainder of this section, we will present both approaches. For both
    algorithms, we will use a 20-fold cross-validation strategy.

    ### Lasso via coordinate descent
    Let's start by making the hyperparameter tuning using
    :class:`~sklearn.linear_model.LassoCV`.


    """
    )
    return


@app.cell
def _(StandardScaler, X_1, make_pipeline, time, y):
    from sklearn.linear_model import LassoCV
    _start_time = time.time()
    model = make_pipeline(StandardScaler(), LassoCV(cv=20)).fit(X_1, y)
    fit_time_1 = time.time() - _start_time
    return fit_time_1, model


@app.cell
def _(fit_time_1, model):
    import matplotlib.pyplot as plt
    ymin, ymax = (2300, 3800)
    _lasso = model[-1]
    plt.semilogx(_lasso.alphas_, _lasso.mse_path_, linestyle=':')
    plt.plot(_lasso.alphas_, _lasso.mse_path_.mean(axis=-1), color='black', label='Average across the folds', linewidth=2)
    plt.axvline(_lasso.alpha_, linestyle='--', color='black', label='alpha: CV estimate')
    plt.ylim(ymin, ymax)
    plt.xlabel('$\\alpha$')
    plt.ylabel('Mean square error')
    plt.legend()
    _ = plt.title(f'Mean square error on each fold: coordinate descent (train time: {fit_time_1:.2f}s)')
    return plt, ymax, ymin


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Lasso via least angle regression
    Let's start by making the hyperparameter tuning using
    :class:`~sklearn.linear_model.LassoLarsCV`.


    """
    )
    return


@app.cell
def _(StandardScaler, X_1, make_pipeline, time, y):
    from sklearn.linear_model import LassoLarsCV
    _start_time = time.time()
    model_1 = make_pipeline(StandardScaler(), LassoLarsCV(cv=20)).fit(X_1, y)
    fit_time_2 = time.time() - _start_time
    return fit_time_2, model_1


@app.cell
def _(fit_time_2, model_1, plt, ymax, ymin):
    _lasso = model_1[-1]
    plt.semilogx(_lasso.cv_alphas_, _lasso.mse_path_, ':')
    plt.semilogx(_lasso.cv_alphas_, _lasso.mse_path_.mean(axis=-1), color='black', label='Average across the folds', linewidth=2)
    plt.axvline(_lasso.alpha_, linestyle='--', color='black', label='alpha CV')
    plt.ylim(ymin, ymax)
    plt.xlabel('$\\alpha$')
    plt.ylabel('Mean square error')
    plt.legend()
    _ = plt.title(f'Mean square error on each fold: Lars (train time: {fit_time_2:.2f}s)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Summary of cross-validation approach
    Both algorithms give roughly the same results.

    Lars computes a solution path only for each kink in the path. As a result, it
    is very efficient when there are only of few kinks, which is the case if
    there are few features or samples. Also, it is able to compute the full path
    without setting any hyperparameter. On the opposite, coordinate descent
    computes the path points on a pre-specified grid (here we use the default).
    Thus it is more efficient if the number of grid points is smaller than the
    number of kinks in the path. Such a strategy can be interesting if the number
    of features is really large and there are enough samples to be selected in
    each of the cross-validation fold. In terms of numerical errors, for heavily
    correlated variables, Lars will accumulate more errors, while the coordinate
    descent algorithm will only sample the path on a grid.

    Note how the optimal value of alpha varies for each fold. This illustrates
    why nested-cross validation is a good strategy when trying to evaluate the
    performance of a method for which a parameter is chosen by cross-validation:
    this choice of parameter may not be optimal for a final evaluation on
    unseen test set only.

    ## Conclusion
    In this tutorial, we presented two approaches for selecting the best
    hyperparameter `alpha`: one strategy finds the optimal value of `alpha`
    by only using the training set and some information criterion, and another
    strategy is based on cross-validation.

    In this example, both approaches are working similarly. The in-sample
    hyperparameter selection even shows its efficacy in terms of computational
    performance. However, it can only be used when the number of samples is large
    enough compared to the number of features.

    That's why hyperparameter optimization via cross-validation is a safe
    strategy: it works in different settings.


    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
