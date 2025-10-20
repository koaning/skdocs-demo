import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Scaling the regularization parameter for SVCs

    The following example illustrates the effect of scaling the regularization
    parameter when using `svm` for `classification <svm_classification>`.
    For SVC classification, we are interested in a risk minimization for the
    equation:


    \begin{align}C \sum_{i=1, n} \mathcal{L} (f(x_i), y_i) + \Omega (w)\end{align}

    where

    - $C$ is used to set the amount of regularization
    - $\mathcal{L}$ is a `loss` function of our samples and our model parameters.
    - $\Omega$ is a `penalty` function of our model parameters

    If we consider the loss function to be the individual error per sample, then the
    data-fit term, or the sum of the error for each sample, increases as we add more
    samples. The penalization term, however, does not increase.

    When using, for example, `cross validation <cross_validation>`, to set the
    amount of regularization with `C`, there would be a different amount of samples
    between the main problem and the smaller problems within the folds of the cross
    validation.

    Since the loss function dependens on the amount of samples, the latter
    influences the selected value of `C`. The question that arises is "How do we
    optimally adjust C to account for the different amount of training samples?"

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
    ## Data generation

    In this example we investigate the effect of reparametrizing the regularization
    parameter `C` to account for the number of samples when using either L1 or L2
    penalty. For such purpose we create a synthetic dataset with a large number of
    features, out of which only a few are informative. We therefore expect the
    regularization to shrink the coefficients towards zero (L2 penalty) or exactly
    zero (L1 penalty).


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_classification

    n_samples, n_features = 100, 300
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=5, random_state=1
    )
    return X, n_samples, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## L1-penalty case
    In the L1 case, theory says that provided a strong regularization, the
    estimator cannot predict as well as a model knowing the true distribution
    (even in the limit where the sample size grows to infinity) as it may set some
    weights of otherwise predictive features to zero, which induces a bias. It does
    say, however, that it is possible to find the right set of non-zero parameters
    as well as their signs by tuning `C`.

    We define a linear SVC with the L1 penalty.


    """
    )
    return


@app.cell
def _():
    from sklearn.svm import LinearSVC

    model_l1 = LinearSVC(penalty="l1", loss="squared_hinge", dual=False, tol=1e-3)
    return LinearSVC, model_l1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We compute the mean test score for different values of `C` via
    cross-validation.


    """
    )
    return


@app.cell
def _(X, model_l1, y):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import ShuffleSplit, validation_curve
    Cs = np.logspace(-2.3, -1.3, 10)
    train_sizes = np.linspace(0.3, 0.7, 3)
    labels = [f'fraction: {_train_size}' for _train_size in train_sizes]
    shuffle_params = {'test_size': 0.3, 'n_splits': 150, 'random_state': 1}
    results = {'C': Cs}
    for _label, _train_size in zip(labels, train_sizes):
        _cv = ShuffleSplit(train_size=_train_size, **shuffle_params)
        _train_scores, _test_scores = validation_curve(model_l1, X, y, param_name='C', param_range=Cs, cv=_cv, n_jobs=2)
        results[_label] = _test_scores.mean(axis=1)
    results = pd.DataFrame(results)
    return (
        Cs,
        ShuffleSplit,
        labels,
        np,
        pd,
        results,
        shuffle_params,
        train_sizes,
        validation_curve,
    )


@app.cell
def _(Cs, labels, n_samples, np, results, train_sizes):
    import matplotlib.pyplot as plt
    _fig, _axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))
    results.plot(x='C', ax=_axes[0], logx=True)
    _axes[0].set_ylabel('CV score')
    # plot results without scaling C
    _axes[0].set_title('No scaling')
    for _label in labels:
        _best_C = results.loc[results[_label].idxmax(), 'C']
        _axes[0].axvline(x=_best_C, linestyle='--', color='grey', alpha=0.7)
    for _train_size_idx, _label in enumerate(labels):
        _train_size = train_sizes[_train_size_idx]
        _results_scaled = results[[_label]].assign(C_scaled=Cs * float(n_samples * np.sqrt(_train_size)))
        _results_scaled.plot(x='C_scaled', ax=_axes[1], logx=True, label=_label)
    # plot results by scaling C
        _best_C_scaled = _results_scaled['C_scaled'].loc[results[_label].idxmax()]
        _axes[1].axvline(x=_best_C_scaled, linestyle='--', color='grey', alpha=0.7)
    _axes[1].set_title('Scaling C by sqrt(1 / n_samples)')
    _ = _fig.suptitle('Effect of scaling C with L1 penalty')
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the region of small `C` (strong regularization) all the coefficients
    learned by the models are zero, leading to severe underfitting. Indeed, the
    accuracy in this region is at the chance level.

    Using the default scale results in a somewhat stable optimal value of `C`,
    whereas the transition out of the underfitting region depends on the number of
    training samples. The reparametrization leads to even more stable results.

    See e.g. theorem 3 of :arxiv:`On the prediction performance of the Lasso
    <1402.1700>` or :arxiv:`Simultaneous analysis of Lasso and Dantzig selector
    <0801.1095>` where the regularization parameter is always assumed to be
    proportional to 1 / sqrt(n_samples).

    ## L2-penalty case
    We can do a similar experiment with the L2 penalty. In this case, the
    theory says that in order to achieve prediction consistency, the penalty
    parameter should be kept constant as the number of samples grow.


    """
    )
    return


@app.cell
def _(
    LinearSVC,
    ShuffleSplit,
    X,
    np,
    pd,
    shuffle_params,
    train_sizes,
    validation_curve,
    y,
):
    model_l2 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True)
    Cs_1 = np.logspace(-8, 4, 11)
    labels_1 = [f'fraction: {_train_size}' for _train_size in train_sizes]
    results_1 = {'C': Cs_1}
    for _label, _train_size in zip(labels_1, train_sizes):
        _cv = ShuffleSplit(train_size=_train_size, **shuffle_params)
        _train_scores, _test_scores = validation_curve(model_l2, X, y, param_name='C', param_range=Cs_1, cv=_cv, n_jobs=2)
        results_1[_label] = _test_scores.mean(axis=1)
    results_1 = pd.DataFrame(results_1)
    return Cs_1, labels_1, results_1


@app.cell
def _(Cs_1, labels_1, n_samples, np, plt, results_1, train_sizes):
    _fig, _axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))
    results_1.plot(x='C', ax=_axes[0], logx=True)
    _axes[0].set_ylabel('CV score')
    _axes[0].set_title('No scaling')
    for _label in labels_1:
        _best_C = results_1.loc[results_1[_label].idxmax(), 'C']
        _axes[0].axvline(x=_best_C, linestyle='--', color='grey', alpha=0.8)
    for _train_size_idx, _label in enumerate(labels_1):
        _results_scaled = results_1[[_label]].assign(C_scaled=Cs_1 * float(n_samples * np.sqrt(train_sizes[_train_size_idx])))
        _results_scaled.plot(x='C_scaled', ax=_axes[1], logx=True, label=_label)
        _best_C_scaled = _results_scaled['C_scaled'].loc[results_1[_label].idxmax()]
        _axes[1].axvline(x=_best_C_scaled, linestyle='--', color='grey', alpha=0.8)
    _axes[1].set_title('Scaling C by sqrt(1 / n_samples)')
    _fig.suptitle('Effect of scaling C with L2 penalty')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For the L2 penalty case, the reparametrization seems to have a smaller impact
    on the stability of the optimal value for the regularization. The transition
    out of the overfitting region occurs in a more spread range and the accuracy
    does not seem to be degraded up to chance level.

    Try increasing the value to `n_splits=1_000` for better results in the L2
    case, which is not shown here due to the limitations on the documentation
    builder.


    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
