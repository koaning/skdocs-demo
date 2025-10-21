import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Effect of transforming the targets in regression model

    In this example, we give an overview of
    :class:`~sklearn.compose.TransformedTargetRegressor`. We use two examples
    to illustrate the benefit of transforming the targets before learning a linear
    regression model. The first example uses synthetic data while the second
    example is based on the Ames housing data set.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    print(__doc__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Synthetic example

    A synthetic random regression dataset is generated. The targets ``y`` are
    modified by:

    1. translating all targets such that all entries are
       non-negative (by adding the absolute value of the lowest ``y``) and
    2. applying an exponential function to obtain non-linear
       targets which cannot be fitted using a simple linear model.

    Therefore, a logarithmic (`np.log1p`) and an exponential function
    (`np.expm1`) will be used to transform the targets before training a linear
    regression model and using it for prediction.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=10_000, noise=100, random_state=0)
    y = np.expm1((y + abs(y.min())) / 200)
    y_trans = np.log1p(y)
    return X, np, y, y_trans


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Below we plot the probability density functions of the target
    before and after applying the logarithmic functions.


    """
    )
    return


@app.cell
def _(X, y, y_trans):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    _f, (_ax0, _ax1) = plt.subplots(1, 2)
    _ax0.hist(y, bins=100, density=True)
    _ax0.set_xlim([0, 2000])
    _ax0.set_ylabel('Probability')
    _ax0.set_xlabel('Target')
    _ax0.set_title('Target distribution')
    _ax1.hist(y_trans, bins=100, density=True)
    _ax1.set_ylabel('Probability')
    _ax1.set_xlabel('Target')
    _ax1.set_title('Transformed target distribution')
    _f.suptitle('Synthetic data', y=1.05)
    plt.tight_layout()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_test, X_train, plt, train_test_split, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    At first, a linear model will be applied on the original targets. Due to the
    non-linearity, the model trained will not be precise during
    prediction. Subsequently, a logarithmic function is used to linearize the
    targets, allowing better prediction even with a similar linear model as
    reported by the median absolute error (MedAE).


    """
    )
    return


@app.cell
def _():
    from sklearn.metrics import median_absolute_error, r2_score

    def compute_score(y_true, y_pred):
        return {'R2': f'{r2_score(y_true, _y_pred):.3f}', 'MedAE': f'{median_absolute_error(y_true, _y_pred):.3f}'}
    return (compute_score,)


@app.cell
def _(X_test, X_train, compute_score, np, plt, y_test, y_train):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import PredictionErrorDisplay
    _f, (_ax0, _ax1) = plt.subplots(1, 2, sharey=True)
    _ridge_cv = RidgeCV().fit(X_train, y_train)
    _y_pred_ridge = _ridge_cv.predict(X_test)
    _ridge_cv_with_trans_target = TransformedTargetRegressor(regressor=RidgeCV(), func=np.log1p, inverse_func=np.expm1).fit(X_train, y_train)
    _y_pred_ridge_with_trans_target = _ridge_cv_with_trans_target.predict(X_test)
    PredictionErrorDisplay.from_predictions(y_test, _y_pred_ridge, kind='actual_vs_predicted', ax=_ax0, scatter_kwargs={'alpha': 0.5})
    PredictionErrorDisplay.from_predictions(y_test, _y_pred_ridge_with_trans_target, kind='actual_vs_predicted', ax=_ax1, scatter_kwargs={'alpha': 0.5})
    for _ax, _y_pred in zip([_ax0, _ax1], [_y_pred_ridge, _y_pred_ridge_with_trans_target]):
        for _name, _score in compute_score(y_test, _y_pred).items():
            _ax.plot([], [], ' ', label=f'{_name}={_score}')
        _ax.legend(loc='upper left')
    _ax0.set_title('Ridge regression \n without target transformation')
    _ax1.set_title('Ridge regression \n with target transformation')
    _f.suptitle('Synthetic data', y=1.05)
    # Add the score in the legend of each axis
    plt.tight_layout()
    return PredictionErrorDisplay, RidgeCV, TransformedTargetRegressor


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Real-world data set

    In a similar manner, the Ames housing data set is used to show the impact
    of transforming the targets before learning a model. In this example, the
    target to be predicted is the selling price of each house.


    """
    )
    return


@app.cell
def _(np):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import quantile_transform
    ames = fetch_openml(name='house_prices', as_frame=True)
    X_1 = ames.data.select_dtypes(np.number)
    # Keep only numeric columns
    X_1 = X_1.drop(columns=['LotFrontage', 'GarageYrBlt', 'MasVnrArea'])
    # Remove columns with NaN or Inf values
    y_1 = ames.target / 1000
    # Let the price be in k$
    y_trans_1 = quantile_transform(y_1.to_frame(), n_quantiles=900, output_distribution='normal', copy=True).squeeze()
    return X_1, y_1, y_trans_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A :class:`~sklearn.preprocessing.QuantileTransformer` is used to normalize
    the target distribution before applying a
    :class:`~sklearn.linear_model.RidgeCV` model.


    """
    )
    return


@app.cell
def _(plt, y_1, y_trans_1):
    _f, (_ax0, _ax1) = plt.subplots(1, 2)
    _ax0.hist(y_1, bins=100, density=True)
    _ax0.set_ylabel('Probability')
    _ax0.set_xlabel('Target')
    _ax0.set_title('Target distribution')
    _ax1.hist(y_trans_1, bins=100, density=True)
    _ax1.set_ylabel('Probability')
    _ax1.set_xlabel('Target')
    _ax1.set_title('Transformed target distribution')
    _f.suptitle('Ames housing data: selling price', y=1.05)
    plt.tight_layout()
    return


@app.cell
def _(X_1, train_test_split, y_1):
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, random_state=1)
    return X_test_1, X_train_1, y_test_1, y_train_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The effect of the transformer is weaker than on the synthetic data. However,
    the transformation results in an increase in $R^2$ and large decrease
    of the MedAE. The residual plot (predicted target - true target vs predicted
    target) without target transformation takes on a curved, 'reverse smile'
    shape due to residual values that vary depending on the value of predicted
    target. With target transformation, the shape is more linear indicating
    better model fit.


    """
    )
    return


@app.cell
def _(
    PredictionErrorDisplay,
    RidgeCV,
    TransformedTargetRegressor,
    X_test_1,
    X_train_1,
    compute_score,
    plt,
    y_test_1,
    y_train_1,
):
    from sklearn.preprocessing import QuantileTransformer
    _f, (_ax0, _ax1) = plt.subplots(2, 2, sharey='row', figsize=(6.5, 8))
    _ridge_cv = RidgeCV().fit(X_train_1, y_train_1)
    _y_pred_ridge = _ridge_cv.predict(X_test_1)
    _ridge_cv_with_trans_target = TransformedTargetRegressor(regressor=RidgeCV(), transformer=QuantileTransformer(n_quantiles=900, output_distribution='normal')).fit(X_train_1, y_train_1)
    _y_pred_ridge_with_trans_target = _ridge_cv_with_trans_target.predict(X_test_1)
    PredictionErrorDisplay.from_predictions(y_test_1, _y_pred_ridge, kind='actual_vs_predicted', ax=_ax0[0], scatter_kwargs={'alpha': 0.5})
    PredictionErrorDisplay.from_predictions(y_test_1, _y_pred_ridge_with_trans_target, kind='actual_vs_predicted', ax=_ax0[1], scatter_kwargs={'alpha': 0.5})
    for _ax, _y_pred in zip([_ax0[0], _ax0[1]], [_y_pred_ridge, _y_pred_ridge_with_trans_target]):
        for _name, _score in compute_score(y_test_1, _y_pred).items():
            _ax.plot([], [], ' ', label=f'{_name}={_score}')
        _ax.legend(loc='upper left')
    _ax0[0].set_title('Ridge regression \n without target transformation')
    _ax0[1].set_title('Ridge regression \n with target transformation')
    PredictionErrorDisplay.from_predictions(y_test_1, _y_pred_ridge, kind='residual_vs_predicted', ax=_ax1[0], scatter_kwargs={'alpha': 0.5})
    PredictionErrorDisplay.from_predictions(y_test_1, _y_pred_ridge_with_trans_target, kind='residual_vs_predicted', ax=_ax1[1], scatter_kwargs={'alpha': 0.5})
    _ax1[0].set_title('Ridge regression \n without target transformation')
    _ax1[1].set_title('Ridge regression \n with target transformation')
    _f.suptitle('Ames housing data: selling price', y=1.05)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
