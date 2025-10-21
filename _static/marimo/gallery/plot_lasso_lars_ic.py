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

    # Lasso model selection via information criteria

    This example reproduces the example of Fig. 2 of [ZHT2007]_. A
    :class:`~sklearn.linear_model.LassoLarsIC` estimator is fit on a
    diabetes dataset and the AIC and the BIC criteria are used to select
    the best model.

    <div class="alert alert-info"><h4>Note</h4><p>It is important to note that the optimization to find `alpha` with
        :class:`~sklearn.linear_model.LassoLarsIC` relies on the AIC or BIC
        criteria that are computed in-sample, thus on the training set directly.
        This approach differs from the cross-validation procedure. For a comparison
        of the two approaches, you can refer to the following example:
        `sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py`.</p></div>

    .. rubric:: References

    .. [ZHT2007] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
        "On the degrees of freedom of the lasso."
        The Annals of Statistics 35.5 (2007): 2173-2192.
        <0712.0881>`

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
    We will use the diabetes dataset.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    n_samples = X.shape[0]
    X.head()
    return X, n_samples, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Scikit-learn provides an estimator called
    :class:`~sklearn.linear_model.LassoLarsIC` that uses either Akaike's
    information criterion (AIC) or the Bayesian information criterion (BIC) to
    select the best model. Before fitting
    this model, we will scale the dataset.

    In the following, we are going to fit two models to compare the values
    reported by AIC and BIC.


    """
    )
    return


@app.cell
def _(X, y):
    from sklearn.linear_model import LassoLarsIC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic")).fit(X, y)
    return (lasso_lars_ic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To be in line with the definition in [ZHT2007]_, we need to rescale the
    AIC and the BIC. Indeed, Zou et al. are ignoring some constant terms
    compared to the original definition of AIC derived from the maximum
    log-likelihood of a linear model. You can refer to
    `mathematical detail section for the User Guide <lasso_lars_ic>`.


    """
    )
    return


@app.cell
def _(np):
    def zou_et_al_criterion_rescaling(criterion, n_samples, noise_variance):
        """Rescale the information criterion to follow the definition of Zou et al."""
        return criterion - n_samples * np.log(2 * np.pi * noise_variance) - n_samples
    return (zou_et_al_criterion_rescaling,)


@app.cell
def _(lasso_lars_ic, n_samples, zou_et_al_criterion_rescaling):
    import numpy as np

    aic_criterion = zou_et_al_criterion_rescaling(
        lasso_lars_ic[-1].criterion_,
        n_samples,
        lasso_lars_ic[-1].noise_variance_,
    )

    index_alpha_path_aic = np.flatnonzero(
        lasso_lars_ic[-1].alphas_ == lasso_lars_ic[-1].alpha_
    )[0]
    return aic_criterion, index_alpha_path_aic, np


@app.cell
def _(X, lasso_lars_ic, n_samples, np, y, zou_et_al_criterion_rescaling):
    lasso_lars_ic.set_params(lassolarsic__criterion="bic").fit(X, y)

    bic_criterion = zou_et_al_criterion_rescaling(
        lasso_lars_ic[-1].criterion_,
        n_samples,
        lasso_lars_ic[-1].noise_variance_,
    )

    index_alpha_path_bic = np.flatnonzero(
        lasso_lars_ic[-1].alphas_ == lasso_lars_ic[-1].alpha_
    )[0]
    return bic_criterion, index_alpha_path_bic


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now that we collected the AIC and BIC, we can as well check that the minima
    of both criteria happen at the same alpha. Then, we can simplify the
    following plot.


    """
    )
    return


@app.cell
def _(index_alpha_path_aic, index_alpha_path_bic):
    index_alpha_path_aic == index_alpha_path_bic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Finally, we can plot the AIC and BIC criterion and the subsequent selected
    regularization parameter.


    """
    )
    return


@app.cell
def _(aic_criterion, bic_criterion, index_alpha_path_bic):
    import matplotlib.pyplot as plt

    plt.plot(aic_criterion, color="tab:blue", marker="o", label="AIC criterion")
    plt.plot(bic_criterion, color="tab:orange", marker="o", label="BIC criterion")
    plt.vlines(
        index_alpha_path_bic,
        aic_criterion.min(),
        aic_criterion.max(),
        color="black",
        linestyle="--",
        label="Selected alpha",
    )
    plt.legend()
    plt.ylabel("Information criterion")
    plt.xlabel("Lasso model sequence")
    _ = plt.title("Lasso model selection via AIC and BIC")
    return

if __name__ == "__main__":
    app.run()
