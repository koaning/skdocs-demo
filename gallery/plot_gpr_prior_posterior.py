import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Illustration of prior and posterior Gaussian process for different kernels

    This example illustrates the prior and posterior of a
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor` with different
    kernels. Mean, standard deviation, and 5 samples are shown for both prior
    and posterior distributions.

    Here, we only give some illustration. To know more about kernels' formulation,
    refer to the `User Guide <gp_kernels>`.

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
    ## Helper function

    Before presenting each individual kernel available for Gaussian processes,
    we will define an helper function allowing us plotting samples drawn from
    the Gaussian process.

    This function will take a
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model and will
    drawn sample from the Gaussian process. If the model was not fit, the samples
    are drawn from the prior distribution while after model fitting, the samples are
    drawn from the posterior distribution.


    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np


    def plot_gpr_samples(gpr_model, n_samples, ax):
        """Plot samples drawn from the Gaussian process model.

        If the Gaussian process model is not trained then the drawn samples are
        drawn from the prior distribution. Otherwise, the samples are drawn from
        the posterior distribution. Be aware that a sample here corresponds to a
        function.

        Parameters
        ----------
        gpr_model : `GaussianProcessRegressor`
            A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
        n_samples : int
            The number of samples to draw from the Gaussian process distribution.
        ax : matplotlib axis
            The matplotlib axis where to plot the samples.
        """
        x = np.linspace(0, 5, 100)
        X = x.reshape(-1, 1)

        y_mean, y_std = gpr_model.predict(X, return_std=True)
        y_samples = gpr_model.sample_y(X, n_samples)

        for idx, single_prior in enumerate(y_samples.T):
            ax.plot(
                x,
                single_prior,
                linestyle="--",
                alpha=0.7,
                label=f"Sampled function #{idx + 1}",
            )
        ax.plot(x, y_mean, color="black", label="Mean")
        ax.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.1,
            color="black",
            label=r"$\pm$ 1 std. dev.",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim([-3, 3])
    return np, plot_gpr_samples, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dataset and Gaussian process generation
    We will create a training dataset that we will use in the different sections.


    """
    )
    return


@app.cell
def _(np):
    rng = np.random.RandomState(4)
    X_train = rng.uniform(0, 5, 10).reshape(-1, 1)
    y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
    n_samples = 5
    return X_train, n_samples, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Kernel cookbook

    In this section, we illustrate some samples drawn from the prior and posterior
    distributions of the Gaussian process with different kernels.

    ### Radial Basis Function kernel


    """
    )
    return


@app.cell
def _(X_train, n_samples, plot_gpr_samples, plt, y_train):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    _fig, _axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
    plot_gpr_samples(gpr, n_samples=n_samples, ax=_axs[0])
    _axs[0].set_title('Samples from prior distribution')
    gpr.fit(X_train, y_train)
    # plot prior
    plot_gpr_samples(gpr, n_samples=n_samples, ax=_axs[1])
    _axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
    _axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
    # plot posterior
    _axs[1].set_title('Samples from posterior distribution')
    _fig.suptitle('Radial Basis Function kernel', fontsize=18)
    plt.tight_layout()
    return GaussianProcessRegressor, gpr, kernel


@app.cell
def _(gpr, kernel):
    print(f"Kernel parameters before fit:\n{kernel})")
    print(
        f"Kernel parameters after fit: \n{gpr.kernel_} \n"
        f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Rational Quadratic kernel


    """
    )
    return


@app.cell
def _(
    GaussianProcessRegressor,
    X_train,
    n_samples,
    plot_gpr_samples,
    plt,
    y_train,
):
    from sklearn.gaussian_process.kernels import RationalQuadratic
    kernel_1 = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-05, 1000000000000000.0))
    gpr_1 = GaussianProcessRegressor(kernel=kernel_1, random_state=0)
    _fig, _axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
    plot_gpr_samples(gpr_1, n_samples=n_samples, ax=_axs[0])
    _axs[0].set_title('Samples from prior distribution')
    gpr_1.fit(X_train, y_train)
    plot_gpr_samples(gpr_1, n_samples=n_samples, ax=_axs[1])
    _axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
    _axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
    _axs[1].set_title('Samples from posterior distribution')
    _fig.suptitle('Rational Quadratic kernel', fontsize=18)
    plt.tight_layout()
    return gpr_1, kernel_1


@app.cell
def _(gpr_1, kernel_1):
    print(f'Kernel parameters before fit:\n{kernel_1})')
    print(f'Kernel parameters after fit: \n{gpr_1.kernel_} \nLog-likelihood: {gpr_1.log_marginal_likelihood(gpr_1.kernel_.theta):.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Exp-Sine-Squared kernel


    """
    )
    return


@app.cell
def _(
    GaussianProcessRegressor,
    X_train,
    n_samples,
    plot_gpr_samples,
    plt,
    y_train,
):
    from sklearn.gaussian_process.kernels import ExpSineSquared
    kernel_2 = 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0))
    gpr_2 = GaussianProcessRegressor(kernel=kernel_2, random_state=0)
    _fig, _axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
    plot_gpr_samples(gpr_2, n_samples=n_samples, ax=_axs[0])
    _axs[0].set_title('Samples from prior distribution')
    gpr_2.fit(X_train, y_train)
    plot_gpr_samples(gpr_2, n_samples=n_samples, ax=_axs[1])
    _axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
    _axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
    _axs[1].set_title('Samples from posterior distribution')
    _fig.suptitle('Exp-Sine-Squared kernel', fontsize=18)
    plt.tight_layout()
    return gpr_2, kernel_2


@app.cell
def _(gpr_2, kernel_2):
    print(f'Kernel parameters before fit:\n{kernel_2})')
    print(f'Kernel parameters after fit: \n{gpr_2.kernel_} \nLog-likelihood: {gpr_2.log_marginal_likelihood(gpr_2.kernel_.theta):.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Dot-product kernel


    """
    )
    return


@app.cell
def _(
    GaussianProcessRegressor,
    X_train,
    n_samples,
    plot_gpr_samples,
    plt,
    y_train,
):
    from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct
    kernel_3 = ConstantKernel(0.1, (0.01, 10.0)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2
    gpr_3 = GaussianProcessRegressor(kernel=kernel_3, random_state=0, normalize_y=True)
    _fig, _axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
    plot_gpr_samples(gpr_3, n_samples=n_samples, ax=_axs[0])
    _axs[0].set_title('Samples from prior distribution')
    gpr_3.fit(X_train, y_train)
    plot_gpr_samples(gpr_3, n_samples=n_samples, ax=_axs[1])
    _axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
    _axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
    _axs[1].set_title('Samples from posterior distribution')
    _fig.suptitle('Dot-product kernel', fontsize=18)
    plt.tight_layout()
    return gpr_3, kernel_3


@app.cell
def _(gpr_3, kernel_3):
    print(f'Kernel parameters before fit:\n{kernel_3})')
    print(f'Kernel parameters after fit: \n{gpr_3.kernel_} \nLog-likelihood: {gpr_3.log_marginal_likelihood(gpr_3.kernel_.theta):.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Matérn kernel


    """
    )
    return


@app.cell
def _(
    GaussianProcessRegressor,
    X_train,
    n_samples,
    plot_gpr_samples,
    plt,
    y_train,
):
    from sklearn.gaussian_process.kernels import Matern
    kernel_4 = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)
    gpr_4 = GaussianProcessRegressor(kernel=kernel_4, random_state=0)
    _fig, _axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
    plot_gpr_samples(gpr_4, n_samples=n_samples, ax=_axs[0])
    _axs[0].set_title('Samples from prior distribution')
    gpr_4.fit(X_train, y_train)
    plot_gpr_samples(gpr_4, n_samples=n_samples, ax=_axs[1])
    _axs[1].scatter(X_train[:, 0], y_train, color='red', zorder=10, label='Observations')
    _axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc='upper left')
    _axs[1].set_title('Samples from posterior distribution')
    _fig.suptitle('Matérn kernel', fontsize=18)
    plt.tight_layout()
    return gpr_4, kernel_4


@app.cell
def _(gpr_4, kernel_4):
    print(f'Kernel parameters before fit:\n{kernel_4})')
    print(f'Kernel parameters after fit: \n{gpr_4.kernel_} \nLog-likelihood: {gpr_4.log_marginal_likelihood(gpr_4.kernel_.theta):.3f}')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
