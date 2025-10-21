import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Ability of Gaussian process regression (GPR) to estimate data noise-level

    This example shows the ability of the
    :class:`~sklearn.gaussian_process.kernels.WhiteKernel` to estimate the noise
    level in the data. Moreover, we show the importance of kernel hyperparameters
    initialization.

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

    We will work in a setting where `X` will contain a single feature. We create a
    function that will generate the target to be predicted. We will add an
    option to add some noise to the generated target.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    def target_generator(X, add_noise=False):
        target = 0.5 + np.sin(3 * X)
        if add_noise:
            rng = np.random.RandomState(1)
            target = target + rng.normal(0, 0.3, size=target.shape)
        return target.squeeze()
    return np, target_generator


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's have a look to the target generator where we will not add any noise to
    observe the signal that we would like to predict.


    """
    )
    return


@app.cell
def _(np, target_generator):
    X = np.linspace(0, 5, num=80).reshape(-1, 1)
    y = target_generator(X, add_noise=False)
    return X, y


@app.cell
def _(X, y):
    import matplotlib.pyplot as plt

    plt.plot(X, y, label="Expected signal")
    plt.legend()
    plt.xlabel("X")
    _ = plt.ylabel("y")
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The target is transforming the input `X` using a sine function. Now, we will
    generate few noisy training samples. To illustrate the noise level, we will
    plot the true signal together with the noisy training samples.


    """
    )
    return


@app.cell
def _(np, target_generator):
    rng = np.random.RandomState(0)
    X_train = rng.uniform(0, 5, size=20).reshape(-1, 1)
    y_train = target_generator(X_train, add_noise=True)
    return X_train, y_train


@app.cell
def _(X, X_train, plt, y, y_train):
    plt.plot(X, y, label="Expected signal")
    plt.scatter(
        x=X_train[:, 0],
        y=y_train,
        color="black",
        alpha=0.4,
        label="Observations",
    )
    plt.legend()
    plt.xlabel("X")
    _ = plt.ylabel("y")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Optimisation of kernel hyperparameters in GPR

    Now, we will create a
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
    using an additive kernel adding a
    :class:`~sklearn.gaussian_process.kernels.RBF` and
    :class:`~sklearn.gaussian_process.kernels.WhiteKernel` kernels.
    The :class:`~sklearn.gaussian_process.kernels.WhiteKernel` is a kernel that
    will able to estimate the amount of noise present in the data while the
    :class:`~sklearn.gaussian_process.kernels.RBF` will serve at fitting the
    non-linearity between the data and the target.

    However, we will show that the hyperparameter space contains several local
    minima. It will highlights the importance of initial hyperparameter values.

    We will create a model using a kernel with a high noise level and a large
    length scale, which will explain all variations in the data by noise.


    """
    )
    return


@app.cell
def _(X, X_train, y_train):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        noise_level=1, noise_level_bounds=(1e-10, 1e1)
    )
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
    gpr.fit(X_train, y_train)
    y_mean, y_std = gpr.predict(X, return_std=True)
    return (
        GaussianProcessRegressor,
        RBF,
        WhiteKernel,
        gpr,
        kernel,
        y_mean,
        y_std,
    )


@app.cell
def _(X, X_train, gpr, kernel, plt, y, y_mean, y_std, y_train):
    plt.plot(X, y, label="Expected signal")
    plt.scatter(x=X_train[:, 0], y=y_train, color="black", alpha=0.4, label="Observations")
    plt.errorbar(X, y_mean, y_std, label="Posterior mean ± std")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    _ = plt.title(
        (
            f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
            f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}"
        ),
        fontsize=8,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We see that the optimum kernel found still has a high noise level and an even
    larger length scale. The length scale reaches the maximum bound that we
    allowed for this parameter and we got a warning as a result.

    More importantly, we observe that the model does not provide useful
    predictions: the mean prediction seems to be constant: it does not follow the
    expected noise-free signal.

    Now, we will initialize the :class:`~sklearn.gaussian_process.kernels.RBF`
    with a larger `length_scale` initial value and the
    :class:`~sklearn.gaussian_process.kernels.WhiteKernel` with a smaller initial
    noise level lower while keeping the parameter bounds unchanged.


    """
    )
    return


@app.cell
def _(GaussianProcessRegressor, RBF, WhiteKernel, X, X_train, y_train):
    kernel_1 = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(0.01, 1000.0)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-10, 10.0))
    gpr_1 = GaussianProcessRegressor(kernel=kernel_1, alpha=0.0)
    gpr_1.fit(X_train, y_train)
    y_mean_1, y_std_1 = gpr_1.predict(X, return_std=True)
    return gpr_1, kernel_1, y_mean_1, y_std_1


@app.cell
def _(X, X_train, gpr_1, kernel_1, plt, y, y_mean_1, y_std_1, y_train):
    plt.plot(X, y, label='Expected signal')
    plt.scatter(x=X_train[:, 0], y=y_train, color='black', alpha=0.4, label='Observations')
    plt.errorbar(X, y_mean_1, y_std_1, label='Posterior mean ± std')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    _ = plt.title(f'Initial: {kernel_1}\nOptimum: {gpr_1.kernel_}\nLog-Marginal-Likelihood: {gpr_1.log_marginal_likelihood(gpr_1.kernel_.theta)}', fontsize=8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    First, we see that the model's predictions are more precise than the
    previous model's: this new model is able to estimate the noise-free
    functional relationship.

    Looking at the kernel hyperparameters, we see that the best combination found
    has a smaller noise level and shorter length scale than the first model.

    We can inspect the negative Log-Marginal-Likelihood (LML) of
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
    for different hyperparameters to get a sense of the local minima.


    """
    )
    return


@app.cell
def _(gpr_1, np):
    from matplotlib.colors import LogNorm
    length_scale = np.logspace(-2, 4, num=80)
    noise_level = np.logspace(-2, 1, num=80)
    length_scale_grid, noise_level_grid = np.meshgrid(length_scale, noise_level)
    log_marginal_likelihood = [gpr_1.log_marginal_likelihood(theta=np.log([0.36, scale, noise])) for scale, noise in zip(length_scale_grid.ravel(), noise_level_grid.ravel())]
    log_marginal_likelihood = np.reshape(log_marginal_likelihood, noise_level_grid.shape)
    return (
        LogNorm,
        length_scale_grid,
        log_marginal_likelihood,
        noise_level_grid,
    )


@app.cell
def _(
    LogNorm,
    length_scale_grid,
    log_marginal_likelihood,
    noise_level_grid,
    np,
    plt,
):
    vmin, vmax = (-log_marginal_likelihood).min(), 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), num=20), decimals=1)
    plt.contour(
        length_scale_grid,
        noise_level_grid,
        -log_marginal_likelihood,
        levels=level,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise-level")
    plt.title("Negative log-marginal-likelihood")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We see that there are two local minima that correspond to the combination of
    hyperparameters previously found. Depending on the initial values for the
    hyperparameters, the gradient-based optimization might or might not
    converge to the best model. It is thus important to repeat the optimization
    several times for different initializations. This can be done by setting the
    `n_restarts_optimizer` parameter of the
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor` class.

    Let's try again to fit our model with the bad initial values but this time
    with 10 random restarts.


    """
    )
    return


@app.cell
def _(GaussianProcessRegressor, RBF, WhiteKernel, X, X_train, y_train):
    kernel_2 = 1.0 * RBF(length_scale=10.0, length_scale_bounds=(0.01, 1000.0)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 10.0))
    gpr_2 = GaussianProcessRegressor(kernel=kernel_2, alpha=0.0, n_restarts_optimizer=10, random_state=0)
    gpr_2.fit(X_train, y_train)
    y_mean_2, y_std_2 = gpr_2.predict(X, return_std=True)
    return gpr_2, kernel_2, y_mean_2, y_std_2


@app.cell
def _(X, X_train, gpr_2, kernel_2, plt, y, y_mean_2, y_std_2, y_train):
    plt.plot(X, y, label='Expected signal')
    plt.scatter(x=X_train[:, 0], y=y_train, color='black', alpha=0.4, label='Observations')
    plt.errorbar(X, y_mean_2, y_std_2, label='Posterior mean ± std')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    _ = plt.title(f'Initial: {kernel_2}\nOptimum: {gpr_2.kernel_}\nLog-Marginal-Likelihood: {gpr_2.log_marginal_likelihood(gpr_2.kernel_.theta)}', fontsize=8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As we hoped, random restarts allow the optimization to find the best set
    of hyperparameters despite the bad initial values.


    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
