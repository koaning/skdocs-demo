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

    # Comparison of kernel ridge and Gaussian process regression

    This example illustrates differences between a kernel ridge regression and a
    Gaussian process regression.

    Both kernel ridge regression and Gaussian process regression are using a
    so-called "kernel trick" to make their models expressive enough to fit
    the training data. However, the machine learning problems solved by the two
    methods are drastically different.

    Kernel ridge regression will find the target function that minimizes a loss
    function (the mean squared error).

    Instead of finding a single target function, the Gaussian process regression
    employs a probabilistic approach : a Gaussian posterior distribution over
    target functions is defined based on the Bayes' theorem, Thus prior
    probabilities on target functions are being combined with a likelihood function
    defined by the observed training data to provide estimates of the posterior
    distributions.

    We will illustrate these differences with an example and we will also focus on
    tuning the kernel hyperparameters.

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
    ## Generating a dataset

    We create a synthetic dataset. The true generative process will take a 1-D
    vector and compute its sine. Note that the period of this sine is thus
    $2 \pi$. We will reuse this information later in this example.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    rng = np.random.RandomState(0)
    data = np.linspace(0, 30, num=1_000).reshape(-1, 1)
    target = np.sin(data).ravel()
    return data, np, rng, target


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, we can imagine a scenario where we get observations from this true
    process. However, we will add some challenges:

    - the measurements will be noisy;
    - only samples from the beginning of the signal will be available.


    """
    )
    return


@app.cell
def _(data, np, rng, target):
    training_sample_indices = rng.choice(np.arange(0, 400), size=40, replace=False)
    training_data = data[training_sample_indices]
    training_noisy_target = target[training_sample_indices] + 0.5 * rng.randn(
        len(training_sample_indices)
    )
    return training_data, training_noisy_target


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's plot the true signal and the noisy measurements available for training.


    """
    )
    return


@app.cell
def _(data, target, training_data, training_noisy_target):
    import matplotlib.pyplot as plt

    plt.plot(data, target, label="True signal", linewidth=2)
    plt.scatter(
        training_data,
        training_noisy_target,
        color="black",
        label="Noisy measurements",
    )
    plt.legend()
    plt.xlabel("data")
    plt.ylabel("target")
    _ = plt.title(
        "Illustration of the true generative process and \n"
        "noisy measurements available during training"
    )
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Limitations of a simple linear model

    First, we would like to highlight the limitations of a linear model given
    our dataset. We fit a :class:`~sklearn.linear_model.Ridge` and check the
    predictions of this model on our dataset.


    """
    )
    return


@app.cell
def _(data, plt, target, training_data, training_noisy_target):
    from sklearn.linear_model import Ridge

    ridge = Ridge().fit(training_data, training_noisy_target)

    plt.plot(data, target, label="True signal", linewidth=2)
    plt.scatter(
        training_data,
        training_noisy_target,
        color="black",
        label="Noisy measurements",
    )
    plt.plot(data, ridge.predict(data), label="Ridge regression")
    plt.legend()
    plt.xlabel("data")
    plt.ylabel("target")
    _ = plt.title("Limitation of a linear model such as ridge")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Such a ridge regressor underfits data since it is not expressive enough.

    ## Kernel methods: kernel ridge and Gaussian process

    ### Kernel ridge

    We can make the previous linear model more expressive by using a so-called
    kernel. A kernel is an embedding from the original feature space to another
    one. Simply put, it is used to map our original data into a newer and more
    complex feature space. This new space is explicitly defined by the choice of
    kernel.

    In our case, we know that the true generative process is a periodic function.
    We can use a :class:`~sklearn.gaussian_process.kernels.ExpSineSquared` kernel
    which allows recovering the periodicity. The class
    :class:`~sklearn.kernel_ridge.KernelRidge` will accept such a kernel.

    Using this model together with a kernel is equivalent to embed the data
    using the mapping function of the kernel and then apply a ridge regression.
    In practice, the data are not mapped explicitly; instead the dot product
    between samples in the higher dimensional feature space is computed using the
    "kernel trick".

    Thus, let's use such a :class:`~sklearn.kernel_ridge.KernelRidge`.


    """
    )
    return


@app.cell
def _(training_data, training_noisy_target):
    import time
    from sklearn.gaussian_process.kernels import ExpSineSquared
    from sklearn.kernel_ridge import KernelRidge
    kernel_ridge = KernelRidge(kernel=ExpSineSquared())
    _start_time = time.time()
    kernel_ridge.fit(training_data, training_noisy_target)
    print(f'Fitting KernelRidge with default kernel: {time.time() - _start_time:.3f} seconds')
    return ExpSineSquared, kernel_ridge, time


@app.cell
def _(data, kernel_ridge, plt, target, training_data, training_noisy_target):
    plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
    plt.scatter(
        training_data,
        training_noisy_target,
        color="black",
        label="Noisy measurements",
    )
    plt.plot(
        data,
        kernel_ridge.predict(data),
        label="Kernel ridge",
        linewidth=2,
        linestyle="dashdot",
    )
    plt.legend(loc="lower right")
    plt.xlabel("data")
    plt.ylabel("target")
    _ = plt.title(
        "Kernel ridge regression with an exponential sine squared\n "
        "kernel using default hyperparameters"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This fitted model is not accurate. Indeed, we did not set the parameters of
    the kernel and instead used the default ones. We can inspect them.


    """
    )
    return


@app.cell
def _(kernel_ridge):
    kernel_ridge.kernel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Our kernel has two parameters: the length-scale and the periodicity. For our
    dataset, we use `sin` as the generative process, implying a
    $2 \pi$-periodicity for the signal. The default value of the parameter
    being $1$, it explains the high frequency observed in the predictions of
    our model.
    Similar conclusions could be drawn with the length-scale parameter. Thus, it
    tells us that the kernel parameters need to be tuned. We will use a randomized
    search to tune the different parameters the kernel ridge model: the `alpha`
    parameter and the kernel parameters.


    """
    )
    return


@app.cell
def _(kernel_ridge, time, training_data, training_noisy_target):
    from scipy.stats import loguniform
    from sklearn.model_selection import RandomizedSearchCV
    param_distributions = {'alpha': loguniform(1.0, 1000.0), 'kernel__length_scale': loguniform(0.01, 100.0), 'kernel__periodicity': loguniform(1.0, 10.0)}
    kernel_ridge_tuned = RandomizedSearchCV(kernel_ridge, param_distributions=param_distributions, n_iter=500, random_state=0)
    _start_time = time.time()
    kernel_ridge_tuned.fit(training_data, training_noisy_target)
    print(f'Time for KernelRidge fitting: {time.time() - _start_time:.3f} seconds')
    return (kernel_ridge_tuned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fitting the model is now more computationally expensive since we have to try
    several combinations of hyperparameters. We can have a look at the
    hyperparameters found to get some intuitions.


    """
    )
    return


@app.cell
def _(kernel_ridge_tuned):
    kernel_ridge_tuned.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Looking at the best parameters, we see that they are different from the
    defaults. We also see that the periodicity is closer to the expected value:
    $2 \pi$. We can now inspect the predictions of our tuned kernel ridge.


    """
    )
    return


@app.cell
def _(data, kernel_ridge_tuned, time):
    _start_time = time.time()
    predictions_kr = kernel_ridge_tuned.predict(data)
    print(f'Time for KernelRidge predict: {time.time() - _start_time:.3f} seconds')
    return (predictions_kr,)


@app.cell
def _(data, plt, predictions_kr, target, training_data, training_noisy_target):
    plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
    plt.scatter(
        training_data,
        training_noisy_target,
        color="black",
        label="Noisy measurements",
    )
    plt.plot(
        data,
        predictions_kr,
        label="Kernel ridge",
        linewidth=2,
        linestyle="dashdot",
    )
    plt.legend(loc="lower right")
    plt.xlabel("data")
    plt.ylabel("target")
    _ = plt.title(
        "Kernel ridge regression with an exponential sine squared\n "
        "kernel using tuned hyperparameters"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We get a much more accurate model. We still observe some errors mainly due to
    the noise added to the dataset.

    ### Gaussian process regression

    Now, we will use a
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor` to fit the same
    dataset. When training a Gaussian process, the hyperparameters of the kernel
    are optimized during the fitting process. There is no need for an external
    hyperparameter search. Here, we create a slightly more complex kernel than
    for the kernel ridge regressor: we add a
    :class:`~sklearn.gaussian_process.kernels.WhiteKernel` that is used to
    estimate the noise in the dataset.


    """
    )
    return


@app.cell
def _(ExpSineSquared, time, training_data, training_noisy_target):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import WhiteKernel
    _kernel = 1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(0.01, 10.0)) + WhiteKernel(0.1)
    gaussian_process = GaussianProcessRegressor(kernel=_kernel)
    _start_time = time.time()
    gaussian_process.fit(training_data, training_noisy_target)
    print(f'Time for GaussianProcessRegressor fitting: {time.time() - _start_time:.3f} seconds')
    return GaussianProcessRegressor, WhiteKernel, gaussian_process


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The computation cost of training a Gaussian process is much less than the
    kernel ridge that uses a randomized search. We can check the parameters of
    the kernels that we computed.


    """
    )
    return


@app.cell
def _(gaussian_process):
    gaussian_process.kernel_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Indeed, we see that the parameters have been optimized. Looking at the
    `periodicity` parameter, we see that we found a period close to the
    theoretical value $2 \pi$. We can have a look now at the predictions of
    our model.


    """
    )
    return


@app.cell
def _(data, gaussian_process, time):
    _start_time = time.time()
    mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(data, return_std=True)
    print(f'Time for GaussianProcessRegressor predict: {time.time() - _start_time:.3f} seconds')
    return mean_predictions_gpr, std_predictions_gpr


@app.cell
def _(
    data,
    mean_predictions_gpr,
    plt,
    predictions_kr,
    std_predictions_gpr,
    target,
    training_data,
    training_noisy_target,
):
    plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
    plt.scatter(
        training_data,
        training_noisy_target,
        color="black",
        label="Noisy measurements",
    )
    # Plot the predictions of the kernel ridge
    plt.plot(
        data,
        predictions_kr,
        label="Kernel ridge",
        linewidth=2,
        linestyle="dashdot",
    )
    # Plot the predictions of the gaussian process regressor
    plt.plot(
        data,
        mean_predictions_gpr,
        label="Gaussian process regressor",
        linewidth=2,
        linestyle="dotted",
    )
    plt.fill_between(
        data.ravel(),
        mean_predictions_gpr - std_predictions_gpr,
        mean_predictions_gpr + std_predictions_gpr,
        color="tab:green",
        alpha=0.2,
    )
    plt.legend(loc="lower right")
    plt.xlabel("data")
    plt.ylabel("target")
    _ = plt.title("Comparison between kernel ridge and gaussian process regressor")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We observe that the results of the kernel ridge and the Gaussian process
    regressor are close. However, the Gaussian process regressor also provide
    an uncertainty information that is not available with a kernel ridge.
    Due to the probabilistic formulation of the target functions, the
    Gaussian process can output the standard deviation (or the covariance)
    together with the mean predictions of the target functions.

    However, it comes at a cost: the time to compute the predictions is higher
    with a Gaussian process.

    ## Final conclusion

    We can give a final word regarding the possibility of the two models to
    extrapolate. Indeed, we only provided the beginning of the signal as a
    training set. Using a periodic kernel forces our model to repeat the pattern
    found on the training set. Using this kernel information together with the
    capacity of the both models to extrapolate, we observe that the models will
    continue to predict the sine pattern.

    Gaussian process allows to combine kernels together. Thus, we could associate
    the exponential sine squared kernel together with a radial basis function
    kernel.


    """
    )
    return


@app.cell
def _(
    ExpSineSquared,
    GaussianProcessRegressor,
    WhiteKernel,
    data,
    training_data,
    training_noisy_target,
):
    from sklearn.gaussian_process.kernels import RBF
    _kernel = 1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(0.01, 10.0)) * RBF(length_scale=15, length_scale_bounds='fixed') + WhiteKernel(0.1)
    gaussian_process_1 = GaussianProcessRegressor(kernel=_kernel)
    gaussian_process_1.fit(training_data, training_noisy_target)
    mean_predictions_gpr_1, std_predictions_gpr_1 = gaussian_process_1.predict(data, return_std=True)
    return mean_predictions_gpr_1, std_predictions_gpr_1


@app.cell
def _(
    data,
    mean_predictions_gpr_1,
    plt,
    predictions_kr,
    std_predictions_gpr_1,
    target,
    training_data,
    training_noisy_target,
):
    plt.plot(data, target, label='True signal', linewidth=2, linestyle='dashed')
    plt.scatter(training_data, training_noisy_target, color='black', label='Noisy measurements')
    plt.plot(data, predictions_kr, label='Kernel ridge', linewidth=2, linestyle='dashdot')
    plt.plot(data, mean_predictions_gpr_1, label='Gaussian process regressor', linewidth=2, linestyle='dotted')
    plt.fill_between(data.ravel(), mean_predictions_gpr_1 - std_predictions_gpr_1, mean_predictions_gpr_1 + std_predictions_gpr_1, color='tab:green', alpha=0.2)
    plt.legend(loc='lower right')
    plt.xlabel('data')
    # Plot the predictions of the kernel ridge
    plt.ylabel('target')
    # Plot the predictions of the gaussian process regressor
    _ = plt.title('Effect of using a radial basis function kernel')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The effect of using a radial basis function kernel will attenuate the
    periodicity effect once that no sample are available in the training.
    As testing samples get further away from the training ones, predictions
    are converging towards their mean and their standard deviation
    also increases.


    """
    )
    return

if __name__ == "__main__":
    app.run()
