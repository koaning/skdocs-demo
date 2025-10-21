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

    # One-Class SVM versus One-Class SVM using Stochastic Gradient Descent

    This example shows how to approximate the solution of
    :class:`sklearn.svm.OneClassSVM` in the case of an RBF kernel with
    :class:`sklearn.linear_model.SGDOneClassSVM`, a Stochastic Gradient Descent
    (SGD) version of the One-Class SVM. A kernel approximation is first used in
    order to apply :class:`sklearn.linear_model.SGDOneClassSVM` which implements a
    linear One-Class SVM using SGD.

    Note that :class:`sklearn.linear_model.SGDOneClassSVM` scales linearly with
    the number of samples whereas the complexity of a kernelized
    :class:`sklearn.svm.OneClassSVM` is at best quadratic with respect to the
    number of samples. It is not the purpose of this example to illustrate the
    benefits of such an approximation in terms of computation time but rather to
    show that we obtain similar results on a toy dataset.

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
    import matplotlib
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import SGDOneClassSVM
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import OneClassSVM
    font = {'weight': 'normal', 'size': 15}
    matplotlib.rc('font', **font)
    random_state = 42
    rng = np.random.RandomState(random_state)
    _X = 0.3 * rng.randn(500, 2)
    X_train = np.r_[_X + 2, _X - 2]
    _X = 0.3 * rng.randn(20, 2)
    X_test = np.r_[_X + 2, _X - 2]
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    # Generate train data
    nu = 0.05
    gamma = 2.0
    # Generate some regular novel observations
    clf = OneClassSVM(gamma=gamma, kernel='rbf', nu=nu)
    clf.fit(X_train)
    # Generate some abnormal novel observations
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    # OCSVM hyperparameters
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    # Fit the One-Class SVM
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    transform = Nystroem(gamma=gamma, random_state=random_state)
    clf_sgd = SGDOneClassSVM(nu=nu, shuffle=True, fit_intercept=True, random_state=random_state, tol=0.0001)
    pipe_sgd = make_pipeline(transform, clf_sgd)
    pipe_sgd.fit(X_train)
    y_pred_train_sgd = pipe_sgd.predict(X_train)
    y_pred_test_sgd = pipe_sgd.predict(X_test)
    y_pred_outliers_sgd = pipe_sgd.predict(X_outliers)
    n_error_train_sgd = y_pred_train_sgd[y_pred_train_sgd == -1].size
    # Fit the One-Class SVM using a kernel approximation and SGD
    n_error_test_sgd = y_pred_test_sgd[y_pred_test_sgd == -1].size
    n_error_outliers_sgd = y_pred_outliers_sgd[y_pred_outliers_sgd == 1].size
    return (
        X_outliers,
        X_test,
        X_train,
        clf,
        mlines,
        n_error_outliers,
        n_error_outliers_sgd,
        n_error_test,
        n_error_test_sgd,
        n_error_train,
        n_error_train_sgd,
        np,
        pipe_sgd,
        plt,
    )


@app.cell
def _(
    X_outliers,
    X_test,
    X_train,
    clf,
    mlines,
    n_error_outliers,
    n_error_test,
    n_error_train,
    np,
    plt,
):
    from sklearn.inspection import DecisionBoundaryDisplay
    _, _ax = plt.subplots(figsize=(9, 6))
    _xx, _yy = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
    _X = np.concatenate([_xx.ravel().reshape(-1, 1), _yy.ravel().reshape(-1, 1)], axis=1)
    DecisionBoundaryDisplay.from_estimator(clf, _X, response_method='decision_function', plot_method='contourf', ax=_ax, cmap='PuBu')
    DecisionBoundaryDisplay.from_estimator(clf, _X, response_method='decision_function', plot_method='contour', ax=_ax, linewidths=2, colors='darkred', levels=[0])
    DecisionBoundaryDisplay.from_estimator(clf, _X, response_method='decision_function', plot_method='contourf', ax=_ax, colors='palevioletred', levels=[0, clf.decision_function(_X).max()])
    _s = 20
    _b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=_s, edgecolors='k')
    _b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=_s, edgecolors='k')
    _c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=_s, edgecolors='k')
    _ax.set(title='One-Class SVM', xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), xlabel=f'error train: {n_error_train}/{X_train.shape[0]}; errors novel regular: {n_error_test}/{X_test.shape[0]}; errors novel abnormal: {n_error_outliers}/{X_outliers.shape[0]}')
    _ = _ax.legend([mlines.Line2D([], [], color='darkred', label='learned frontier'), _b1, _b2, _c], ['learned frontier', 'training observations', 'new regular observations', 'new abnormal observations'], loc='upper left')
    return (DecisionBoundaryDisplay,)


@app.cell
def _(
    DecisionBoundaryDisplay,
    X_outliers,
    X_test,
    X_train,
    mlines,
    n_error_outliers_sgd,
    n_error_test_sgd,
    n_error_train_sgd,
    np,
    pipe_sgd,
    plt,
):
    _, _ax = plt.subplots(figsize=(9, 6))
    _xx, _yy = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
    _X = np.concatenate([_xx.ravel().reshape(-1, 1), _yy.ravel().reshape(-1, 1)], axis=1)
    DecisionBoundaryDisplay.from_estimator(pipe_sgd, _X, response_method='decision_function', plot_method='contourf', ax=_ax, cmap='PuBu')
    DecisionBoundaryDisplay.from_estimator(pipe_sgd, _X, response_method='decision_function', plot_method='contour', ax=_ax, linewidths=2, colors='darkred', levels=[0])
    DecisionBoundaryDisplay.from_estimator(pipe_sgd, _X, response_method='decision_function', plot_method='contourf', ax=_ax, colors='palevioletred', levels=[0, pipe_sgd.decision_function(_X).max()])
    _s = 20
    _b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=_s, edgecolors='k')
    _b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=_s, edgecolors='k')
    _c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=_s, edgecolors='k')
    _ax.set(title='Online One-Class SVM', xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), xlabel=f'error train: {n_error_train_sgd}/{X_train.shape[0]}; errors novel regular: {n_error_test_sgd}/{X_test.shape[0]}; errors novel abnormal: {n_error_outliers_sgd}/{X_outliers.shape[0]}')
    _ax.legend([mlines.Line2D([], [], color='darkred', label='learned frontier'), _b1, _b2, _c], ['learned frontier', 'training observations', 'new regular observations', 'new abnormal observations'], loc='upper left')
    plt.show()
    return

if __name__ == "__main__":
    app.run()
