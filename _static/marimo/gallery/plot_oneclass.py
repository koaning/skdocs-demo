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

    # One-class SVM with non-linear kernel (RBF)

    An example using a one-class SVM for novelty detection.

    `One-class SVM <svm_outlier_detection>` is an unsupervised
    algorithm that learns a decision function for novelty detection:
    classifying new data as similar or different to the training set.

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
    import numpy as np
    from sklearn import svm
    _X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[_X + 2, _X - 2]
    # Generate train data
    _X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[_X + 2, _X - 2]
    # Generate some regular novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
    # Generate some abnormal novel observations
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    # fit the model
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    return (
        X_outliers,
        X_test,
        X_train,
        clf,
        n_error_outliers,
        n_error_test,
        n_error_train,
        np,
    )


@app.cell
def _(
    X_outliers,
    X_test,
    X_train,
    clf,
    n_error_outliers,
    n_error_test,
    n_error_train,
    np,
):
    import matplotlib.font_manager
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay
    _, ax = plt.subplots()
    xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    _X = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    DecisionBoundaryDisplay.from_estimator(clf, _X, response_method='decision_function', plot_method='contourf', ax=ax, cmap='PuBu')
    # generate grid for the boundary display
    DecisionBoundaryDisplay.from_estimator(clf, _X, response_method='decision_function', plot_method='contourf', ax=ax, levels=[0, 10000], colors='palevioletred')
    DecisionBoundaryDisplay.from_estimator(clf, _X, response_method='decision_function', plot_method='contour', ax=ax, levels=[0], colors='darkred', linewidths=2)
    s = 40
    b1 = ax.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = ax.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
    c = ax.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s, edgecolors='k')
    plt.legend([mlines.Line2D([], [], color='darkred'), b1, b2, c], ['learned frontier', 'training observations', 'new regular observations', 'new abnormal observations'], loc='upper left', prop=matplotlib.font_manager.FontProperties(size=11))
    ax.set(xlabel=f'error train: {n_error_train}/200 ; errors novel regular: {n_error_test}/40 ; errors novel abnormal: {n_error_outliers}/40', title='Novelty Detection', xlim=(-5, 5), ylim=(-5, 5))
    plt.show()
    return

if __name__ == "__main__":
    app.run()
