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

    # Examples of Using `FrozenEstimator`

    This example showcases some use cases of :class:`~sklearn.frozen.FrozenEstimator`.

    :class:`~sklearn.frozen.FrozenEstimator` is a utility class that allows to freeze a
    fitted estimator. This is useful, for instance, when we want to pass a fitted estimator
    to a meta-estimator, such as :class:`~sklearn.model_selection.FixedThresholdClassifier`
    without letting the meta-estimator refit the estimator.

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
    ## Setting a decision threshold for a pre-fitted classifier
    Fitted classifiers in scikit-learn use an arbitrary decision threshold to decide
    which class the given sample belongs to. The decision threshold is either `0.0` on the
    value returned by :term:`decision_function`, or `0.5` on the probability returned by
    :term:`predict_proba`.

    However, one might want to set a custom decision threshold. We can do this by
    using :class:`~sklearn.model_selection.FixedThresholdClassifier` and wrapping the
    classifier with :class:`~sklearn.frozen.FrozenEstimator`.


    """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_classification
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import FixedThresholdClassifier, train_test_split

    X, y = make_classification(n_samples=1000, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classifier = LogisticRegression().fit(X_train, y_train)

    print(
        "Probability estimates for three data points:\n"
        f"{classifier.predict_proba(X_test[-3:]).round(3)}"
    )
    print(
        "Predicted class for the same three data points:\n"
        f"{classifier.predict(X_test[-3:])}"
    )
    return (
        FixedThresholdClassifier,
        FrozenEstimator,
        X_test,
        X_train,
        classifier,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now imagine you'd want to set a different decision threshold on the probability
    estimates. We can do this by wrapping the classifier with
    :class:`~sklearn.frozen.FrozenEstimator` and passing it to
    :class:`~sklearn.model_selection.FixedThresholdClassifier`.


    """
    )
    return


@app.cell
def _(FixedThresholdClassifier, FrozenEstimator, classifier):
    threshold_classifier = FixedThresholdClassifier(
        estimator=FrozenEstimator(classifier), threshold=0.9
    )
    return (threshold_classifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that in the above piece of code, calling `fit` on
    :class:`~sklearn.model_selection.FixedThresholdClassifier` does not refit the
    underlying classifier.

    Now, let's see how the predictions changed with respect to the probability
    threshold.


    """
    )
    return


@app.cell
def _(X_test, threshold_classifier):
    print(
        "Probability estimates for three data points with FixedThresholdClassifier:\n"
        f"{threshold_classifier.predict_proba(X_test[-3:]).round(3)}"
    )
    print(
        "Predicted class for the same three data points with FixedThresholdClassifier:\n"
        f"{threshold_classifier.predict(X_test[-3:])}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We see that the probability estimates stay the same, but since a different decision
    threshold is used, the predicted classes are different.

    Please refer to
    `sphx_glr_auto_examples_model_selection_plot_cost_sensitive_learning.py`
    to learn about cost-sensitive learning and decision threshold tuning.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calibration of a pre-fitted classifier
    You can use :class:`~sklearn.frozen.FrozenEstimator` to calibrate a pre-fitted
    classifier using :class:`~sklearn.calibration.CalibratedClassifierCV`.


    """
    )
    return


@app.cell
def _(FrozenEstimator, X_test, X_train, classifier, y_test, y_train):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    calibrated_classifier = CalibratedClassifierCV(
        estimator=FrozenEstimator(classifier)
    ).fit(X_train, y_train)

    prob_pos_clf = classifier.predict_proba(X_test)[:, 1]
    clf_score = brier_score_loss(y_test, prob_pos_clf)
    print(f"No calibration: {clf_score:.3f}")

    prob_pos_calibrated = calibrated_classifier.predict_proba(X_test)[:, 1]
    calibrated_score = brier_score_loss(y_test, prob_pos_calibrated)
    print(f"With calibration: {calibrated_score:.3f}")
    return

if __name__ == "__main__":
    app.run()
