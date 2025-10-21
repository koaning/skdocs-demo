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

    # Precision-Recall

    Example of Precision-Recall metric to evaluate classifier output quality.

    Precision-Recall is a useful measure of success of prediction when the
    classes are very imbalanced. In information retrieval, precision is a
    measure of the fraction of relevant items among actually returned items while recall
    is a measure of the fraction of items that were returned among all items that should
    have been returned. 'Relevancy' here refers to items that are
    positively labeled, i.e., true positives and false negatives.

    Precision ($P$) is defined as the number of true positives ($T_p$)
    over the number of true positives plus the number of false positives
    ($F_p$).

    \begin{align}P = \frac{T_p}{T_p+F_p}\end{align}

    Recall ($R$) is defined as the number of true positives ($T_p$)
    over the number of true positives plus the number of false negatives
    ($F_n$).

    \begin{align}R = \frac{T_p}{T_p + F_n}\end{align}

    The precision-recall curve shows the tradeoff between precision and
    recall for different thresholds. A high area under the curve represents
    both high recall and high precision. High precision is achieved by having
    few false positives in the returned results, and high recall is achieved by
    having few false negatives in the relevant results.
    High scores for both show that the classifier is returning
    accurate results (high precision), as well as returning a majority of all relevant
    results (high recall).

    A system with high recall but low precision returns most of the relevant items, but
    the proportion of returned results that are incorrectly labeled is high. A
    system with high precision but low recall is just the opposite, returning very
    few of the relevant items, but most of its predicted labels are correct when compared
    to the actual labels. An ideal system with high precision and high recall will
    return most of the relevant items, with most results labeled correctly.

    The definition of precision ($\frac{T_p}{T_p + F_p}$) shows that lowering
    the threshold of a classifier may increase the denominator, by increasing the
    number of results returned. If the threshold was previously set too high, the
    new results may all be true positives, which will increase precision. If the
    previous threshold was about right or too low, further lowering the threshold
    will introduce false positives, decreasing precision.

    Recall is defined as $\frac{T_p}{T_p+F_n}$, where $T_p+F_n$ does
    not depend on the classifier threshold. Changing the classifier threshold can only
    change the numerator, $T_p$. Lowering the classifier
    threshold may increase recall, by increasing the number of true positive
    results. It is also possible that lowering the threshold may leave recall
    unchanged, while the precision fluctuates. Thus, precision does not necessarily
    decrease with recall.

    The relationship between recall and precision can be observed in the
    stairstep area of the plot - at the edges of these steps a small change
    in the threshold considerably reduces precision, with only a minor gain in
    recall.

    **Average precision** (AP) summarizes such a plot as the weighted mean of
    precisions achieved at each threshold, with the increase in recall from the
    previous threshold used as the weight:

    $\text{AP} = \sum_n (R_n - R_{n-1}) P_n$

    where $P_n$ and $R_n$ are the precision and recall at the
    nth threshold. A pair $(R_k, P_k)$ is referred to as an
    *operating point*.

    AP and the trapezoidal area under the operating points
    (:func:`sklearn.metrics.auc`) are common ways to summarize a precision-recall
    curve that lead to different results. Read more in the
    `User Guide <precision_recall_f_measure_metrics>`.

    Precision-recall curves are typically used in binary classification to study
    the output of a classifier. In order to extend the precision-recall curve and
    average precision to multi-class or multi-label classification, it is necessary
    to binarize the output. One curve can be drawn per label, but one can also draw
    a precision-recall curve by considering each element of the label indicator
    matrix as a binary prediction (`micro-averaging <average>`).

    <div class="alert alert-info"><h4>Note</h4><p>See also :func:`sklearn.metrics.average_precision_score`,
                 :func:`sklearn.metrics.recall_score`,
                 :func:`sklearn.metrics.precision_score`,
                 :func:`sklearn.metrics.f1_score`</p></div>

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
    ## In binary classification settings

    ### Dataset and model

    We will use a Linear SVC classifier to differentiate two types of irises.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)

    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)

    # Limit to the two first classes, and split into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X[y < 2], y[y < 2], test_size=0.5, random_state=random_state
    )
    return (
        X,
        X_test,
        X_train,
        np,
        random_state,
        train_test_split,
        y,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Linear SVC will expect each feature to have a similar range of values. Thus,
    we will first scale the data using a
    :class:`~sklearn.preprocessing.StandardScaler`.


    """
    )
    return


@app.cell
def _(X_train, random_state, y_train):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))
    classifier.fit(X_train, y_train)
    return LinearSVC, StandardScaler, classifier, make_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Plot the Precision-Recall curve

    To plot the precision-recall curve, you should use
    :class:`~sklearn.metrics.PrecisionRecallDisplay`. Indeed, there is two
    methods available depending if you already computed the predictions of the
    classifier or not.

    Let's first plot the precision-recall curve without the classifier
    predictions. We use
    :func:`~sklearn.metrics.PrecisionRecallDisplay.from_estimator` that
    computes the predictions for us before plotting the curve.


    """
    )
    return


@app.cell
def _(X_test, classifier, y_test):
    from sklearn.metrics import PrecisionRecallDisplay
    _display = PrecisionRecallDisplay.from_estimator(classifier, X_test, y_test, name='LinearSVC', plot_chance_level=True, despine=True)
    _ = _display.ax_.set_title('2-class Precision-Recall curve')
    return (PrecisionRecallDisplay,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    If we already got the estimated probabilities or scores for
    our model, then we can use
    :func:`~sklearn.metrics.PrecisionRecallDisplay.from_predictions`.


    """
    )
    return


@app.cell
def _(PrecisionRecallDisplay, X_test, classifier, y_test):
    y_score = classifier.decision_function(X_test)
    _display = PrecisionRecallDisplay.from_predictions(y_test, y_score, name='LinearSVC', plot_chance_level=True, despine=True)
    _ = _display.ax_.set_title('2-class Precision-Recall curve')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## In multi-label settings

    The precision-recall curve does not support the multilabel setting. However,
    one can decide how to handle this case. We show such an example below.

    ### Create multi-label data, fit, and predict

    We create a multi-label dataset, to illustrate the precision-recall in
    multi-label settings.


    """
    )
    return


@app.cell
def _(X, random_state, train_test_split, y):
    from sklearn.preprocessing import label_binarize
    Y = label_binarize(y, classes=[0, 1, 2])
    # Use label_binarize to be multi-label like settings
    n_classes = Y.shape[1]
    # Split into training and test
    X_train_1, X_test_1, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=random_state)
    return X_test_1, X_train_1, Y_test, Y_train, n_classes


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We use :class:`~sklearn.multiclass.OneVsRestClassifier` for multi-label
    prediction.


    """
    )
    return


@app.cell
def _(
    LinearSVC,
    StandardScaler,
    X_test_1,
    X_train_1,
    Y_train,
    make_pipeline,
    random_state,
):
    from sklearn.multiclass import OneVsRestClassifier
    classifier_1 = OneVsRestClassifier(make_pipeline(StandardScaler(), LinearSVC(random_state=random_state)))
    classifier_1.fit(X_train_1, Y_train)
    y_score_1 = classifier_1.decision_function(X_test_1)
    return (y_score_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The average precision score in multi-label settings


    """
    )
    return


@app.cell
def _(Y_test, n_classes, y_score_1):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for _i in range(n_classes):
        precision[_i], recall[_i], _ = precision_recall_curve(Y_test[:, _i], y_score_1[:, _i])
        average_precision[_i] = average_precision_score(Y_test[:, _i], y_score_1[:, _i])
    precision['micro'], recall['micro'], _ = precision_recall_curve(Y_test.ravel(), y_score_1.ravel())
    average_precision['micro'] = average_precision_score(Y_test, y_score_1, average='micro')
    return average_precision, precision, recall


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Plot the micro-averaged Precision-Recall curve


    """
    )
    return


@app.cell
def _(PrecisionRecallDisplay, Y_test, average_precision, precision, recall):
    from collections import Counter
    _display = PrecisionRecallDisplay(recall=recall['micro'], precision=precision['micro'], average_precision=average_precision['micro'], prevalence_pos_label=Counter(Y_test.ravel())[1] / Y_test.size)
    _display.plot(plot_chance_level=True, despine=True)
    _ = _display.ax_.set_title('Micro-averaged over all classes')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Plot Precision-Recall curve for each class and iso-f1 curves


    """
    )
    return


@app.cell
def _(
    PrecisionRecallDisplay,
    average_precision,
    n_classes,
    np,
    precision,
    recall,
):
    from itertools import cycle
    import matplotlib.pyplot as plt
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    _, ax = plt.subplots(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = ([], [])
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y_1 = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y_1 >= 0], y_1[y_1 >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y_1[45] + 0.02))
    _display = PrecisionRecallDisplay(recall=recall['micro'], precision=precision['micro'], average_precision=average_precision['micro'])
    _display.plot(ax=ax, name='Micro-average precision-recall', color='gold')
    for _i, color in zip(range(n_classes), colors):
        _display = PrecisionRecallDisplay(recall=recall[_i], precision=precision[_i], average_precision=average_precision[_i])
        _display.plot(ax=ax, name=f'Precision-recall for class {_i}', color=color, despine=True)
    handles, labels = _display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(['iso-f1 curves'])
    ax.legend(handles=handles, labels=labels, loc='best')
    ax.set_title('Extension of Precision-Recall curve to multi-class')
    plt.show()
    return

if __name__ == "__main__":
    app.run()
