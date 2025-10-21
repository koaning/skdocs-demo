import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Label Propagation digits: Demonstrating performance

    This example demonstrates the power of semisupervised learning by
    training a Label Spreading model to classify handwritten digits
    with sets of very few labels.

    The handwritten digit dataset has 1797 total points. The model will
    be trained using all points, but only 30 will be labeled. Results
    in the form of a confusion matrix and a series of metrics over each
    class will be very good.

    At the end, the top 10 most uncertain predictions will be shown.

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

    We use the digits dataset. We only use a subset of randomly selected samples.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn import datasets

    digits = datasets.load_digits()
    rng = np.random.RandomState(2)
    indices = np.arange(len(digits.data))
    rng.shuffle(indices)
    return digits, indices, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We selected 340 samples of which only 40 will be associated with a known label.
    Therefore, we store the indices of the 300 other samples for which we are not
    supposed to know their labels.


    """
    )
    return


@app.cell
def _(digits, indices, np):
    X = digits.data[indices[:340]]
    y = digits.target[indices[:340]]
    images = digits.images[indices[:340]]
    n_total_samples = len(y)
    n_labeled_points = 40
    indices_1 = np.arange(n_total_samples)
    unlabeled_set = indices_1[n_labeled_points:]
    return X, images, n_labeled_points, n_total_samples, unlabeled_set, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Shuffle everything around


    """
    )
    return


@app.cell
def _(np, unlabeled_set, y):
    y_train = np.copy(y)
    y_train[unlabeled_set] = -1
    return (y_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Semi-supervised learning

    We fit a :class:`~sklearn.semi_supervised.LabelSpreading` and use it to predict
    the unknown labels.


    """
    )
    return


@app.cell
def _(X, n_labeled_points, n_total_samples, unlabeled_set, y, y_train):
    from sklearn.metrics import classification_report
    from sklearn.semi_supervised import LabelSpreading

    lp_model = LabelSpreading(gamma=0.25, max_iter=20)
    lp_model.fit(X, y_train)
    predicted_labels = lp_model.transduction_[unlabeled_set]
    true_labels = y[unlabeled_set]

    print(
        "Label Spreading model: %d labeled & %d unlabeled points (%d total)"
        % (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples)
    )
    return classification_report, lp_model, predicted_labels, true_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Classification report


    """
    )
    return


@app.cell
def _(classification_report, predicted_labels, true_labels):
    print(classification_report(true_labels, predicted_labels))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Confusion matrix


    """
    )
    return


@app.cell
def _(lp_model, predicted_labels, true_labels):
    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(
        true_labels, predicted_labels, labels=lp_model.classes_
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot the most uncertain predictions

    Here, we will pick and show the 10 most uncertain predictions.


    """
    )
    return


@app.cell
def _(lp_model):
    from scipy import stats

    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
    return (pred_entropies,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Pick the top 10 most uncertain labels


    """
    )
    return


@app.cell
def _(np, pred_entropies):
    uncertainty_index = np.argsort(pred_entropies)[-10:]
    return (uncertainty_index,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Plot


    """
    )
    return


@app.cell
def _(images, lp_model, uncertainty_index, y):
    import matplotlib.pyplot as plt

    f = plt.figure(figsize=(7, 5))
    for index, image_index in enumerate(uncertainty_index):
        image = images[image_index]

        sub = f.add_subplot(2, 5, index + 1)
        sub.imshow(image, cmap=plt.cm.gray_r)
        plt.xticks([])
        plt.yticks([])
        sub.set_title(
            "predict: %i\ntrue: %i" % (lp_model.transduction_[image_index], y[image_index])
        )

    f.suptitle("Learning with small amount of labeled data")
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
