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

    # Recognizing hand-written digits

    This example shows how scikit-learn can be used to recognize images of
    hand-written digits, from 0-9.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    # Standard scientific Python imports
    import matplotlib.pyplot as plt

    # Import datasets, classifiers and performance metrics
    from sklearn import datasets, metrics, svm
    from sklearn.model_selection import train_test_split
    return datasets, metrics, plt, svm, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Digits dataset

    The digits dataset consists of 8x8
    pixel images of digits. The ``images`` attribute of the dataset stores
    8x8 arrays of grayscale values for each image. We will use these arrays to
    visualize the first 4 images. The ``target`` attribute of the dataset stores
    the digit each image represents and this is included in the title of the 4
    plots below.

    Note: if we were working from image files (e.g., 'png' files), we would load
    them using :func:`matplotlib.pyplot.imread`.


    """
    )
    return


@app.cell
def _(datasets, plt):
    digits = datasets.load_digits()
    _, _axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for _ax, _image, label in zip(_axes, digits.images, digits.target):
        _ax.set_axis_off()
        _ax.imshow(_image, cmap=plt.cm.gray_r, interpolation='nearest')
        _ax.set_title('Training: %i' % label)
    return (digits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Classification

    To apply a classifier on this data, we need to flatten the images, turning
    each 2-D array of grayscale values from shape ``(8, 8)`` into shape
    ``(64,)``. Subsequently, the entire dataset will be of shape
    ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
    ``n_features`` is the total number of pixels in each image.

    We can then split the data into train and test subsets and fit a support
    vector classifier on the train samples. The fitted classifier can
    subsequently be used to predict the value of the digit for the samples
    in the test subset.


    """
    )
    return


@app.cell
def _(digits, svm, train_test_split):
    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    return X_test, clf, predicted, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Below we visualize the first 4 test samples and show their predicted
    digit value in the title.


    """
    )
    return


@app.cell
def _(X_test, plt, predicted):
    _, _axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for _ax, _image, prediction in zip(_axes, X_test, predicted):
        _ax.set_axis_off()
        _image = _image.reshape(8, 8)
        _ax.imshow(_image, cmap=plt.cm.gray_r, interpolation='nearest')
        _ax.set_title(f'Prediction: {prediction}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    :func:`~sklearn.metrics.classification_report` builds a text report showing
    the main classification metrics.


    """
    )
    return


@app.cell
def _(clf, metrics, predicted, y_test):
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can also plot a `confusion matrix <confusion_matrix>` of the
    true digit values and the predicted digit values.


    """
    )
    return


@app.cell
def _(metrics, plt, predicted, y_test):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()
    return (disp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    If the results from evaluating a classifier are stored in the form of a
    `confusion matrix <confusion_matrix>` and not in terms of `y_true` and
    `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
    as follows:


    """
    )
    return


@app.cell
def _(disp, metrics):
    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )
    return

if __name__ == "__main__":
    app.run()
