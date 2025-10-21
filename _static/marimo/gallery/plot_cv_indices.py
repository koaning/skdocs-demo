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

    # Visualizing cross-validation behavior in scikit-learn

    Choosing the right cross-validation object is a crucial part of fitting a
    model properly. There are many ways to split data into training and test
    sets in order to avoid model overfitting, to standardize the number of
    groups in test sets, etc.

    This example visualizes the behavior of several common scikit-learn objects
    for comparison.

    """
    )
    return


@app.cell
def _():
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        KFold,
        ShuffleSplit,
        StratifiedGroupKFold,
        StratifiedKFold,
        StratifiedShuffleSplit,
        TimeSeriesSplit,
    )

    rng = np.random.RandomState(1338)
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm
    n_splits = 4
    return (
        GroupKFold,
        GroupShuffleSplit,
        KFold,
        Patch,
        ShuffleSplit,
        StratifiedGroupKFold,
        StratifiedKFold,
        StratifiedShuffleSplit,
        TimeSeriesSplit,
        cmap_cv,
        cmap_data,
        n_splits,
        np,
        plt,
        rng,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualize our data

    First, we must understand the structure of our data. It has 100 randomly
    generated input datapoints, 3 classes split unevenly across datapoints,
    and 10 "groups" split evenly across datapoints.

    As we'll see, some cross-validation objects do specific things with
    labeled data, others behave differently with grouped data, and others
    do not use this information.

    To begin, we'll visualize our data.


    """
    )
    return


@app.cell
def _(cmap_data, np, plt, rng):
    # Generate the class/group data
    n_points = 100
    X = rng.randn(100, 10)
    percentiles_classes = [0.1, 0.3, 0.6]
    y = np.hstack([[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)])
    group_prior = rng.dirichlet([2] * 10)
    groups = np.repeat(np.arange(10), rng.multinomial(100, group_prior))
    # Generate uneven groups

    def visualize_groups(classes, groups, name):
        _fig, _ax = plt.subplots()
        _ax.scatter(range(len(groups)), [0.5] * len(groups), c=groups, marker='_', lw=50, cmap=cmap_data)
        _ax.scatter(range(len(groups)), [3.5] * len(groups), c=classes, marker='_', lw=50, cmap=cmap_data)
        _ax.set(ylim=[-1, 5], yticks=[0.5, 3.5], yticklabels=['Data\ngroup', 'Data\nclass'], xlabel='Sample index')  # Visualize dataset groups
    visualize_groups(y, groups, 'no groups')
    return X, groups, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define a function to visualize cross-validation behavior

    We'll define a function that lets us visualize the behavior of each
    cross-validation object. We'll perform 4 splits of the data. On each
    split, we'll visualize the indices chosen for the training set
    (in blue) and the test set (in red).


    """
    )
    return


@app.cell
def _(cmap_cv, cmap_data, np):
    def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
        """Create a sample plot for indices of a cross-validation object."""
        use_groups = 'Group' in type(_cv).__name__
        groups = group if use_groups else None
        for ii, (tr, tt) in enumerate(_cv.split(X=X, y=y, groups=groups)):  # Generate the training/testing visualizations for each CV split
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1  # Fill in indices with the training/test groups
            indices[tr] = 0
            _ax.scatter(range(len(indices)), [ii + 0.5] * len(indices), c=indices, marker='_', lw=lw, cmap=cmap_cv, vmin=-0.2, vmax=1.2)
        _ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker='_', lw=lw, cmap=cmap_data)
        _ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker='_', lw=lw, cmap=cmap_data)
        yticklabels = list(range(n_splits)) + ['class', 'group']  # Visualize the results
        _ax.set(yticks=np.arange(n_splits + 2) + 0.5, yticklabels=yticklabels, xlabel='Sample index', ylabel='CV iteration', ylim=[n_splits + 2.2, -0.2], xlim=[0, 100])
        _ax.set_title('{}'.format(type(_cv).__name__), fontsize=15)
        return _ax  # Plot the data classes and groups at the end  # Formatting
    return (plot_cv_indices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's see how it looks for the :class:`~sklearn.model_selection.KFold`
    cross-validation object:


    """
    )
    return


@app.cell
def _(KFold, X, groups, n_splits, plot_cv_indices, plt, y):
    _fig, _ax = plt.subplots()
    _cv = KFold(n_splits)
    plot_cv_indices(_cv, X, y, groups, _ax, n_splits)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see, by default the KFold cross-validation iterator does not
    take either datapoint class or group into consideration. We can change this
    by using either:

    - ``StratifiedKFold`` to preserve the percentage of samples for each class.
    - ``GroupKFold`` to ensure that the same group will not appear in two
      different folds.
    - ``StratifiedGroupKFold`` to keep the constraint of ``GroupKFold`` while
      attempting to return stratified folds.


    """
    )
    return


@app.cell
def _(
    GroupKFold,
    Patch,
    StratifiedGroupKFold,
    StratifiedKFold,
    X,
    cmap_cv,
    groups,
    n_splits,
    plot_cv_indices,
    plt,
    y,
):
    _cvs = [StratifiedKFold, GroupKFold, StratifiedGroupKFold]
    for _cv in _cvs:
        _fig, _ax = plt.subplots(figsize=(6, 3))
        plot_cv_indices(_cv(n_splits), X, y, groups, _ax, n_splits)
        _ax.legend([Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))], ['Testing set', 'Training set'], loc=(1.02, 0.8))
        plt.tight_layout()
        _fig.subplots_adjust(right=0.7)  # Make the legend fit
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Next we'll visualize this behavior for a number of CV iterators.

    ## Visualize cross-validation indices for many CV objects

    Let's visually compare the cross validation behavior for many
    scikit-learn cross-validation objects. Below we will loop through several
    common cross-validation objects, visualizing the behavior of each.

    Note how some use the group/class information while others do not.


    """
    )
    return


@app.cell
def _(
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    Patch,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    X,
    cmap_cv,
    groups,
    n_splits,
    plot_cv_indices,
    plt,
    y,
):
    _cvs = [KFold, GroupKFold, ShuffleSplit, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit]
    for _cv in _cvs:
        this_cv = _cv(n_splits=n_splits)
        _fig, _ax = plt.subplots(figsize=(6, 3))
        plot_cv_indices(this_cv, X, y, groups, _ax, n_splits)
        _ax.legend([Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))], ['Testing set', 'Training set'], loc=(1.02, 0.8))
        plt.tight_layout()
        _fig.subplots_adjust(right=0.7)
    plt.show()  # Make the legend fit
    return

if __name__ == "__main__":
    app.run()
