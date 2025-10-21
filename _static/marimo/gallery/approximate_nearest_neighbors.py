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

    # Approximate nearest neighbors in TSNE

    This example presents how to chain KNeighborsTransformer and TSNE in a pipeline.
    It also shows how to wrap the packages `nmslib` and `pynndescent` to replace
    KNeighborsTransformer and perform approximate nearest neighbors. These packages
    can be installed with `pip install nmslib pynndescent`.

    Note: In KNeighborsTransformer we use the definition which includes each
    training point as its own neighbor in the count of `n_neighbors`, and for
    compatibility reasons, one extra neighbor is computed when `mode == 'distance'`.
    Please note that we do the same in the proposed `nmslib` wrapper.

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
    First we try to import the packages and warn the user in case they are
    missing.


    """
    )
    return


@app.cell
def _():
    import sys

    try:
        import nmslib
    except ImportError:
        print("The package 'nmslib' is required to run this example.")
        sys.exit()

    try:
        from pynndescent import PyNNDescentTransformer
    except ImportError:
        print("The package 'pynndescent' is required to run this example.")
        sys.exit()
    return PyNNDescentTransformer, nmslib


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We define a wrapper class for implementing the scikit-learn API to the
    `nmslib`, as well as a loading function.


    """
    )
    return


@app.cell
def _(nmslib):
    import joblib
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.datasets import fetch_openml
    from sklearn.utils import shuffle

    class NMSlibTransformer(TransformerMixin, BaseEstimator):
        """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

        def __init__(self, n_neighbors=5, metric='euclidean', method='sw-graph', n_jobs=-1):
            self.n_neighbors = n_neighbors
            self.method = method
            self.metric = metric
            self.n_jobs = n_jobs

        def fit(self, X):
            self.n_samples_fit_ = _X.shape[0]
            space = {'euclidean': 'l2', 'cosine': 'cosinesimil', 'l1': 'l1', 'l2': 'l2'}[self.metric]
            self.nmslib_ = nmslib.init(method=self.method, space=space)
            self.nmslib_.addDataPointBatch(_X.copy())
            self.nmslib_.createIndex()  # see more metric in the manual
            return self  # https://github.com/nmslib/nmslib/tree/master/manual

        def transform(self, X):
            n_samples_transform = _X.shape[0]
            n_neighbors = self.n_neighbors + 1
            if self.n_jobs < 0:
                num_threads = joblib.cpu_count() + self.n_jobs + 1
            else:
                num_threads = self.n_jobs
            results = self.nmslib_.knnQueryBatch(_X.copy(), k=n_neighbors, num_threads=num_threads)
            indices, distances = zip(*results)
            indices, distances = (np.vstack(indices), np.vstack(distances))
            indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
            kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(), indptr), shape=(n_samples_transform, self.n_samples_fit_))
            return kneighbors_graph

    def load_mnist(n_samples):  # For compatibility reasons, as each sample is considered as its own
        """Load MNIST, shuffle the data, and return only n_samples."""  # neighbor, one extra neighbor will be computed.
        mnist = fetch_openml('mnist_784', as_frame=False)
        _X, _y = shuffle(mnist.data, mnist.target, random_state=2)
        return (_X[:n_samples] / 255, _y[:n_samples])  # Same handling as done in joblib for negative values of n_jobs:  # in particular, `n_jobs == -1` means "as many threads as CPUs".
    return NMSlibTransformer, load_mnist, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We benchmark the different exact/approximate nearest neighbors transformers.


    """
    )
    return


@app.cell
def _(NMSlibTransformer, PyNNDescentTransformer, load_mnist, np):
    import time
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsTransformer
    from sklearn.pipeline import make_pipeline
    datasets = [('MNIST_10000', load_mnist(n_samples=10000)), ('MNIST_20000', load_mnist(n_samples=20000))]
    max_iter = 500
    perplexity = 30
    metric = 'euclidean'
    n_neighbors = int(3.0 * perplexity + 1) + 1
    tsne_params = dict(init='random', perplexity=perplexity, method='barnes_hut', random_state=42, max_iter=max_iter, learning_rate='auto')
    _transformers = [('KNeighborsTransformer', KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance', metric=metric)), ('NMSlibTransformer', NMSlibTransformer(n_neighbors=n_neighbors, metric=metric)), ('PyNNDescentTransformer', PyNNDescentTransformer(n_neighbors=n_neighbors, metric=metric, parallel_batch_queries=True))]
    for _dataset_name, (_X, _y) in datasets:
        _msg = f'Benchmarking on {_dataset_name}:'
        print(f'\n{_msg}\n' + str('-' * len(_msg)))
    # TSNE requires a certain number of neighbors which depends on the
    # perplexity parameter.
    # Add one since we include each sample as its own neighbor.
        for _transformer_name, _transformer in _transformers:
            _longest = np.max([len(name) for name, model in _transformers])
            _start = time.time()
            _transformer.fit(_X)  # pca cannot be used with precomputed distances
            fit_duration = time.time() - _start
            print(f'{_transformer_name:<{_longest}} {fit_duration:.3f} sec (fit)')
            _start = time.time()
            _Xt = _transformer.transform(_X)
            _transform_duration = time.time() - _start
            print(f'{_transformer_name:<{_longest}} {_transform_duration:.3f} sec (transform)')
            if _transformer_name == 'PyNNDescentTransformer':
                _start = time.time()
                _Xt = _transformer.transform(_X)
                _transform_duration = time.time() - _start
                print(f'{_transformer_name:<{_longest}} {_transform_duration:.3f} sec (transform)')
    return (
        KNeighborsTransformer,
        TSNE,
        datasets,
        make_pipeline,
        metric,
        n_neighbors,
        time,
        tsne_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Sample output::

        Benchmarking on MNIST_10000:
        ----------------------------
        KNeighborsTransformer  0.007 sec (fit)
        KNeighborsTransformer  1.139 sec (transform)
        NMSlibTransformer      0.208 sec (fit)
        NMSlibTransformer      0.315 sec (transform)
        PyNNDescentTransformer 4.823 sec (fit)
        PyNNDescentTransformer 4.884 sec (transform)
        PyNNDescentTransformer 0.744 sec (transform)

        Benchmarking on MNIST_20000:
        ----------------------------
        KNeighborsTransformer  0.011 sec (fit)
        KNeighborsTransformer  5.769 sec (transform)
        NMSlibTransformer      0.733 sec (fit)
        NMSlibTransformer      1.077 sec (transform)
        PyNNDescentTransformer 14.448 sec (fit)
        PyNNDescentTransformer 7.103 sec (transform)
        PyNNDescentTransformer 1.759 sec (transform)

    Notice that the `PyNNDescentTransformer` takes more time during the first
    `fit` and the first `transform` due to the overhead of the numba just in time
    compiler. But after the first call, the compiled Python code is kept in a
    cache by numba and subsequent calls do not suffer from this initial overhead.
    Both :class:`~sklearn.neighbors.KNeighborsTransformer` and `NMSlibTransformer`
    are only run once here as they would show more stable `fit` and `transform`
    times (they don't have the cold start problem of PyNNDescentTransformer).


    """
    )
    return


@app.cell
def _(
    KNeighborsTransformer,
    NMSlibTransformer,
    TSNE,
    datasets,
    make_pipeline,
    metric,
    n_neighbors,
    np,
    time,
    tsne_params,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    _transformers = [('TSNE with internal NearestNeighbors', TSNE(metric=metric, **tsne_params)), ('TSNE with KNeighborsTransformer', make_pipeline(KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance', metric=metric), TSNE(metric='precomputed', **tsne_params))), ('TSNE with NMSlibTransformer', make_pipeline(NMSlibTransformer(n_neighbors=n_neighbors, metric=metric), TSNE(metric='precomputed', **tsne_params)))]
    nrows = len(datasets)
    ncols = np.sum([1 for name, model in _transformers if 'TSNE' in name])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(5 * ncols, 4 * nrows))
    axes = axes.ravel()
    i_ax = 0
    for _dataset_name, (_X, _y) in datasets:
        _msg = f'Benchmarking on {_dataset_name}:'
        print(f'\n{_msg}\n' + str('-' * len(_msg)))
        for _transformer_name, _transformer in _transformers:
            _longest = np.max([len(name) for name, model in _transformers])
            _start = time.time()
            _Xt = _transformer.fit_transform(_X)
            _transform_duration = time.time() - _start
            print(f'{_transformer_name:<{_longest}} {_transform_duration:.3f} sec (fit_transform)')
            axes[i_ax].set_title(_transformer_name + '\non ' + _dataset_name)
            axes[i_ax].scatter(_Xt[:, 0], _Xt[:, 1], c=_y.astype(np.int32), alpha=0.2, cmap=plt.cm.viridis)
            axes[i_ax].xaxis.set_major_formatter(NullFormatter())
            axes[i_ax].yaxis.set_major_formatter(NullFormatter())
            axes[i_ax].axis('tight')
            i_ax += 1
    # init the plot
    fig.tight_layout()
    plt.show()  # plot TSNE embedding which should be very similar across methods
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Sample output::

        Benchmarking on MNIST_10000:
        ----------------------------
        TSNE with internal NearestNeighbors 24.828 sec (fit_transform)
        TSNE with KNeighborsTransformer     20.111 sec (fit_transform)
        TSNE with NMSlibTransformer         21.757 sec (fit_transform)

        Benchmarking on MNIST_20000:
        ----------------------------
        TSNE with internal NearestNeighbors 51.955 sec (fit_transform)
        TSNE with KNeighborsTransformer     50.994 sec (fit_transform)
        TSNE with NMSlibTransformer         43.536 sec (fit_transform)

    We can observe that the default :class:`~sklearn.manifold.TSNE` estimator with
    its internal :class:`~sklearn.neighbors.NearestNeighbors` implementation is
    roughly equivalent to the pipeline with :class:`~sklearn.manifold.TSNE` and
    :class:`~sklearn.neighbors.KNeighborsTransformer` in terms of performance.
    This is expected because both pipelines rely internally on the same
    :class:`~sklearn.neighbors.NearestNeighbors` implementation that performs
    exacts neighbors search. The approximate `NMSlibTransformer` is already
    slightly faster than the exact search on the smallest dataset but this speed
    difference is expected to become more significant on datasets with a larger
    number of samples.

    Notice however that not all approximate search methods are guaranteed to
    improve the speed of the default exact search method: indeed the exact search
    implementation significantly improved since scikit-learn 1.1. Furthermore, the
    brute-force exact search method does not require building an index at `fit`
    time. So, to get an overall performance improvement in the context of the
    :class:`~sklearn.manifold.TSNE` pipeline, the gains of the approximate search
    at `transform` need to be larger than the extra time spent to build the
    approximate search index at `fit` time.

    Finally, the TSNE algorithm itself is also computationally intensive,
    irrespective of the nearest neighbors search. So speeding-up the nearest
    neighbors search step by a factor of 5 would not result in a speed up by a
    factor of 5 for the overall pipeline.


    """
    )
    return

if __name__ == "__main__":
    app.run()
