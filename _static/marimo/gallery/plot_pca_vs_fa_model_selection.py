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

    # Model selection with Probabilistic PCA and Factor Analysis (FA)

    Probabilistic PCA and Factor Analysis are probabilistic models.
    The consequence is that the likelihood of new data can be used
    for model selection and covariance estimation.
    Here we compare PCA and FA with cross-validation on low rank data corrupted
    with homoscedastic noise (noise variance
    is the same for each feature) or heteroscedastic noise (noise variance
    is the different for each feature). In a second step we compare the model
    likelihood to the likelihoods obtained from shrinkage covariance estimators.

    One can observe that with homoscedastic noise both FA and PCA succeed
    in recovering the size of the low rank subspace. The likelihood with PCA
    is higher than FA in this case. However PCA fails and overestimates
    the rank when heteroscedastic noise is present. Under appropriate
    circumstances (choice of the number of components), the held-out
    data is more likely for low rank models than for shrinkage models.

    The automatic estimation from
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
    by Thomas P. Minka is also compared.

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
    ## Create the data


    """
    )
    return


@app.cell
def _():
    import numpy as np
    from scipy import linalg
    n_samples, n_features, rank = (500, 25, 5)
    sigma = 1.0
    rng = np.random.RandomState(42)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    _X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    X_homo = _X + sigma * rng.randn(n_samples, n_features)
    sigmas = sigma * rng.rand(n_features) + sigma / 2.0
    # Adding homoscedastic noise
    # Adding heteroscedastic noise
    X_hetero = _X + rng.randn(n_samples, n_features) * sigmas
    return X_hetero, X_homo, n_features, np, rank


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fit the models


    """
    )
    return


@app.cell
def _(X_hetero, X_homo, n_features, np, rank):
    import matplotlib.pyplot as plt
    from sklearn.covariance import LedoitWolf, ShrunkCovariance
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.model_selection import GridSearchCV, cross_val_score
    n_components = np.arange(0, n_features, 5)

    def compute_scores(X):  # options for n_components
        pca = PCA(svd_solver='full')
        fa = FactorAnalysis()
        pca_scores, fa_scores = ([], [])
        for n in n_components:
            pca.n_components = n
            fa.n_components = n
            pca_scores.append(np.mean(cross_val_score(pca, _X)))
            fa_scores.append(np.mean(cross_val_score(fa, _X)))
        return (pca_scores, fa_scores)

    def shrunk_cov_score(X):
        shrinkages = np.logspace(-2, 0, 30)
        cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
        return np.mean(cross_val_score(cv.fit(_X).best_estimator_, _X))

    def lw_score(X):
        return np.mean(cross_val_score(LedoitWolf(), _X))
    for _X, title in [(X_homo, 'Homoscedastic Noise'), (X_hetero, 'Heteroscedastic Noise')]:
        pca_scores, fa_scores = compute_scores(_X)
        n_components_pca = n_components[np.argmax(pca_scores)]
        n_components_fa = n_components[np.argmax(fa_scores)]
        pca = PCA(svd_solver='full', n_components='mle')
        pca.fit(_X)
        n_components_pca_mle = pca.n_components_
        print('best n_components by PCA CV = %d' % n_components_pca)
        print('best n_components by FactorAnalysis CV = %d' % n_components_fa)
        print('best n_components by PCA MLE = %d' % n_components_pca_mle)
        plt.figure()
        plt.plot(n_components, pca_scores, 'b', label='PCA scores')
        plt.plot(n_components, fa_scores, 'r', label='FA scores')
        plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
        plt.axvline(n_components_pca, color='b', label='PCA CV: %d' % n_components_pca, linestyle='--')
        plt.axvline(n_components_fa, color='r', label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
        plt.axvline(n_components_pca_mle, color='k', label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')
        plt.axhline(shrunk_cov_score(_X), color='violet', label='Shrunk Covariance MLE', linestyle='-.')
        plt.axhline(lw_score(_X), color='orange', label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')
        plt.xlabel('nb of components')
        plt.ylabel('CV scores')
        plt.legend(loc='lower right')
        plt.title(title)
    plt.show()  # compare with other covariance estimators
    return

if __name__ == "__main__":
    app.run()
