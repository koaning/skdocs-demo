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

    # Evaluation of outlier detection estimators

    This example compares two outlier detection algorithms, namely
    `local_outlier_factor` (LOF) and `isolation_forest` (IForest), on
    real-world datasets available in :class:`sklearn.datasets`. The goal is to show
    that different algorithms perform well on different datasets and contrast their
    training speed and sensitivity to hyperparameters.

    The algorithms are trained (without labels) on the whole dataset assumed to
    contain outliers.

    1. The ROC curves are computed using knowledge of the ground-truth labels
       and displayed using :class:`~sklearn.metrics.RocCurveDisplay`.

    2. The performance is assessed in terms of the ROC-AUC.

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
    ## Dataset preprocessing and model training

    Different outlier detection models require different preprocessing. In the
    presence of categorical variables,
    :class:`~sklearn.preprocessing.OrdinalEncoder` is often a good strategy for
    tree-based models such as :class:`~sklearn.ensemble.IsolationForest`, whereas
    neighbors-based models such as :class:`~sklearn.neighbors.LocalOutlierFactor`
    would be impacted by the ordering induced by ordinal encoding. To avoid
    inducing an ordering, on should rather use
    :class:`~sklearn.preprocessing.OneHotEncoder`.

    Neighbors-based models may also require scaling of the numerical features (see
    for instance `neighbors_scaling`). In the presence of outliers, a good
    option is to use a :class:`~sklearn.preprocessing.RobustScaler`.


    """
    )
    return


@app.cell
def _():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

    def make_estimator(name, categorical_columns=None, iforest_kw=None, lof_kw=None):
        """Create an outlier detection estimator based on its name."""
        if name == 'LOF':
            outlier_detector = LocalOutlierFactor(**lof_kw or {})
            if categorical_columns is None:
                _preprocessor = RobustScaler()
            else:
                _preprocessor = ColumnTransformer(transformers=[('categorical', OneHotEncoder(), categorical_columns)], remainder=RobustScaler())
        else:
            outlier_detector = IsolationForest(**iforest_kw or {})
            if categorical_columns is None:
                _preprocessor = None
            else:
                ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                _preprocessor = ColumnTransformer(transformers=[('categorical', ordinal_encoder, categorical_columns)], remainder='passthrough')
        return make_pipeline(_preprocessor, outlier_detector)  # name == "IForest"
    return LocalOutlierFactor, RobustScaler, make_estimator, make_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The following `fit_predict` function returns the average outlier score of X.


    """
    )
    return


@app.cell
def _(model_name):
    from time import perf_counter


    def fit_predict(estimator, X):
        tic = perf_counter()
        if estimator[-1].__class__.__name__ == "LocalOutlierFactor":
            estimator.fit(X)
            y_score = estimator[-1].negative_outlier_factor_
        else:  # "IsolationForest"
            y_score = estimator.fit(X).decision_function(X)
        toc = perf_counter()
        print(f"Duration for {model_name}: {toc - tic:.2f} s")
        return y_score
    return (fit_predict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On the rest of the example we process one dataset per section. After loading
    the data, the targets are modified to consist of two classes: 0 representing
    inliers and 1 representing outliers. Due to computational constraints of the
    scikit-learn documentation, the sample size of some datasets is reduced using
    a stratified :class:`~sklearn.model_selection.train_test_split`.

    Furthermore, we set `n_neighbors` to match the expected number of anomalies
    `expected_n_anomalies = n_samples * expected_anomaly_fraction`. This is a good
    heuristic as long as the proportion of outliers is not very low, the reason
    being that `n_neighbors` should be at least greater than the number of samples
    in the less populated cluster (see
    `sphx_glr_auto_examples_neighbors_plot_lof_outlier_detection.py`).

    ### KDDCup99 - SA dataset

    The `kddcup99_dataset` was generated using a closed network and
    hand-injected attacks. The SA dataset is a subset of it obtained by simply
    selecting all the normal data and an anomaly proportion of around 3%.


    """
    )
    return


@app.cell
def _():
    import numpy as np

    from sklearn.datasets import fetch_kddcup99
    from sklearn.model_selection import train_test_split

    X, y = fetch_kddcup99(
        subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True
    )
    y = (y != b"normal.").astype(np.int32)
    X, _, y, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)

    n_samples, anomaly_frac = X.shape[0], y.mean()
    print(f"{n_samples} datapoints with {y.sum()} anomalies ({anomaly_frac:.02%})")
    return X, anomaly_frac, n_samples, np, train_test_split, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The SA dataset contains 41 features out of which 3 are categorical:
    "protocol_type", "service" and "flag".


    """
    )
    return


@app.cell
def _(X, anomaly_frac, fit_predict, make_estimator, n_samples, y):
    y_true = {}
    y_score = {'LOF': {}, 'IForest': {}}
    model_names = ['LOF', 'IForest']
    _cat_columns = ['protocol_type', 'service', 'flag']
    y_true['KDDCup99 - SA'] = y
    for model_name in model_names:
        _model = make_estimator(name=model_name, categorical_columns=_cat_columns, lof_kw={'n_neighbors': int(n_samples * anomaly_frac)}, iforest_kw={'random_state': 42})
        y_score[model_name]['KDDCup99 - SA'] = fit_predict(_model, X)
    return model_name, model_names, y_score, y_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Forest covertypes dataset

    The `covtype_dataset` is a multiclass dataset where the target is the
    dominant species of tree in a given patch of forest. It contains 54 features,
    some of which ("Wilderness_Area" and "Soil_Type") are already binary encoded.
    Though originally meant as a classification task, one can regard inliers as
    samples encoded with label 2 and outliers as those with label 4.


    """
    )
    return


@app.cell
def _(np, train_test_split):
    from sklearn.datasets import fetch_covtype
    X_1, y_1 = fetch_covtype(return_X_y=True, as_frame=True)
    _s = (y_1 == 2) + (y_1 == 4)
    X_1 = X_1.loc[_s]
    y_1 = y_1.loc[_s]
    y_1 = (y_1 != 2).astype(np.int32)
    X_1, _, y_1, _ = train_test_split(X_1, y_1, train_size=0.05, stratify=y_1, random_state=42)
    X_forestcover = X_1
    n_samples_1, anomaly_frac_1 = (X_1.shape[0], y_1.mean())
    print(f'{n_samples_1} datapoints with {y_1.sum()} anomalies ({anomaly_frac_1:.02%})')
    return X_1, X_forestcover, anomaly_frac_1, n_samples_1, y_1


@app.cell
def _(
    X_1,
    anomaly_frac_1,
    fit_predict,
    make_estimator,
    model_names,
    n_samples_1,
    y_1,
    y_score,
    y_true,
):
    y_true['forestcover'] = y_1
    for model_name_1 in model_names:
        _model = make_estimator(name=model_name_1, lof_kw={'n_neighbors': int(n_samples_1 * anomaly_frac_1)}, iforest_kw={'random_state': 42})
        y_score[model_name_1]['forestcover'] = fit_predict(_model, X_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Ames Housing dataset

    The [Ames housing dataset](http://www.openml.org/d/43926) is originally a
    regression dataset where the target are sales prices of houses in Ames, Iowa.
    Here we convert it into an outlier detection problem by regarding houses with
    price over 70 USD/sqft. To make the problem easier, we drop intermediate
    prices between 40 and 70 USD/sqft.


    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    X_2, y_2 = fetch_openml(name='ames_housing', version=1, return_X_y=True, as_frame=True)
    y_2 = y_2.div(X_2['Lot_Area'])
    X_2['Misc_Feature'] = X_2['Misc_Feature'].cat.add_categories('NoInfo').fillna('NoInfo')
    X_2['Mas_Vnr_Type'] = X_2['Mas_Vnr_Type'].cat.add_categories('NoInfo').fillna('NoInfo')
    X_2.drop(columns='Lot_Area', inplace=True)
    # None values in pandas 1.5.1 were mapped to np.nan in pandas 2.0.1
    mask = (y_2 < 40) | (y_2 > 70)
    X_2 = X_2.loc[mask]
    y_2 = y_2.loc[mask]
    y_2.hist(bins=20, edgecolor='black')
    plt.xlabel('House price in USD/sqft')
    _ = plt.title('Distribution of house prices in Ames')
    return X_2, fetch_openml, plt, y_2


@app.cell
def _(X_2, np, y_2):
    y_3 = (y_2 > 70).astype(np.int32)
    n_samples_2, anomaly_frac_2 = (X_2.shape[0], y_3.mean())
    print(f'{n_samples_2} datapoints with {y_3.sum()} anomalies ({anomaly_frac_2:.02%})')
    return anomaly_frac_2, n_samples_2, y_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The dataset contains 46 categorical features. In this case it is easier use a
    :class:`~sklearn.compose.make_column_selector` to find them instead of passing
    a list made by hand.


    """
    )
    return


@app.cell
def _(
    X_2,
    anomaly_frac_2,
    fit_predict,
    make_estimator,
    model_names,
    n_samples_2,
    y_3,
    y_score,
    y_true,
):
    from sklearn.compose import make_column_selector as selector
    categorical_columns_selector = selector(dtype_include='category')
    _cat_columns = categorical_columns_selector(X_2)
    y_true['ames_housing'] = y_3
    for model_name_2 in model_names:
        _model = make_estimator(name=model_name_2, categorical_columns=_cat_columns, lof_kw={'n_neighbors': int(n_samples_2 * anomaly_frac_2)}, iforest_kw={'random_state': 42})
        y_score[model_name_2]['ames_housing'] = fit_predict(_model, X_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Cardiotocography dataset

    The [Cardiotocography dataset](http://www.openml.org/d/1466) is a multiclass
    dataset of fetal cardiotocograms, the classes being the fetal heart rate (FHR)
    pattern encoded with labels from 1 to 10. Here we set class 3 (the minority
    class) to represent the outliers. It contains 30 numerical features, some of
    which are binary encoded and some are continuous.


    """
    )
    return


@app.cell
def _(fetch_openml, np):
    X_3, y_4 = fetch_openml(name='cardiotocography', version=1, return_X_y=True, as_frame=False)
    X_cardiotocography = X_3
    _s = y_4 == '3'
    y_4 = _s.astype(np.int32)
    n_samples_3, anomaly_frac_3 = (X_3.shape[0], y_4.mean())
    print(f'{n_samples_3} datapoints with {y_4.sum()} anomalies ({anomaly_frac_3:.02%})')
    return X_3, X_cardiotocography, anomaly_frac_3, n_samples_3, y_4


@app.cell
def _(
    X_3,
    anomaly_frac_3,
    fit_predict,
    make_estimator,
    model_names,
    n_samples_3,
    y_4,
    y_score,
    y_true,
):
    y_true['cardiotocography'] = y_4
    for model_name_3 in model_names:
        _model = make_estimator(name=model_name_3, lof_kw={'n_neighbors': int(n_samples_3 * anomaly_frac_3)}, iforest_kw={'random_state': 42})
        y_score[model_name_3]['cardiotocography'] = fit_predict(_model, X_3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot and interpret results

    The algorithm performance relates to how good the true positive rate (TPR) is
    at low value of the false positive rate (FPR). The best algorithms have the
    curve on the top-left of the plot and the area under curve (AUC) close to 1.
    The diagonal dashed line represents a random classification of outliers and
    inliers.


    """
    )
    return


@app.cell
def _(model_names, plt, y_score, y_true):
    import math
    from sklearn.metrics import RocCurveDisplay
    cols = 2
    pos_label = 0
    datasets_names = y_true.keys()
    rows = math.ceil(len(datasets_names) / cols)
    _fig, axs = plt.subplots(nrows=rows, ncols=cols, squeeze=False, figsize=(10, rows * 4))
    for _ax, dataset_name in zip(axs.ravel(), datasets_names):
        for _model_idx, model_name_4 in enumerate(model_names):
            _display = RocCurveDisplay.from_predictions(y_true[dataset_name], y_score[model_name_4][dataset_name], pos_label=pos_label, name=model_name_4, ax=_ax, plot_chance_level=_model_idx == len(model_names) - 1, chance_level_kw={'linestyle': ':'})
        _ax.set_title(dataset_name)
    _ = plt.tight_layout(pad=2.0)
    return RocCurveDisplay, pos_label


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We observe that once the number of neighbors is tuned, LOF and IForest perform
    similarly in terms of ROC AUC for the forestcover and cardiotocography
    datasets. The score for IForest is slightly better for the SA dataset and LOF
    performs considerably better on the Ames housing dataset than IForest.

    Recall however that Isolation Forest tends to train much faster than LOF on
    datasets with a large number of samples. LOF needs to compute pairwise
    distances to find nearest neighbors, which has a quadratic complexity with respect
    to the number of observations. This can make this method prohibitive on large
    datasets.

    ## Ablation study

    In this section we explore the impact of the hyperparameter `n_neighbors` and
    the choice of scaling the numerical variables on the LOF model. Here we use
    the `covtype_dataset` dataset as the binary encoded categories introduce
    a natural scale of euclidean distances between 0 and 1. We then want a scaling
    method to avoid granting a privilege to non-binary features and that is robust
    enough to outliers so that the task of finding them does not become too
    difficult.


    """
    )
    return


@app.cell
def _(
    LocalOutlierFactor,
    RobustScaler,
    RocCurveDisplay,
    X_forestcover,
    make_pipeline,
    np,
    plt,
    pos_label,
    y_true,
):
    X_4 = X_forestcover
    y_5 = y_true['forestcover']
    n_samples_4 = X_4.shape[0]
    n_neighbors_list = (n_samples_4 * np.array([0.2, 0.02, 0.01, 0.001])).astype(np.int32)
    _model = make_pipeline(RobustScaler(), LocalOutlierFactor())
    linestyles = ['solid', 'dashed', 'dashdot', ':', (5, (10, 3))]
    _fig, _ax = plt.subplots()
    for _model_idx, (_linestyle, n_neighbors) in enumerate(zip(linestyles, n_neighbors_list)):
        _model.set_params(localoutlierfactor__n_neighbors=n_neighbors)
        _model.fit(X_4)
        y_score_1 = _model[-1].negative_outlier_factor_
        _display = RocCurveDisplay.from_predictions(y_5, y_score_1, pos_label=pos_label, name=f'n_neighbors = {n_neighbors}', ax=_ax, plot_chance_level=_model_idx == len(n_neighbors_list) - 1, chance_level_kw={'linestyle': (0, (1, 10))}, curve_kwargs=dict(linestyle=_linestyle, linewidth=2))
    _ = _ax.set_title('RobustScaler with varying n_neighbors\non forestcover dataset')
    return X_4, linestyles, n_samples_4, y_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We observe that the number of neighbors has a big impact on the performance of
    the model. If one has access to (at least some) ground truth labels, it is
    then important to tune `n_neighbors` accordingly. A convenient way to do so is
    to explore values for `n_neighbors` of the order of magnitud of the expected
    contamination.


    """
    )
    return


@app.cell
def _(
    LocalOutlierFactor,
    RobustScaler,
    RocCurveDisplay,
    X_4,
    linestyles,
    make_pipeline,
    n_samples_4,
    plt,
    pos_label,
    y_5,
):
    from sklearn.preprocessing import MinMaxScaler, SplineTransformer, StandardScaler
    preprocessor_list = [None, RobustScaler(), StandardScaler(), MinMaxScaler(), SplineTransformer()]
    _expected_anomaly_fraction = 0.02
    _lof = LocalOutlierFactor(n_neighbors=int(n_samples_4 * _expected_anomaly_fraction))
    _fig, _ax = plt.subplots()
    for _model_idx, (_linestyle, _preprocessor) in enumerate(zip(linestyles, preprocessor_list)):
        _model = make_pipeline(_preprocessor, _lof)
        _model.fit(X_4)
        y_score_2 = _model[-1].negative_outlier_factor_
        _display = RocCurveDisplay.from_predictions(y_5, y_score_2, pos_label=pos_label, name=str(_preprocessor).split('(')[0], ax=_ax, plot_chance_level=_model_idx == len(preprocessor_list) - 1, chance_level_kw={'linestyle': (0, (1, 10))}, curve_kwargs=dict(linestyle=_linestyle, linewidth=2))
    _ = _ax.set_title('Fixed n_neighbors with varying preprocessing\non forestcover dataset')
    return (preprocessor_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On the one hand, :class:`~sklearn.preprocessing.RobustScaler` scales each
    feature independently by using the interquartile range (IQR) by default, which
    is the range between the 25th and 75th percentiles of the data. It centers the
    data by subtracting the median and then scale it by dividing by the IQR. The
    IQR is robust to outliers: the median and interquartile range are less
    affected by extreme values than the range, the mean and the standard
    deviation. Furthermore, :class:`~sklearn.preprocessing.RobustScaler` does not
    squash marginal outlier values, contrary to
    :class:`~sklearn.preprocessing.StandardScaler`.

    On the other hand, :class:`~sklearn.preprocessing.MinMaxScaler` scales each
    feature individually such that its range maps into the range between zero and
    one. If there are outliers in the data, they can skew it towards either the
    minimum or maximum values, leading to a completely different distribution of
    data with large marginal outliers: all non-outlier values can be collapsed
    almost together as a result.

    We also evaluated no preprocessing at all (by passing `None` to the pipeline),
    :class:`~sklearn.preprocessing.StandardScaler` and
    :class:`~sklearn.preprocessing.SplineTransformer`. Please refer to their
    respective documentation for more details.

    Note that the optimal preprocessing depends on the dataset, as shown below:


    """
    )
    return


@app.cell
def _(
    LocalOutlierFactor,
    RocCurveDisplay,
    X_cardiotocography,
    linestyles,
    make_pipeline,
    plt,
    pos_label,
    preprocessor_list,
    y_true,
):
    X_5 = X_cardiotocography
    y_6 = y_true['cardiotocography']
    n_samples_5, _expected_anomaly_fraction = (X_5.shape[0], 0.025)
    _lof = LocalOutlierFactor(n_neighbors=int(n_samples_5 * _expected_anomaly_fraction))
    _fig, _ax = plt.subplots()
    for _model_idx, (_linestyle, _preprocessor) in enumerate(zip(linestyles, preprocessor_list)):
        _model = make_pipeline(_preprocessor, _lof)
        _model.fit(X_5)
        y_score_3 = _model[-1].negative_outlier_factor_
        _display = RocCurveDisplay.from_predictions(y_6, y_score_3, pos_label=pos_label, name=str(_preprocessor).split('(')[0], ax=_ax, plot_chance_level=_model_idx == len(preprocessor_list) - 1, chance_level_kw={'linestyle': (0, (1, 10))}, curve_kwargs=dict(linestyle=_linestyle, linewidth=2))
    _ax.set_title('Fixed n_neighbors with varying preprocessing\non cardiotocography dataset')
    plt.show()
    return

if __name__ == "__main__":
    app.run()
