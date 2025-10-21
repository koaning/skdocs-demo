import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    # Image denoising using dictionary learning

    An example comparing the effect of reconstructing noisy fragments
    of a raccoon face image using firstly online `DictionaryLearning` and
    various transform methods.

    The dictionary is fitted on the distorted left half of the image, and
    subsequently used to reconstruct the right half. Note that even better
    performance could be achieved by fitting to an undistorted (i.e.
    noiseless) image, but here we start from the assumption that it is not
    available.

    A common practice for evaluating the results of image denoising is by looking
    at the difference between the reconstruction and the original image. If the
    reconstruction is perfect this will look like Gaussian noise.

    It can be seen from the plots that the results of `omp` with two
    non-zero coefficients is a bit less biased than when keeping only one
    (the edges look less prominent). It is in addition closer from the ground
    truth in Frobenius norm.

    The result of `least_angle_regression` is much more strongly biased: the
    difference is reminiscent of the local intensity value of the original image.

    Thresholding is clearly not useful for denoising, but it is here to show that
    it can produce a suggestive output with very high speed, and thus be useful
    for other tasks such as object classification, where performance is not
    necessarily related to visualisation.

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
    ## Generate distorted image


    """
    )
    return


app._unparsable_cell(
    r"""
    import numpy as np
    try:
        from scipy.datasets import face  # Scipy >= 1.10
    except ImportError:
    raccoon_face = face(gray=True)
    raccoon_face = raccoon_face / 255.0
    raccoon_face = raccoon_face[::4, ::4] + raccoon_face[1::4, ::4] + raccoon_face[::4, 1::4] + raccoon_face[1::4, 1::4]
    raccoon_face /= 4.0
    height, width = raccoon_face.shape
    # Convert from uint8 representation with values between 0 and 255 to
    # a floating point representation with values between 0 and 1.
    print('Distorting image...')
    distorted = raccoon_face.copy()
    # downsample for higher speed
    # Distort the right half of the image
    distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Display the distorted image


    """
    )
    return


@app.cell
def _(distorted, np, raccoon_face):
    import matplotlib.pyplot as plt


    def show_with_diff(image, reference, title):
        """Helper function to display denoising"""
        plt.figure(figsize=(5, 3.3))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
        plt.subplot(1, 2, 2)
        difference = image - reference

        plt.title("Difference (norm: %.2f)" % np.sqrt(np.sum(difference**2)))
        plt.imshow(
            difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr, interpolation="nearest"
        )
        plt.xticks(())
        plt.yticks(())
        plt.suptitle(title, size=16)
        plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)


    show_with_diff(distorted, raccoon_face, "Distorted image")
    return plt, show_with_diff


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Extract reference patches


    """
    )
    return


@app.cell
def _(distorted, np, width):
    from time import time
    from sklearn.feature_extraction.image import extract_patches_2d
    print('Extracting reference patches...')
    t0 = time()
    # Extract all reference patches from the left half of the image
    patch_size = (7, 7)
    data = extract_patches_2d(distorted[:, :width // 2], patch_size)
    data = data.reshape(data.shape[0], -1)
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)
    print(f'{data.shape[0]} patches extracted in %.2fs.' % (time() - t0))
    return data, extract_patches_2d, patch_size, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Learn the dictionary from reference patches


    """
    )
    return


@app.cell
def _(data, patch_size, plt, time):
    from sklearn.decomposition import MiniBatchDictionaryLearning
    print('Learning the dictionary...')
    t0_1 = time()
    dico = MiniBatchDictionaryLearning(n_components=50, batch_size=200, alpha=1.0, max_iter=10)
    V = dico.fit(data).components_
    dt = time() - t0_1  # increase to 300 for higher quality results at the cost of slower
    print(f'{dico.n_iter_} iterations / {dico.n_steps_} steps in {dt:.2f}.')  # training times.
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Dictionary learned from face patches\n' + 'Train time %.1fs on %d patches' % (dt, len(data)), fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    return V, dico


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Extract noisy patches and reconstruct them using the dictionary


    """
    )
    return


@app.cell
def _(
    V,
    dico,
    distorted,
    extract_patches_2d,
    height,
    np,
    patch_size,
    plt,
    raccoon_face,
    show_with_diff,
    time,
    width,
):
    from sklearn.feature_extraction.image import reconstruct_from_patches_2d
    print('Extracting noisy patches... ')
    t0_2 = time()
    data_1 = extract_patches_2d(distorted[:, width // 2:], patch_size)
    data_1 = data_1.reshape(data_1.shape[0], -1)
    intercept = np.mean(data_1, axis=0)
    data_1 = data_1 - intercept
    print('done in %.2fs.' % (time() - t0_2))
    transform_algorithms = [('Orthogonal Matching Pursuit\n1 atom', 'omp', {'transform_n_nonzero_coefs': 1}), ('Orthogonal Matching Pursuit\n2 atoms', 'omp', {'transform_n_nonzero_coefs': 2}), ('Least-angle regression\n4 atoms', 'lars', {'transform_n_nonzero_coefs': 4}), ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': 0.1})]
    reconstructions = {}
    for title, transform_algorithm, kwargs in transform_algorithms:
        print(title + '...')
        reconstructions[title] = raccoon_face.copy()
        t0_2 = time()
        dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
        code = dico.transform(data_1)
        patches = np.dot(code, V)
        patches = patches + intercept
        patches = patches.reshape(len(data_1), *patch_size)
        if transform_algorithm == 'threshold':
            patches = patches - patches.min()
            patches = patches / patches.max()
        reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(patches, (height, width // 2))
        dt_1 = time() - t0_2
        print('done in %.2fs.' % dt_1)
        show_with_diff(reconstructions[title], raccoon_face, title + ' (time: %.1fs)' % dt_1)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
