"""
Created on 07/04/2021
@author: ingridtveten

TODO: Description...
"""

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from skimage.util import compare_images

import logging
logger = logging.getLogger()


def plot_image_diff(img1, img2, img_labels=['Image 1', 'Image 2']):

    from skimage.util import compare_images
    from matplotlib.gridspec import GridSpec

    diff_img = compare_images(img1, img2, method='diff')

    fig = plt.figure(figsize=(8, 9))
    gs = GridSpec(3, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1:, :])

    ax0.imshow(img1, cmap='bone')
    ax0.set_title(img_labels[0])
    ax1.imshow(img1, cmap='Reds')
    ax1.set_title(img_labels[1])
    ax2.imshow(diff_img, cmap='gray')
    ax2.set_title('Diff comparison')

    for a in (ax0, ax1, ax2):
        a.axis('off')
    plt.tight_layout()
    plt.plot()
    plt.show()
    plt.close(fig)


def plot_3d_image_series(img_array, grid_edge_size=None,
                         plot_title=None, cmap='gist_gray',
                         save_fig=False):
    """
    Plot a N x N grid of slice images using matplotlib

    Parameters
    ----------
    image : ndarray
        A ndarray (3D volume)
    grid_edge_size : int
        The wanted edge length N of the N x N grid of subimages. Defaults to
        the largest edge size possible with the number of subimages to be
        plotted.
    """

    num_slices_image = np.shape(img_array)[0] # z-size
    N = get_image_grid_edge_size(num_slices_image, grid_edge_size)
    NROWS = N
    NCOLS = N

    FIGSIZE = (16, 16)
    IMSHOW_ARGS = {'cmap': cmap}

    fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS+1, figsize=FIGSIZE)
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16)

    mid = num_slices_image // 2
    for i in range(NROWS):
        for j in range(NCOLS):
            img = axes[i, j].imshow(img_array[mid + i * NCOLS + j],
                                    **IMSHOW_ARGS)
            axes[i, j].set_axis_off()
        axes[i, NCOLS].set_axis_off()

    cb = fig.colorbar(img, ax=axes[:, NCOLS], shrink=2.0)

    plt.tight_layout(pad=1, w_pad=0.1, h_pad=0.1)

    if save_fig:
        from utilities.util import save_figure, timestamp
        from constants import IMAGEREG_OUT_PATH
        fn = f"3d_image_series-{timestamp()}.png"
        save_figure(fig, IMAGEREG_OUT_PATH, fn)

    plt.show()
    plt.close(fig)


def plot_image_series_comparison(image1, image2,
                                 mode='overlay',
                                 cmap1='gist_gray', cmap2='gist_heat',
                                 labels=['Image 1', 'Image 2'],
                                 grid_edge_size=None, save_fig=False):
    """
    Plot a N x N grid of slice images using matplotlib

    Parameters
    ----------
    image1 : sitk.Image
        A sitk.Image vector (3D volume)
    image2 : sitk.Image
        A sitk.Image vector (3D volume)
    mode : str ('overlay' or 'checkerboard')
        How to plot comparison between image1 and image2
    grid_edge_size : int
        The wanted edge length N of the N x N grid of subimages. Defaults to
        the largest edge size possible with the number of subimages to be
        plotted.
    """

    num_slices_image = image1.GetSize()[2]
    N = get_image_grid_edge_size(num_slices_image, grid_edge_size)
    NROWS = N
    NCOLS = N

    img1_array = sitk.GetArrayFromImage(image1)
    img2_array = sitk.GetArrayFromImage(image2)
    #diff_img = np.abs(compare_images(img1_array, img2_array, method='diff'))

    MIN_INTENSITY1 = np.min(img1_array)
    MAX_INTENSITY1 = np.max(img1_array)
    MIN_INTENSITY2 = np.min(img2_array)
    MAX_INTENSITY2 = np.max(img2_array)

    #img1_array = np.divide(img1_array, MAX_INTENSITY)
    #img2_array = np.divide(img2_array, MAX_INTENSITY)

    FIGSIZE = (16, 16)
    IMSHOW_ARGS = {'interpolation': 'nearest'}

    fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS+1, figsize=FIGSIZE)
    fig.suptitle(f"Overlay of 3D images. Using cmaps '{cmap1}' and '{cmap2}'",
                 fontsize=16)
    mid = num_slices_image // 2
    for i in range(NROWS):
        for j in range(NCOLS):
            slice1 = img1_array[mid + i * NCOLS + j]
            slice2 = img2_array[mid + i * NCOLS + j]
            if mode == 'overlay':
                im1 = axes[i, j].imshow(
                    slice1, cmap=cmap1, alpha=0.9,
                    vmin=MIN_INTENSITY1, vmax=MAX_INTENSITY1)
                im2 = axes[i, j].imshow(
                    slice2, cmap=cmap2, alpha=0.2,
                    vmin=MIN_INTENSITY2, vmax=MAX_INTENSITY2)
            elif mode == 'checkerboard':
                img_comparison = compare_images(
                    slice1, slice2, method='checkerboard', n_tiles=(6, 6))
                img = axes[i, j].imshow(img_comparison, cmap=cmap1)
            else:
                raise ValueError("Argument 'mode' must be one of 'overlay' "
                                 "and 'checkerboard'.")
            axes[i, j].set_axis_off()
        axes[i, NCOLS].set_axis_off()

    if mode == 'overlay':
        cb1 = fig.colorbar(im1, ax=axes[0:1, N], shrink=2.0)
        cb1.set_label(labels[0] + ' (HU)', fontsize=18)
        cb2 = fig.colorbar(im2, ax=axes[2:3, N], shrink=2.0)
        cb2.set_label(labels[1] + ' (HU)', fontsize=18)
    else: # 'checkerboard'
        cb = fig.colorbar(img, ax=axes[0:1, N], shrink=2.0)

    #plt.tight_layout(pad=1, w_pad=0.1, h_pad=0.1)

    if save_fig:
        from utilities.util import save_figure, timestamp
        from constants import IMAGEREG_OUT_PATH
        fn = f"3d_image_overlay-{cmap1}-{cmap2}-{timestamp()}.png"
        save_figure(fig, IMAGEREG_OUT_PATH, fn)

    plt.show()
    plt.close(fig)


def get_image_grid_edge_size(num_slices_image, wanted_edge_size=None):

    if (wanted_edge_size is not None and isinstance(wanted_edge_size, int)
            and num_slices_image > np.square(wanted_edge_size)):
        return wanted_edge_size

    # else
    grid_sizes = range(6) # (1, 2, 3, ..., 10)
    max_edge_size = 1
    for n in grid_sizes:
        if num_slices_image > np.square(n):
            max_edge_size = n
        else: break
    return max_edge_size


def plot_deformation_field(deformation_field, slice=None, plot_title=None,
                           save_fig=False):

    logger.info("Plotting deformation field as quiver plot")

    print("\n")

    Nx, Ny, Nz = deformation_field.GetSize()
    x0, y0, z0 = deformation_field.GetOrigin()
    if slice is None:
        slice = Nz // 2

    vectors = sitk.GetArrayFromImage(deformation_field)
    dets = np.zeros(shape=(Nz, Ny, Nx))

    # Calculate deformation vector field for given slice
    for j in range(Ny):
        for k in range(Nx):
            vec = vectors[slice, j, k]
            magnitude = np.sqrt(vec.dot(vec))
            dets[slice, j, k] = magnitude

    print(f"Shape (x, y, z): {Nx} {Ny} {Nz}")
    print(f"Origin (x, y, z): {x0} {y0} {z0}")
    print(vectors.shape)
    print(dets.shape)


    # ============================================
    # Plot quiver and magnitude of deformation
    # ============================================

    vector = vectors[slice]
    #vector = np.divide(vector, np.max(vector)/10)   # Normalize vector
    det = dets[slice]
    #det = np.divide(det, np.max(det))

    X, Y = np.meshgrid(np.arange(0, Nx, 1), np.arange(0, Ny, 1))
    U = vector[:, :, 0] # get x component
    V = vector[:, :, 1] # get y component

    print("U", U.shape)
    print("V", V.shape)

    dx = 16
    fig = plt.figure(figsize=(16,12))
    plt.imshow(det, interpolation='bilinear', cmap='jet')#, vmin=0., vmax=1.)
    plt.colorbar()
    plt.quiver(X[::dx, ::dx], Y[::dx, ::dx], U[::dx, ::dx], V[::dx, ::dx],
               pivot='mid', units='x')
    if plot_title is not None:
        plt.title(plot_title, fontsize=20)
    ax = plt.gca()
    ax.set_axis_off()

    if save_fig:
        from utilities.util import save_figure, timestamp
        from constants import IMAGEREG_OUT_PATH
        save_figure(fig, IMAGEREG_OUT_PATH, f"deformation-{timestamp()}.png")

    plt.show()
    plt.close(fig)

