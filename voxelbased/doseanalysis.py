"""
Created on 21/05/2021
@author: ingridtveten

TODO: Description...
"""

import os
import numpy as np
import utilities.util as util
import voxelbased.util as voxelutil



def get_dose_distributions(patients, template_patient_id):

    import utilities.util as util

    dose_images = {}

    try:
        template_dose = util.get_registered_dose_from_file(
            template_patient_id, is_fixed_patient=True)
        reference_metadata = voxelutil.get_image_metadata(template_dose)
        dose_images[template_patient_id] = template_dose
        print(f"ID {template_patient_id:<6}\tGot dose distribution")
    except FileNotFoundError as err:
        print(f"ID {template_patient_id:<6}\tDose file does not exist."
              )
    except Exception as err:
        print(f"ID {template_patient_id:<6}\tAn error occurred. "
              f"{err.__class__.__name__}: {err}")

    for patient_id in patients:
        try:
            if patient_id != template_patient_id:
                registered_dose = util.get_registered_dose_from_file(
                    patient_id, is_fixed_patient=False)
                metadata = voxelutil.get_image_metadata(registered_dose)
                metadata_is_equal = voxelutil.metadata_is_equal(
                    metadata, reference_metadata)
                if metadata_is_equal:
                    dose_images[patient_id] = registered_dose
                    print(f"ID {patient_id:<6}\tGot dose distribution")
                else:
                    print(f"ID {patient_id:<6}\t"
                          f"Image metadata does not match. Cannot compare.")
            else:
                print(f"ID {patient_id:<6}\tAlready got dose for template")
        except FileNotFoundError as err:
            print(f"ID {patient_id:<6}\tDose file does not exist."
                  )
        except Exception as err:
            print(f"ID {patient_id:<6}\tAn error occurred. "
                  f"{err.__class__.__name__}: {err}")

    return dose_images


def per_voxel_ttest(dose_with_compl, dose_without_compl):
    """
    
    Parameters
    ----------
    dose_with_compl, dose_without_compl : ndarray (4D)
        First dimension is along each sample. Each sample has a 3D dose matrix
        representing the 3 spatial dimensions
    dose_without_compl : ndarray (4D)
        See dose_with_compl above

    Returns
    -------


    """

    from scipy import stats

    shape1 = np.shape(dose_with_compl)
    shape2 = np.shape(dose_without_compl)
    N_with    = shape1[0]
    N_without = shape2[0]
    spatial_dim1 = shape1[1:]
    spatial_dim2 = shape2[1:]
    if spatial_dim1 != spatial_dim2: print("Dimension mismatch when t-testing")

    t_stats = np.zeros(shape=spatial_dim1, dtype=float)
    p_value = np.zeros(shape=spatial_dim1, dtype=float)
    nan_mask = np.zeros(shape=spatial_dim1, dtype=float)

    for i in range(spatial_dim1[0]):
        for j in range(spatial_dim1[1]):
            for k in range(spatial_dim1[2]):

                sample1 = dose_with_compl[:, i, j, k]
                sample2 = dose_without_compl[:, i, j, k]
                t, p = stats.ttest_ind(sample1, sample2, equal_var=False,
                                       nan_policy='propagate')
                if np.isnan(t) or np.isnan(p):
                    nan_mask[i, j, k] = 1   # Mask 'nan' as 1
                    #print("Got 'nan' when performing t-test")

                t_stats[i, j, k] = t
                p_value[i, j, k] = p

        print(f"Slice {i+1} of {spatial_dim1[0]}")


    return t_stats, p_value, nan_mask




def run_voxel_analysis():

    import SimpleITK as sitk
    import matplotlib.pyplot as plt

    import constants
    from complicationdata import PROMEnum, RBS_PROMS, get_patient_outcomes, \
        get_cumulative_outcomes

    # ====== DEFINE WANTED PARAMETERS ======
    FIXED_PATIENT_ID = 31

    PATIENTS = constants.RECTUM_DSC_ABOVE_0_5
    STRUCT = util.StructureEnum.RECTUM   # Define structure

    # OUTCOME: See PROMEnum class for options or further description
    OUTCOME = PROMEnum.RECTAL_BOTHER_SCORE
    AFTER_MONTHS = 36
    GRADING_THRESHOLD = 2.0

    WRITE_P_VALUE_AS_NIFTI = False
    # ======================================


    complication_data = get_patient_outcomes(OUTCOME, months=AFTER_MONTHS)
    if complication_data is None: exit()
    compl_avail = []
    for i in complication_data.keys():
        if i in PATIENTS: compl_avail.append(complication_data[i])

    """cumulative_compl = get_cumulative_outcomes(PATIENTS, OUTCOME,
                                               up_to_months=36)
    if cumulative_compl is None: exit()
    compl_avail = []
    for id in cumulative_compl.keys():
        if id in PATIENTS:
            cumulative_proms = cumulative_compl[id]
            prom = max(cumulative_proms)
            compl_avail.append(prom)"""

    N_tot = len(compl_avail)
    N_with = len(np.nonzero(np.asarray(compl_avail) >= GRADING_THRESHOLD)[0])
    print_outcome_stats(f"{OUTCOME.name} >= {GRADING_THRESHOLD} after "
                        f"{AFTER_MONTHS} months\n"
                        f"Number with/without complications",
                        N_with, N_tot - N_with)


    mask_image = util.get_mask_from_file(
        STRUCT, FIXED_PATIENT_ID, is_fixed_patient=True)
    mask_array = sitk.GetArrayFromImage(mask_image)
    img_spacing = mask_image.GetSpacing()
    Nx, Ny, Nz = mask_image.GetSize()

    dose_images = get_dose_distributions(PATIENTS, FIXED_PATIENT_ID)
    dose_arrays = {}
    for id, dose_with in dose_images.items():
        dose_arrays[id] = sitk.GetArrayFromImage(dose_with)


    #for AFTER_MONTHS in constants.VERIFIED_FOLLOW_UP:
    #for OUTCOME in RBS_PROMS:

    complication_data = get_patient_outcomes(OUTCOME, months=AFTER_MONTHS)
    if complication_data is None: exit()
    compl_avail = []
    for i in complication_data.keys():
        if i in PATIENTS: compl_avail.append(complication_data[i])

    N_tot = len(compl_avail)
    N_with = len(np.nonzero(np.asarray(compl_avail) >= GRADING_THRESHOLD)[0])
    print_outcome_stats(f"{OUTCOME.name} >= {GRADING_THRESHOLD} after "
                        f"{AFTER_MONTHS} months\n"
                        f"Number with/without complications",
                        N_with, N_tot-N_with)

    with_compl = []
    dose_compl = []
    without_compl = []
    dose_no_compl = []
    for patient_id in PATIENTS:
        if patient_id in complication_data.keys() \
                and patient_id in dose_arrays.keys():

            dose = dose_arrays[patient_id]
            #dose = np.ma.masked_array(dose, mask=~mask_array)

            if complication_data[patient_id] < GRADING_THRESHOLD:
                without_compl.append(patient_id)
                dose_no_compl.append(dose)
            else:
                with_compl.append(patient_id)
                dose_compl.append(dose)

    dose_no_compl = np.asarray(dose_no_compl)
    dose_compl = np.asarray(dose_compl)


    from scipy import stats
    t_stat, p_value = stats.ttest_ind(dose_compl, dose_no_compl,
                                      axis=0, equal_var=False)

    pval_image = sitk.GetImageFromArray(p_value)
    pval_image.SetOrigin(mask_image.GetOrigin())
    pval_image.SetSpacing(mask_image.GetSpacing())
    pval_image.SetDirection(mask_image.GetDirection())

    out_path = os.path.join(constants.EXTERNAL_STORAGE_OUT_PATH, 'VBA')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    if WRITE_P_VALUE_AS_NIFTI:
        filename = f"dose-p_val-{OUTCOME.name}-month{AFTER_MONTHS}-" \
                   f"{GRADING_THRESHOLD}-{util.timestamp()}.nii"
        sitk.WriteImage(pval_image, os.path.join(out_path, filename))


    # ======================================================
    #   Plot axial/sagittal plane: Dose comparison, t-test
    # ======================================================
    mean_with    = np.mean(dose_compl, axis=0)
    mean_without = np.mean(dose_no_compl, axis=0)
    diff = np.subtract(mean_with, mean_without)

    fig_ax = plot_axial(
        mean_with, mean_without, diff, mask_array, p_value, img_spacing,
        N_tot, N_with, slice_no=65,
        plot_title=f"{OUTCOME.name} at {AFTER_MONTHS} months, "
                   f"cutoff = {GRADING_THRESHOLD}, N_tot = {N_tot}")
    fig_sag = plot_sagittal(
        mean_with, mean_without, diff, mask_array, p_value, img_spacing,
        N_tot, N_with,
        plot_title=f"{OUTCOME.name} at {AFTER_MONTHS} months, "
                   f"cutoff = {GRADING_THRESHOLD}, N_tot = {N_tot}")

    t = util.timestamp()
    ax_filename = f"dose-p_val-{OUTCOME.name}-month{AFTER_MONTHS}-" \
                  f"{GRADING_THRESHOLD}-{t}-ax.png"
    sag_filename = f"dose-p_val-{OUTCOME.name}-month{AFTER_MONTHS}-" \
                   f"{GRADING_THRESHOLD}-{t}-sag.png"
    util.save_figure(fig_ax, path=out_path, fn=ax_filename)
    util.save_figure(fig_sag, path=out_path, fn=sag_filename)

    #fig_ax.show()
    fig_sag.show()
    plt.close(fig_ax)
    plt.close(fig_sag)



    # ======================================================
    #   Quantify p-value (volume of regions with p<0.05 or
    #   p<0.01, mean dose (w+w/o), mean dose difference
    # ======================================================

    mask_as_bool = mask_array.astype(dtype=bool)
    p_value_rectum = np.ma.masked_array(p_value, mask=~mask_as_bool)
    print(f"\np-value in {STRUCT.value}")
    print(f"Min  = {np.min(p_value_rectum):.4f}")
    print(f"Mean = {np.mean(p_value_rectum):.4f}")
    print(f"Max  = {np.max(p_value_rectum):.4f}")

    masked_mask_array = np.ma.masked_array(mask_array, mask=~mask_as_bool)
    mask_volume = util.roi_volume_cm3(masked_mask_array,
                                      px_spacing=img_spacing)
    print(f"\nWhole {STRUCT.value} mask volume = {mask_volume:.2f}")

    SIGNIFICANCE_LEVELS = (0.01, 0.05)

    p_val_masks = {}
    print("\nDose/volume stats for each p-value subregion\n"
          f"{'p-value':<10}{'Volume [cm3]':<14}"
          f"{'Mean with [Gy]':<16}"
          f"{'Mean without [Gy]':<20}"
          f"{'Dose diff [Gy]':<16}")
    for p_level in SIGNIFICANCE_LEVELS:
        p_val_masks[p_level] = np.zeros(shape=(Nz, Ny, Nx)).astype(dtype=bool)
        p_mask = p_value < p_level
        p_val_masks[p_level] = p_mask

        struct_pval_mask = np.logical_and(p_mask, mask_as_bool)
        mean_with_masked = np.ma.masked_array(mean_with, mask=~struct_pval_mask)
        mean_without_masked = np.ma.masked_array(mean_without, mask=~struct_pval_mask)
        diff_dose_masked = np.ma.masked_array(diff, mask=~struct_pval_mask)

        print(f"{p_level:<10}"
              f"{util.roi_volume_cm3(struct_pval_mask, img_spacing):<14.3f}"
              f"{np.mean(mean_with_masked):<16.3f}"
              f"{np.mean(mean_without_masked):<20.3f}"
              f"{np.mean(diff_dose_masked):<16.3f}")
        p_val_masks[p_level] = struct_pval_mask

        p_fig = plot_sagittal(
            mean_with_masked, mean_without_masked, diff_dose_masked,
            mask_array, struct_pval_mask, img_spacing, N_tot, N_with,
            plot_title=f"p < {p_level}")
        pval_filename = f"p_val_{p_level}-{OUTCOME.name}-month{AFTER_MONTHS}-" \
                      f"{GRADING_THRESHOLD}-{t}.png"
        util.save_figure(p_fig, path=out_path, fn=pval_filename)
        #p_fig.show()
        plt.close(p_fig)







DOSE_KW = {'cmap': 'hot', 'vmin': 0, 'vmax': 80, 'alpha': 1.0}
DIFF_KW = {'cmap': 'seismic', 'vmin': -20, 'vmax': 20, 'alpha': 1.0}
#PVAL_KW = {'cmap': 'hot_r', 'vmin': 0, 'vmax': 0.25}
PVAL_KW = {'cmap': 'hot_r', 'vmin': 0, 'vmax': 0.05, 'alpha': 1.0}
MASK_KW = {'colors': 'blue', 'levels': [0.999], 'linewidths': 2}
MASK_KW2 = {'colors': 'k',   'levels': [0.999], 'linewidths': 2}
#CBAR_KW = {'orientation': 'horizontal', 'shrink': 0.8}
CBAR_KW = {'orientation': 'vertical', 'shrink': 1.0}

def plot_slices(fig, axes,
                plot_with, plot_without, plot_diff, plot_pval,
                N_tot, N_with):

    # ax1 - top left - with complications
    ax1 = axes[0, 0]
    ax1.set_title(f"With complication (N = {N_with})", fontsize=20)
    dose_with = ax1.imshow(plot_with, **DOSE_KW)
    fig.colorbar(dose_with, ax=ax1, label='Dose [Gy]', **CBAR_KW)

    # ax2 - top right - without complications
    ax2 = axes[0, 1]
    ax2.set_title(f"Without complication (N = {N_tot - N_with})", fontsize=20)
    dose_without = ax2.imshow(plot_without, **DOSE_KW)
    fig.colorbar(dose_without, ax=ax2, label='Dose [Gy]', **CBAR_KW)

    # ax3 - bottom left - dose difference
    ax3 = axes[1, 0]
    ax3.set_title(r'$D_{\mathrm{diff}} = '
                  r'(D_{\mathrm{with}} - D_{\mathrm{without}})$', fontsize=20)
    diff = ax3.imshow(plot_diff, **DIFF_KW)
    # diff = ax3.imshow(abs_diff[SLICE], **diff_kw)
    fig.colorbar(diff, ax=ax3, label='Dose diff [Gy]', **CBAR_KW)

    # ax4 - bottom right - p-values
    ax4 = axes[1, 1]
    ax4.set_title('p-value', fontsize=20)
    pval = ax4.imshow(plot_pval, **PVAL_KW)
    fig.colorbar(pval, ax=ax4, label='p-value', **CBAR_KW)


    return fig, (ax1, ax2, ax3, ax4)


def plot_sagittal(mean_with, mean_without, diff, mask_array, p_value,
                  image_spacing, N_tot, N_with, plot_title=None,
                  slice_no=None):

    import matplotlib.pyplot as plt

    Nx = np.shape(mean_with)[-1]  # last dimension gives Nx
    if slice_no is None or slice_no < 0 or slice_no >= Nx:
        PLOT_SLICE = Nx // 2
    else:
        PLOT_SLICE = slice_no


    # Get data (slices) for plotting
    plot_with = mean_with[:, :, PLOT_SLICE]
    plot_without = mean_without[:, :, PLOT_SLICE]
    plot_diff = diff[:, :, PLOT_SLICE]
    plot_mask = mask_array[:, :, PLOT_SLICE]
    plot_pval = p_value[:, :, PLOT_SLICE]

    # Plot figure
    fig, axes = plt.subplots(figsize=(14, 10), nrows=2, ncols=2)

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=24)
    fig, axes = plot_slices(fig, axes,
                            plot_with, plot_without, plot_diff, plot_pval,
                            N_tot, N_with)

    dx, dy, dz = image_spacing
    for ax in axes:
        # Plot contour mask as overlay
        if ax != 2:
            ax.contour(plot_mask, **MASK_KW)
        else: # Use MASK_KW2 with black contour since colormap is red/blue
            ax.contour(plot_mask, **MASK_KW2)

        ax.set_aspect(aspect=float(dz) / dy)
        #ax.set_xlim(60, 410)
        #ax.set_ylim(20, 100)
        ax.set_xlim(200, 350)
        ax.set_ylim(40, 90)

    return fig


def plot_axial(mean_with, mean_without, diff, mask_array, p_value,
               image_spacing, N_tot, N_with, plot_title=None,
               slice_no=None):

    import matplotlib.pyplot as plt

    Nx = np.shape(mean_with)[-1]  # last dimension gives Nx
    if slice_no is None or slice_no < 0 or slice_no >= Nx:
        PLOT_SLICE = Nx // 2
    else:
        PLOT_SLICE = slice_no

    # Get data (slices) for plotting
    plot_with    = mean_with[PLOT_SLICE, :, :]
    plot_without = mean_without[PLOT_SLICE, :, :]
    plot_diff    = diff[PLOT_SLICE, :, :]
    plot_mask    = mask_array[PLOT_SLICE, :, :]
    plot_pval    = p_value[PLOT_SLICE, :, :]

    # Plot figure
    fig, axes = plt.subplots(figsize=(14, 10), nrows=2, ncols=2)

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=24)
    fig, axes = plot_slices(fig, axes,
                            plot_with, plot_without, plot_diff, plot_pval,
                            N_tot, N_with)

    dx, dy, dz = image_spacing
    for ax in axes:
        # Plot contour mask as overlay
        if ax != 2:
            ax.contour(plot_mask, **MASK_KW)
        else: # Use MASK_KW2 with black contour since colormap is red/blue
            ax.contour(plot_mask, **MASK_KW2)

        ax.set_aspect(aspect=float(dy)/dx)
        #ax.set_xlim(100, 400)
        ax.set_ylim(350, 80)

    return fig



def print_outcome_stats(description, N_with, N_without):
    N_TOT = N_with + N_without
    PERCENT_WITH = 100 * N_with / N_TOT
    PERCENT_WITHOUT = 100 * N_without / N_TOT

    print("\n" + description)
    print("-" * 50)
    print(f"{'Complication?':<16}{'#':<6}{'%':<7}")
    print(f"{'With':<16}{N_with:<6}{PERCENT_WITH:<7.1f}")
    print(f"{'Without':<16}{N_without:<6}{PERCENT_WITHOUT:<7.1f}")
    print()



if __name__ == "__main__":
    from time import time
    t0 = time()


    run_voxel_analysis()


    t = time()
    time_str = f"Process took {t - t0:.6f} s"
    print(f"\n\n--------------------------\n{time_str}")

