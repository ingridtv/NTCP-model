"""
Created on 05/10/2020
@author: ingridtveten
"""

import pydicom
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from patient import Patient
from utilities.util import DoseGroup, timestamp, save_figure
from constants import CT_MIN_INTENSITY, CT_MAX_INTENSITY, DOSE_MIN, DOSE_MAX,\
    PLOT_PATH

import logging
logger = logging.getLogger(__name__)



def plot_contours(patient, slice, dose_group=None):
    """ Plot the contours that are present in the specified slice """
    ct_loc = patient.sorted_ct_data[dose_group][slice]['location_sort']

    for roi_id in patient.rt_structures[dose_group]:

        this_roi = patient.rt_structures[dose_group][roi_id]
        #if this_roi.name == "Rectal mucosa":
        roi_slice = this_roi.get_slice(ct_loc)
        if roi_slice is None: continue  # ROIStructure.get_slice yielded None
        plt.plot(roi_slice.x, roi_slice.y,
                 c=roi_slice.color, label=roi_slice.name)

    plt.legend(loc='upper right', fontsize=10)


def plot_ct_colormap(patient, slice, dose_group=None, cbar=True):
    """ Plot the CT image as a plt.pcolormesh """
    if dose_group is None:
        dose_group = DoseGroup.TOTAL

    ct_slice = patient.sorted_ct_data[dose_group][slice]
    x = ct_slice['x']
    y = ct_slice['y']
    xx, yy = np.meshgrid(x, y)

    # Plotting
    ct_image = plt.pcolormesh(xx, yy, ct_slice['image_sort'],
                              cmap='gray', shading='auto',
                              vmin=CT_MIN_INTENSITY, vmax=CT_MAX_INTENSITY)
    if cbar:
        ct_colorbar = plt.colorbar(ct_image, shrink=0.9)
        ct_colorbar.set_label('CT grayscale', fontsize=18)

    # x & y axis limited by CT. y-axis flipped in image ((0,0) is top left)
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.max(y), np.min(y))


def plot_dose_matrix(dose_matrix, slice, cbar=True):
    """
    :param dose_data: Dictionary with (z, y, x) coordinates and 3D dose matrix
        (with z-major axis)
    """

    dose_slice = dose_matrix.dose[slice]
    x = dose_matrix.x
    y = dose_matrix.y
    xx, yy = np.meshgrid(x, y)

    # Plotting
    dose_image = plt.pcolormesh(xx, yy, dose_slice,
                                cmap='hot', alpha=0.5, shading='gouraud',
                                vmin=DOSE_MIN, vmax=DOSE_MAX)
    if cbar:
        dose_colorbar = plt.colorbar(dose_image, shrink=0.9)
        dose_colorbar.set_label('Dose [Gy]', fontsize=18)


def plot_slice(patient, slice_no, dose_group=None,
               ct=True, roi=True, colorbar=True):
    """
    TODO: Docstring
    """

    plot_doses = True
    if dose_group is None:
        plot_doses = False
        dose_group = DoseGroup.TOTAL

    if colorbar: fig = plt.figure(figsize=(11, 8))
    else: fig = plt.figure(figsize=(9, 8))

    # Plot CT, ROI and/or dose
    if ct: plot_ct_colormap(patient, slice_no, dose_group, cbar=colorbar)
    if roi: plot_contours(patient, slice_no, dose_group)
    if plot_doses:
        dose_matrix = patient.get_dose_matrix(dose_group)
        #dose_matrix = patient.get_masked_dose_matrix(roi_number=20)
        plot_dose_matrix(dose_matrix, slice_no, cbar=colorbar)

    # Set title and display figure
    plt.title("Patient ID: {}\nDoseGroup: {}\nSlice: {}".format(
        patient.patient_id, dose_group.name, slice_no), fontsize=16)

    ax = plt.gca()
    ax.set_axis_off()
    ax.set_aspect(1/1)  # dy/dx
    # Save figure
    path = join(PLOT_PATH, str(patient.patient_id))
    if plot_doses:
        if dose_matrix.masked:
            masked_by = dose_matrix.masked_by.lower()
            filename = f"dose_RIC{patient.get_id()}_" \
                       f"{masked_by.strip()}_" \
                       f"{dose_group.name.lower()}_{timestamp()}.png"
        else:
            filename = f"dose_RIC{patient.get_id()}_{dose_group.name.lower()}_"\
                       f"{timestamp()}.png"
    else:
        filename = f"ct_roi_RIC{patient.get_id()}_{dose_group.name.lower()}_"\
                   f"{timestamp()}.png"
    save_figure(fig=fig, path=path, fn=filename)
    plt.show()


def plot_total_dose(patient_id, slices=(70, 71)):
    # Initialize patient data
    patient = Patient(patient_id)
    # Plot slice(s)
    for slice in slices:
        plot_slice(patient, slice, dose_group=DoseGroup.TOTAL)


def plot_masks(patient_id, roi_id, slices=(70, 71)):
    # Initialize Patient
    patient = Patient(patient_id)
    roi_name = patient.get_roi_name(roi_id)

    for slice in slices:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #fig.suptitle(f"Patient ID: {patient.get_id()}, Slice: {slice}",
        #             fontsize=16)

        #ax1.set_title("UNMASKED")
        #ax2.set_title(f"MASKED BY {roi_name.upper()}")
        matrix = patient.get_dose_matrix()
        xx, yy = np.meshgrid(matrix.x, matrix.y)

        kw_dose =  {'cmap': 'hot', 'shading': 'gouraud', 'alpha': 0.5,
                    'vmin': DOSE_MIN, 'vmax': DOSE_MAX}
        ext_matrix = patient.get_masked_dose_matrix(roi_number=1)
        plt.sca(ax1)
        plot_ct_colormap(patient, slice, cbar=False)
        img1 = ax1.pcolormesh(xx, yy, ext_matrix.get_dose_plane_number(slice),
            **kw_dose)
        masked_matrix = patient.get_masked_dose_matrix(roi_id)
        plt.sca(ax2)
        plot_ct_colormap(patient, slice, cbar=False)
        img2 = ax2.pcolormesh(xx, yy,
            masked_matrix.get_dose_plane_number(slice), **kw_dose)

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cb1 = fig.colorbar(img1, cax=cax)
        cb1.ax.set_ylabel('Dose [Gy]', fontsize=18)
        cb1.ax.tick_params(labelsize=16)
        # cb2 = fig.colorbar(img2, ax=ax2)

        for ax in (ax1, ax2):
            ax.set_aspect(1/1) #dy/dx
            ax.set_axis_off()
            #ax.invert_yaxis()
            ax.set_ylim(220, -50)
        plt.show()

        path = join(PLOT_PATH, str(patient.get_id()))
        filename = f"roi_mask_RIC{patient.get_id()}_total_{timestamp()}.png"
        save_figure(fig=fig, path=path, fn=filename)


def dose_and_mask_plot(patient_id=6):
    roi_id = 7  # RIC 10: 4=bladder, 7=rectum, 19=PTV70-78

    slice_min = 70
    slice_max = 72
    slice_arr = np.arange(slice_min, slice_max, 1, dtype=int)

    #plot_total_dose(patient_id, slices=slice_arr)
    plot_masks(patient_id, roi_id, slices=slice_arr)



# ===========================
#       TEST FUNCTIONS
#============================

def test_read_plot_ct():
    ds = pydicom.dcmread(
      "/Users/ingridtveten/Workspace/NTCP-modell/data/testdata/test_ct.dcm"
    )
    p = plt.imshow(ds.pixel_array, cmap='gist_gray')#, vmin=0, vmax=2000)
    plt.colorbar(p)
    plt.show()


def test_plot_total_dose():
    # Initialize patient data
    patient = Patient(10)
    # Plot slice
    slice = 60
    plot_slice(patient, slice, dose_group=DoseGroup.TOTAL, roi=False)


def test_plot_ct(patient_id):
    # Initialize patient data
    patient = Patient(patient_id)
    # Plot slice
    for slice in range(65, 75):
        plot_slice(patient, slice, roi=True)






if __name__ == "__main__":
    from time import time
    t1 = time()

    #plot_total_dose(patient_id=10, slices=(70,))

    #test_read_plot_ct()
    #test_plot_ct(patient_id=10)
    test_plot_total_dose()

    #dose_and_mask_plot(patient_id=10)

    #patient = Patient(6)
    #plot_slice(patient, 65, ct=True, roi=True, colorbar=False)


    t2 = time()
    print("\n--------------------------\nProcess took {:.6f} s".format(t2-t1))
    logger.info("Process took {} s".format(t2-t1))


