"""
Created on 29/09/2020
@author: ingridtveten
"""

# ======= EXTERNAL IMPORT =========
import logging
import numpy as np
import matplotlib.pyplot as plt
from time import time

# ======= LOCAL IMPORT =========
from utilities.util import StructureEnum, print_structure_stats_header, \
    print_structure_stats, timestamp, get_dvhs_from_file, save_figure, \
    initialize_patient, get_volumes_from_dvh_file
from constants import AVAILABLE_PATIENTS, NO_PATIENT_DATA, DVH_COMPARISON_PATH

from utilities.logger import init_logger, close_logger

logger = logging.getLogger()


def compare_volumes_by_centre():
    from utilities.util import filter_treatment_centre, filter_study_arms
    from scipy.stats import ttest_ind

    # ====== DEFINE WANTED STRUCTURE ======
    PATIENTS = AVAILABLE_PATIENTS
    STRUCT = StructureEnum.PTV_0_70
    ARM = 1    # 1, 2, or 'both'
    # ======================================

    PATIENTS = filter_study_arms(AVAILABLE_PATIENTS, wanted_arm=ARM)

    stolavs = filter_treatment_centre(PATIENTS, wanted_centre='stolavs')
    aalesund = filter_treatment_centre(PATIENTS, wanted_centre='aalesund')

    vol_stolavs = get_volumes_from_dvh_file(STRUCT, patients=stolavs)
    vol_aalesund = get_volumes_from_dvh_file(STRUCT, patients=aalesund)
    # Get volumes as list
    vol_stolavs = [v for v in vol_stolavs.values()]
    vol_aalesund = [v for v in vol_aalesund.values()]

    # Perform t-test
    t_stat, p_value = ttest_ind(vol_stolavs, vol_aalesund,
                                equal_var=False, axis=0, nan_policy='omit')

    N1 = np.size(vol_stolavs)
    N2 = np.size(vol_aalesund)
    N_TOT = N1 + N2
    PERCENT1 = 100 * N1 / N_TOT
    PERCENT2 = 100 * N2 / N_TOT
    MEAN1 = np.mean(vol_stolavs)
    MEAN2 = np.mean(vol_aalesund)

    print(f"\nVolumes for {STRUCT.value} for each treatment centre")
    print("-" * 50)
    print(f"{'Centre':<10}\t{'#':<7}\t{'%':<8}\t{'Mean volume (cm3)':<14}")
    print(f"{'St.Olavs':<10}\t{N1:<7}\t{PERCENT1:<8.1f}\t{MEAN1:<14.1f}")
    print(f"{'Aalesund':<10}\t{N2:<7}\t{PERCENT2:<8.1f}\t{MEAN2:<14.1f}")
    print(f"Volume t-test:\tp = {p_value:.6f}")


#================================================================
#   GENERATE DVH/VOXEL-DOSE FILES
#================================================================

def create_dose_data_per_patient_and_organ():
    """
    For each patient, and for each volume in the StructureEnum class, the DVH
        stats are calculated and printed (format approximately as below).
        Further, the dose-voxel data and DVHs are written to files so computing
        effort is not wasted.

    Example format:
    ---------------
    PatientID   ROI name    Volume  V50    V60     V70     D50     D60  ...
    1           roi1        < values ... >
    1           roi2        < values ... >
    PatientID   ROI name    Volume  V50    V60     V70     D50     D60  ...
    2           roi1        < values ... >
    ...

    """

    from dvhcalculator import calculate_and_write_dvh_to_file

    # ====== Define desired data ======
    patients = AVAILABLE_PATIENTS
    structs = StructureEnum
    # Enter below for only some patients/structures, else comment next two lines
    patients = (1, 3, 4, 5, 6)
    structs = [StructureEnum.RECTAL_MUCOSA]
    # ================================
    t0 = time()
    excluded_no_data = []
    missing_ROI = []

    for patient_id in patients:

        # ======= Initialize patient =======
        if patient_id in NO_PATIENT_DATA:
            print("{:<10}\tData not available/Excluded\n".format(patient_id))
            excluded_no_data.append(patient_id)
            continue

        patient = initialize_patient(patient_id)

        # ========= Calculate DVH and voxel data for each structure =========
        for struct in structs:

            if struct == StructureEnum.VOID:
                continue

            # Calculate DVH. Below function handles certain errors, in
            # which case it prints custom error messages
            # err_patient_id has possible error, store in list to print last
            err_patient_id = calculate_and_write_dvh_to_file(patient, struct)
            if err_patient_id is not None:
                missing_ROI.append(err_patient_id + " " + str(struct.name))

        t = time()
        print("\nDVHs for ID {} took {:.4f} s\n".format(patient_id, t - t0))
        logger.debug("DVHs for ID {} took {:.4f} s".format(patient_id, t - t0))
        t0 = t

    print("Excluded patients:", excluded_no_data)
    print("Check following ROI. Not read unexpectedly:", missing_ROI)
    logger.info("Excluded patients: {}".format(excluded_no_data))
    logger.info("Check following ROI. Not read unexpectedly: {}".format(
        missing_ROI))


#================================================================
#   IMPORT DVH AND DOSE DATA
#================================================================

def read_and_plot_dvh_from_file():
    """
    The function looks for the patient/structure DVH file in the outdata/DVH
        folder and prints a selection of DVH stats.
    """
    from os.path import join
    from fileio.readwritexlsx import read_workbook, xlsx_data_to_dict

    # ====== Define desired data ======
    patients = AVAILABLE_PATIENTS  # For all patients
    structs = StructureEnum  # For all structures

    # Define below for only some structures, else comment out next two lines
    # NOTE: 193 missing rectal mucosa
    patients = (1, 31, 33, 35, 39, 58, 75, 90, 99,
                101, 108, 152, 186, 193, 221, 233)
    structs = [#StructureEnum.RECTAL_WALL,
               StructureEnum.BLADDER, StructureEnum.PENILE_BULB,
               StructureEnum.RECTUM, StructureEnum.RECTAL_MUCOSA
               ]
    # ================================

    COLORS = {1: 'red', 31: 'darkgreen', 33: 'forestgreen', 35: 'orange',
              39: 'darkred', 58: 'hotpink', 75: 'midnightblue', 90: 'blue',
              99: 'indigo', 101: 'firebrick', 108: 'seagreen', 152: 'dimgray',
              186: 'darkgoldenrod', 193: 'gold', 221: 'k', 233: 'cyan'}
    COMPARE_TO_FILE = False
    AVAIL_FOR_FILE_COMPARISON = (StructureEnum.BLADDER, StructureEnum.RECTUM,
        StructureEnum.PENILE_BULB, StructureEnum.RECTAL_MUCOSA)

    for struct in structs:

        if struct == StructureEnum.VOID: continue
        print_structure_stats_header(struct)
        dvhs, volumes = get_dvhs_from_file(struct, patients)

        fig = plt.figure(figsize=(8, 5))

        for patient_id in patients:
            if (patient_id in NO_PATIENT_DATA or patient_id not in dvhs.keys()):
                continue
            if (struct == StructureEnum.RECTAL_MUCOSA and patient_id == 193):
                continue  # ID 193 has no rectal mucosa contour

            cum_dvh = dvhs[patient_id]
            roi_volume = volumes[patient_id]
            print_structure_stats(patient_id, cum_dvh.name, roi_volume, cum_dvh)

            rgb = np.random.rand(3, )
            plt.plot(cum_dvh.bincenters, cum_dvh.counts,
                     label=f'{patient_id} (program)',
                     color=COLORS[patient_id], linestyle='solid')
                     #color=rgb, linestyle='solid')

            if COMPARE_TO_FILE:
                if not struct in AVAIL_FOR_FILE_COMPARISON:
                    continue
                # Get DVH data from TPS (stored in data/test-dvhs)
                path = join('.', join('data', 'test-dvhs'))
                fn = f"{struct.name}-id{patient_id}.xlsx"
                dvh_data = read_workbook(join(path, fn))
                dvh_dict = xlsx_data_to_dict(dvh_data)
                max_vol = dvh_dict['volume'][0]
                print(f"{patient_id:<10}\t{'':<16}\t{max_vol:.1f} (from TPS)")
                plt.plot(dvh_dict['dose'],
                         100 * np.divide(dvh_dict['volume'], max_vol),
                         #label=f'{patient_id} (TPS)',
                         color=COLORS[patient_id], linestyle='dashed')
                         #color=rgb, linestyle='dashed')

        if COMPARE_TO_FILE:
            plt.title(f"Program and TPS DVHs: {struct.name}", fontsize=14)
            plt.legend(loc='best', fontsize=10)
        else:
            plt.title(f"DVHs: {struct.name}", fontsize=14)
            #plt.legend(loc='best', fontsize=11)

        plt.xlabel(f'Dose [{cum_dvh.dose_units}]', fontsize=14)
        plt.ylabel(f'Volume [{cum_dvh.volume_units}]', fontsize=14)
        save_figure(fig,
                    path=join(DVH_COMPARISON_PATH, 'tps-vs-program'),
                    fn=f"{struct.name}-{timestamp()}.png")
        plt.show()


def read_voxel_data_file():
    """
    Test reading of voxel files from outdata/voxeldata folder
    """
    from fileio.readwritedata import read_voxel_data_from_file

    # ====== Define desired data ======
    patients = AVAILABLE_PATIENTS  # For all patients
    structs = StructureEnum    # For all structures
    # Define below for only some structures, else comment out next two lines
    #patients = (10,)
    #structs = [StructureEnum.RECTUM]
    # ================================

    for struct in structs:

        if struct == StructureEnum.VOID: continue
        print_structure_stats_header(struct)

        for patient_id in patients:

            # ========= CALCULATE DVH =========
            try:
                voxel_data = read_voxel_data_from_file(patient_id, struct)
                print("{:<10}\tRead voxel data file for ROI {}.".format(
                    patient_id, struct.name))
            except FileNotFoundError as err:
                print("{:<10}\tNo voxel data file found for ROI {}. Error: {}"
                      "".format(patient_id, struct.name, err))
            except Exception as err:
                print("{:<10}\tCould not get voxel data for ROI {}. {}: {}".format(
                    patient_id, struct.name, err.__class__.__name__, err))


#==================================================
#       BOXPLOT OF DVH CHARACTERISTICS
#==================================================

def get_boxplot_data(struct, dvh_stats):
    """
    Helper function for make_boxplots() below. Reads existing DVH stats from
        file and formats for box plot (labels and corresponding values).

    Parameters
    ----------
    struct : StructureEnum

    dvh_stats : list of str

    """
    from fileio.readwritedata import dvh_from_file


    # ====== INITIALIZE DATA CONTAINER ======
    data = {}
    for key in dvh_stats:
        data[key] = np.array([])

    # ====== READ DVH STATS ======
    for id in AVAILABLE_PATIENTS:
        try:
            dvh, volume = dvh_from_file(id, struct)
        except FileNotFoundError:
            continue

        #dvh = dvh.absolute_dose()
        dvh = dvh.relative_volume
        dvh = dvh.cumulative
        for key in data.keys():
            stat = dvh.statistic(str(key)).value
            data[key] = np.append(data[key], stat)

    # ==== Print output (part of LaTeX table) for project report ====
    print_output_for_report = False
    if print_output_for_report:
        print(struct.value)
        print("\\toprule")
        print(f"DVH measure & {struct.value} \\\\")
        print("\\midrule")
        for key in data.keys():
            dt = data[key]
            if key[0] == 'V':
                print(f"{key} [\%]", end='')
            elif key[0] == 'D':
                print(f"{key} [Gy]", end='')
            print(f" & {np.median(dt):.1f} "
                  f"({np.min(dt):.1f}-{np.max(dt):.1f}) \\\\")
        print("\\bottomrule")
        print("{}: {} subjects".format(struct.value, np.size(data[key])))

    # ==== Format data ====
    labels = []
    arr = []
    for key, val in data.items():
        labels.append(key)
        arr.append(val)

    return labels, arr


def make_boxplots():
    """
    Generates box plots with the DVH data for all available patients for each
        ROI/structure. Used for project report.
    """

    import matplotlib.pyplot as plt
    from utilities.util import save_figure
    from constants import PLOT_PATH, BOXPLOT_FILE_NAME

    # ====== DEFINE WANTED STRUCTURE AND DVH STATS ======
    STRUCTURES = [StructureEnum.CTV_70_78,
                  StructureEnum.PTV_70_78,
                  StructureEnum.RECTUM, #StructureEnum.RECTAL_MUCOSA,
                  StructureEnum.RECTAL_WALL]
    DVH_STATS = ['D2',  # 'D10', 'D20', 'D30', 'D40',
                 'D50',  # 'D60', 'D70', 'D80',
                 'D90', 'D98',
                 #'V10Gy', 'V20Gy', 'V30Gy', 'V40Gy',
                 'V50Gy', 'V60Gy', 'V65Gy',
                 'V70Gy', 'V75Gy', 'V78Gy',  # 'V80Gy',
                 ]

    # ==== Plot settings ====
    title_size = 24
    axis_fontsize = 18
    tick_fontsize = 16
    main_col = 'cornflowerblue' #'indianred'
    other_col = 'indianred'
    black = 'k'
    box_properties = {'patch_artist': True,
                      'widths': 0.75,
                      'boxprops': dict(facecolor=main_col, color=main_col,
                                       linewidth=0),
                      'medianprops': dict(color=black),
                      'whiskerprops': dict(color='k', linewidth=2),
                      'whis': (5, 95),  # Whiskers from 5th to 95th percentile
                      'flierprops': dict(markeredgecolor=black,
                                         markerfacecolor=black,
                                         markersize=10,
                                         marker='x'),
                      #'capprops': dict(color='k'),
                      }
    hlines_options = dict(linestyle='--', linewidth=2, color='r')
    hline_text_optn = dict(fontsize=tick_fontsize, ha='left', va='center')

    # ==== Generate plot ====
    fig, axs = plt.subplots(4, 1, sharex='col', #sharey='row',
                            figsize=(16, 13))

    for ax, struct in zip(axs.flatten(), STRUCTURES):
        labels, data = get_boxplot_data(struct, DVH_STATS)
        print("{}: {} subjects".format(struct.value, np.shape(data)[1]))

        ax.boxplot(data, labels=labels, **box_properties)

        plt.hlines(78.0, xmin=0.5, xmax=4.5, **hlines_options)
        #plt.text(4.55, 78, 'Prescribed dose', **hline_text_optn)

        ax.set_title(struct.value, fontsize=title_size)
        #ax.set_xlabel("DVH characteristic", fontsize=axis_fontsize)
        plt.setp(ax.get_xticklabels(), fontsize=tick_fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)
        ax.set_ylabel("Dose [Gy]", fontsize=axis_fontsize)
        ax.set_ylim(0, 105)

        ax2 = ax.twinx()  # Secondary axis for volume DVH stats
        plt.setp(ax2.get_yticklabels(), fontsize=tick_fontsize)
        ax2.set_ylabel('Volume fraction [%]', fontsize=axis_fontsize)
        ax2.set_ylim(0, 105)

        # ===== Plot dose-constraints reference lines for OARs =====
        if (struct == StructureEnum.RECTUM or
                struct == StructureEnum.RECTAL_MUCOSA or
                struct ==  StructureEnum.RECTAL_WALL):
            ax2.hlines(50.0, xmin=4.5, xmax=6.5, **hlines_options)
            #ax2.text(6.3, 57, 'Volume constraint', **hline_text_optn)


    plt.xlabel("DVH characteristic", fontsize=axis_fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=tick_fontsize)
    #ax.set_ylabel('Dose [Gy]', fontsize=axis_fontsize)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_figure(fig=fig, path=PLOT_PATH,
                fn=BOXPLOT_FILE_NAME.format(time=timestamp()))
    plt.show()




if __name__ == "__main__":
    init_logger("ntcp-model")
    t1 = time()


    """ Run analyses """
    #dvh_analysis_treatment_arms()
    #dvh_analysis_outcome()
    #optimize_lkb_parameters()
    #run_pca()

    #compare_volumes_by_centre()
    from voxelbased.doseanalysis import run_voxel_analysis
    run_voxel_analysis()


    """ Display structure names or numbers """
    #print_structures(id=(129,))
    #print_structure_names(StructureEnum.TESTICLE_SIN, (1, 100))

    """ Read/write DVH and voxel data """
    #create_dose_data_per_patient_and_organ()
    #read_and_plot_dvh_from_file()

    """ Generate plots for report """
    #make_boxplots() # Define wanted structures and DVH statistics in function


    t2 = time()
    time_str = "Process took {:.6f} s".format(t2 - t1)
    print("\n--------------------------\n" + time_str)
    logger.info(time_str)
    close_logger()