"""
Created on 27/01/2021
@author: ingridtveten

TODO: Description...
"""

import numpy as np

from constants import AVAILABLE_PATIENTS, RECTUM_DSC_ABOVE_0_5
from utilities.util import StructureEnum, save_figure, timestamp,\
    get_dvhs_from_file, get_dvh_counts, filter_study_arms,\
    get_wanted_dvh_type, get_registered_dvhs



TOTAL_BINS = 8300+1  # Binsize 0.01 Gy --> 10000 bins = 100 Gy
DOSE_BINS = np.linspace(0, 0.01 * TOTAL_BINS, TOTAL_BINS)


#==================================================
#       ANALYSIS OF DVHs
#==================================================

def dvh_analysis_treatment_arms():
    """
    Compare DVHs from each treatment arm
    """

    # ====== DEFINE WANTED PARAMETERS ======
    PATIENTS = AVAILABLE_PATIENTS   # For all patients
    STRUCT = StructureEnum.PENILE_BULB   # Define structure

    DVH_TYPE = 'cumulative'         # 'differential'/'cumulative' (default)
    DVH_VOLUME_TYPE = 'relative'    # 'absolute'/'relative' (default)
    BINSIZE = 0.1       # set binsize (≥ 0.01 Gy) if 'differential'
    # ======================================

    arm1 = filter_study_arms(PATIENTS, wanted_arm=1)
    arm2 = filter_study_arms(PATIENTS, wanted_arm=2)
    tvs = [StructureEnum.PTV_0_70, StructureEnum.PTV_70_78,
         StructureEnum.CTV_0_70, StructureEnum.CTV_70_78]
    oars = [StructureEnum.RECTUM, StructureEnum.RECTAL_MUCOSA,
         StructureEnum.RECTAL_WALL, StructureEnum.BLADDER,
         StructureEnum.PENILE_BULB]
    for STRUCT in oars:
        dvhs, volumes = get_dvhs_from_file(STRUCT)
        dvhs = get_wanted_dvh_type(
            dvhs, volumes, DVH_TYPE, DVH_VOLUME_TYPE, BINSIZE)
        dose_bins = np.linspace(0, BINSIZE*TOTAL_BINS, TOTAL_BINS)

        arm1_dvhs = []
        arm2_dvhs = []
        volumes_arm1 = []
        volumes_arm2 = []
        for patient_id in dvhs.keys():
            vol = volumes[patient_id]
            if patient_id in arm1:
                arm1_dvhs.append(dvhs[patient_id])
                volumes_arm1.append(vol)
            if patient_id in arm2:
                arm2_dvhs.append(dvhs[patient_id])
                volumes_arm2.append(vol)

        # Perform Student t-test
        t_stat, p_value = check_significance_student_t(arm1_dvhs, arm2_dvhs)

        # Plot p-value on semilog plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogy(dose_bins, p_value)
        plt.xlim(0, 80)
        plt.ylim(1E-2, 1)
        plt.grid()
        plt.xlabel("Dose [Gy]")
        plt.ylabel("p-value")
        plt.title(f"p-value for DVH comparison: {STRUCT.value}")
        plt.show()

        # Generate figure: Plot mean DVH and STD cloud
        avg1, std1 = get_dvh_mean_and_std(arm1_dvhs)
        avg2, std2 = get_dvh_mean_and_std(arm2_dvhs)
        head_title = f"DVHs for {STRUCT.value} for each treatment arm"
        print_arm_comparison_stats(head_title, arm1_dvhs, arm2_dvhs,
                                   volumes_arm1, volumes_arm2)
        fig = plot_dvh_comparison(dose_bins, avg1, std1, avg2, std2,
                                  ['Mean, arm A', 'Mean, arm B'],
                                  p_value, #head_title
                                  )
        # Save figure to .png file
        from os.path import join
        from constants import DVH_COMPARISON_PATH, DVH_ARMS_COMPARISON_NAME
        path = join(DVH_COMPARISON_PATH, 'arms')
        fn = DVH_ARMS_COMPARISON_NAME.format(structure=STRUCT.name, time=timestamp())
        save_figure(fig, path, fn)

        # Show figure AFTER saving to file
        fig.show()


def dvh_analysis_outcome():
    """
    Compare two sets of DVHs based on different outcome, i.e. above/below a
        threshold set for the outcome measure defined by OUTCOME
    """

    from complicationdata import PROMEnum, get_patient_outcomes, RBS_PROMS

    # ====== DEFINE WANTED PARAMETERS ======
    PATIENTS = AVAILABLE_PATIENTS   # For all patients
    STRUCT = StructureEnum.RECTAL_WALL   # Define structure
    STUDY_ARM = 1#'both'      # STUDY_ARM: 1, 2 or 'both'

    # OUTCOME: See PROMEnum class for options or further description
    OUTCOME = PROMEnum.RECTAL_BOTHER_SCORE
    AFTER_MONTHS = 36
    GRADING_THRESHOLD = 4.5

    DVH_TYPE = 'cumulative'       # 'differential'/'cumulative' (default)
    DVH_VOLUME_TYPE = 'relative'    # 'absolute'/'relative' (default)
    BINSIZE = 0.1       # set binsize (≥ 0.01 Gy)
    # ======================================

    ALL_PROMS = [PROMEnum.RECTAL_BOTHER_SCORE] + RBS_PROMS
    #for OUTCOME in [PROMEnum.RECTAL_BOTHER_SCORE] + RBS_PROMS:
    #    PATIENTS = AVAILABLE_PATIENTS

    complication_data = get_patient_outcomes(OUTCOME, months=AFTER_MONTHS)
    if complication_data is None: exit()
    # Filter patients by study arm
    patients = filter_study_arms(PATIENTS, STUDY_ARM)

    # ==== GET DVHs AND COMPLICATION STATUS ====
    #dvhs, volumes = get_registered_dvhs(STRUCT, patients, use_fixed_mask_id=31)
    dvhs, volumes = get_dvhs_from_file(STRUCT, patients)
    dvhs = get_wanted_dvh_type(
        dvhs, volumes, DVH_TYPE, DVH_VOLUME_TYPE, BINSIZE)
    if BINSIZE != 0.01:
        dose_bins = np.linspace(0, BINSIZE*TOTAL_BINS, TOTAL_BINS)
    else:
        dose_bins = DOSE_BINS

    dvhs_no_compl = []
    dvhs_compl = []
    with_compl = []
    without_compl = []
    volume_no_compl = []
    volume_compl = []
    for patient_id in patients:
        if patient_id in complication_data.keys() and patient_id in dvhs.keys():
            dvh = dvhs[patient_id]
            vol = volumes[patient_id]
            if complication_data[patient_id] < GRADING_THRESHOLD:
                dvhs_no_compl.append(dvh)
                without_compl.append(patient_id)
                volume_no_compl.append(vol)
            else:
                dvhs_compl.append(dvh)
                with_compl.append(patient_id)
                volume_compl.append(vol)

    # Perform Student t-test and get mean DVHs
    t_stat, p_value = check_significance_student_t(dvhs_no_compl, dvhs_compl)
    avg, std = get_dvh_mean_and_std(dvhs_no_compl)
    avg_compl, std_compl = get_dvh_mean_and_std(dvhs_compl)


    # ========== GENERATE FIGURES ===========
    from os.path import join
    from constants import DVH_COMPARISON_PATH, DVH_OUTCOME_COMPARISON_NAME
    path = join(DVH_COMPARISON_PATH, 'outcome')

    if STUDY_ARM == 1: arm = 'A'
    elif STUDY_ARM == 2: arm = 'B'
    else: arm = 'both'
    head_title =f"{OUTCOME.name} at {AFTER_MONTHS} months, " \
                f"cutoff={GRADING_THRESHOLD}, arm={arm}"
    # PRINT NUMBER WITH/WITHOUT COMPLICATION
    print_outcome_comparison_stats(head_title, dvhs_compl, dvhs_no_compl,
                                   volume_compl, volume_no_compl)

    # Plot and save mean DVH figure
    COMPL = np.size(dvhs_compl)
    NO_COMPL = np.size(dvhs_no_compl)
    N_TOT = COMPL + NO_COMPL
    fig = plot_dvh_comparison(dose_bins, avg, std, avg_compl, std_compl,
        [f"No complication: {NO_COMPL}/{N_TOT}",
         f"Has complication: {COMPL}/{N_TOT}"], p_value, #head_title
                              )
    fn = DVH_OUTCOME_COMPARISON_NAME.format(study_arm=STUDY_ARM,
            structure=STRUCT.name, outcome=OUTCOME.name,
            threshold=GRADING_THRESHOLD, time=timestamp())
    #save_figure(fig, path, fn)
    # Show figure AFTER saving to file
    fig.show()

    # Plot and save all DVHs figure
    '''fig2 = plot_with_without_complication(
        DOSE_BINS, dvhs_compl, dvhs_no_compl, title=head_title)
    fn2 = f"dvhs_by_outcome-arm_{STUDY_ARM}-{STRUCT.name}-{OUTCOME.name}" \
          f"-{GRADING_THRESHOLD}-{timestamp()}.png"
    save_figure(fig2, path, fn2)
    # Show figure AFTER saving to file
    fig2.show()'''


def compare_initial_and_registered_dvhs():
    """
    Compare DVHs generated from original dose distributions and from images
        that have been registered to a template patient
    """

    # ====== DEFINE WANTED PARAMETERS ======
    PATIENTS = filter_study_arms(AVAILABLE_PATIENTS, 2) #RECTUM_DSC_ABOVE_0_5
    STRUCT = StructureEnum.RECTUM   # Define structure
    FIXED_PATIENT_ID = 31
    COMPARE_TO_TEMPLATE = False
    N = 4   # Number of DVHs to compare and plot

    DVH_TYPE = 'cumulative'         # 'differential'/'cumulative' (default)
    DVH_VOLUME_TYPE = 'relative'    # 'absolute'/'relative' (default)
    BINSIZE = 0.01                  # set binsize (≥ 0.01 Gy) if 'differential'
    # ======================================
    dose_bins = np.linspace(0, BINSIZE*TOTAL_BINS, TOTAL_BINS)

    # Get (1) initial DVHs
    init_dvhs, init_volumes = get_dvhs_from_file(STRUCT, patients=PATIENTS)
    # (2) DVHs using registered dose and template structure
    template_dvhs, template_volumes = get_registered_dvhs(
        STRUCT, patients=PATIENTS, use_fixed_mask_id=FIXED_PATIENT_ID)

    # Read DVHs from warped dose files and contours
    if COMPARE_TO_TEMPLATE:
        other_dvhs = template_dvhs
        other_volumes = template_volumes
    else:
        other_dvhs = init_dvhs
        other_volumes = init_volumes

    # Get DVHs using registered dose distributions and structures
    reg_dvhs, reg_volumes = get_registered_dvhs(STRUCT, patients=PATIENTS)

    # Get wanted DVH type and binsize
    other_dvhs = get_wanted_dvh_type(
        other_dvhs, other_volumes, DVH_TYPE, DVH_VOLUME_TYPE, BINSIZE)
    reg_dvhs = get_wanted_dvh_type(
        reg_dvhs, reg_volumes, DVH_TYPE, DVH_VOLUME_TYPE, BINSIZE)

    patient_ids = []
    registered_dvhs = []
    registered_vol = []
    dvhs = []
    volumes = []
    for patient_id in other_dvhs.keys():
        if patient_id in reg_dvhs.keys(): # Key/patient ID available for both
            patient_ids.append(patient_id)
            registered_dvhs.append(reg_dvhs[patient_id])
            registered_vol.append(reg_volumes[patient_id])
            dvhs.append(other_dvhs[patient_id])
            volumes.append(other_volumes[patient_id])


    # Perform Student t-test
    t_stat, p_value = check_significance_student_t(dvhs, registered_dvhs)

    # Generate figure: Plot mean DVH and STD cloud
    avg1, std1 = get_dvh_mean_and_std(registered_dvhs)
    avg2, std2 = get_dvh_mean_and_std(dvhs)
    head_title = f"Registered DVHs for {STRUCT.value} compared to"
    if COMPARE_TO_TEMPLATE:
        head_title += f"\ntemplate (ID {FIXED_PATIENT_ID})"
        labels = ['Registered DVHs', 'Template DVHs']
    else:
        head_title += f"\nnative DVHs"
        labels = ['Registered DVHs', 'Native DVHs']

    calculate_registration_comparison_stats(head_title,
                                            registered_dvhs, dvhs,
                                            registered_vol, volumes)

    #fig = plot_dvh_comparison(dose_bins, avg1, std1, avg2, std2,
    #                          labels, p_value, head_title)
    fig2 = plot_dvhs_with_id_labels(dose_bins, registered_dvhs[:N],
                                    patient_ids[:N], dvhs2=dvhs[:N],
                                    title=head_title)

    # Save figure to .png file
    from os.path import join
    from constants import DVH_COMPARISON_PATH, DVH_REGISTRATION_COMPARISON_NAME
    path = join(DVH_COMPARISON_PATH, 'registration')

    #fn = DVH_REGISTRATION_COMPARISON_NAME.format(structure=STRUCT.name,
    #                                             time=timestamp())
    #save_figure(fig, path, fn)
    #fig.show()

    fn2 = f"dvh_before_after_registration-{STRUCT.name}-{timestamp()}.png"
    save_figure(fig2, path, fn2)
    fig2.show()


# ======================
#   HELPER FUNCTIONS
# ======================

def get_dvh_mean_and_std(dvh_array):
    """
    Gets the mean DVH and standard deviation from a list of DVHs.

    Parameters
    ----------
    dvh_array : array/list of dicompylercore.dvh.DVH objects
        Array of DVHs for several subjects

    Returns
    -------
    avg : array
        Array of mean DVH counts (volume) for each DVH dose value
    std : array
        Array of standard deviation of DVH counts for each dose value
    """

    # Get DVH counts and pad arrays to same length
    counts_arrays = get_dvh_counts(dvh_array, pad_to=TOTAL_BINS)
    # Calculate mean and STD across patient population
    avg = np.average(counts_arrays, axis=0)
    std = np.std(counts_arrays, axis=0)

    return avg, std


def check_significance_student_t(dvh_group1, dvh_group2):
    """
    Uses two-sided Student T-test to test for significant difference between
        the DVHs for two different patient groups.
    """

    # Get DVH counts and pad arrays to same length
    counts1 = get_dvh_counts(dvh_group1, pad_to=TOTAL_BINS)
    counts2 = get_dvh_counts(dvh_group2, pad_to=TOTAL_BINS)
    t_stat, p_value = indep_student_t(counts1, counts2)
    return t_stat, p_value


def indep_student_t(sample1, sample2):
    """
    Uses two-sided Student T-test to test for significant difference between
        two different samples.
    """
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(sample1, sample2,
                                equal_var=False, axis=0, nan_policy='omit')
    return t_stat, p_value


def print_arm_comparison_stats(description, arm1_dvhs, arm2_dvhs,
                               volumes1, volumes2):
    N1 = np.size(arm1_dvhs)
    N2 = np.size(arm2_dvhs)
    N_TOT = N1 + N2
    PERCENT_ARM1 = 100 * N1 / N_TOT
    PERCENT_ARM2 = 100 * N2 / N_TOT

    print("\n" + description)
    print("-" * 50)
    print(f"{'Arm':<6}{'#':<7}{'%':<8}{'Mean volume (cm3)':<14}")
    print(f"{'A':<6}{N1:<7}{PERCENT_ARM1:<8.1f}{np.mean(volumes1):<14.1f}")
    print(f"{'B':<6}{N2:<7}{PERCENT_ARM2:<8.1f}{np.mean(volumes2):<14.1f}")
    print(f"Volume t-test:\tp = {indep_student_t(volumes2, volumes1)[1]:.4f}")


def calculate_registration_comparison_stats(description, reg_dvhs, other_dvhs,
                                            reg_vol, other_vol):

    print("\n" + description)
    print("-" * 75)

    N1 = np.size(reg_dvhs)
    N2 = np.size(other_dvhs)
    vol_diff = np.abs(np.subtract(reg_vol, other_vol))

    print(f"{'':<14}{'Registered':<12}{'Other':<12}{'Difference':<12}")
    print(f"{'Number':<14}{N1:<12}{N2:<12}{'-':<12}")
    print(f"{'Min [cm3]':<14}{np.min(reg_vol):<12.1f}"
          f"{np.min(other_vol):<12.1f}{np.min(vol_diff):<12.1f}")
    print(f"{'5% [cm3]':<14}{np.percentile(reg_vol, 5):<12.1f}"
          f"{np.percentile(other_vol, 5):<12.1f}{np.percentile(vol_diff, 5):<12.1f}")
    print(f"{'Median [cm3]':<14}{np.median(reg_vol):<12.1f}"
          f"{np.median(other_vol):<12.1f}{np.median(vol_diff):<12.1f}")
    print(f"{'Mean [cm3]':<14}{np.mean(reg_vol):<12.1f}"
          f"{np.mean(other_vol):<12.1f}{np.mean(vol_diff):<12.1f}")
    print(f"{'95% [cm3]':<14}{np.percentile(reg_vol, 95):<12.1f}"
          f"{np.percentile(other_vol, 95):<12.1f}{np.percentile(vol_diff, 95):<12.1f}")
    print(f"{'Max [cm3]':<14}{np.max(reg_vol):<12.1f}"
          f"{np.max(other_vol):<12.1f}{np.max(vol_diff):<12.1f}")

    print(f"Volume t-test:"
          f"\tp = {indep_student_t(reg_vol, other_vol)[1]:.4f}")


    reg_stats = {'min': [], 'mean': [],'max': []}
    init_stats = {'min': [], 'mean': [],'max': []}
    for i in range(N1):
        reg_stats['min'].append(reg_dvhs[i].statistic('D98').value)
        reg_stats['mean'].append(reg_dvhs[i].mean)
        reg_stats['max'].append(reg_dvhs[i].statistic('D2').value)
        init_stats['min'].append(other_dvhs[i].statistic('D98').value)
        init_stats['mean'].append(other_dvhs[i].mean)
        init_stats['max'].append(other_dvhs[i].statistic('D2').value)

    diff_min_dose  = np.abs(np.subtract(reg_stats['min'],  init_stats['min']))
    diff_mean_dose = np.abs(np.subtract(reg_stats['mean'], init_stats['mean']))
    diff_max_dose  = np.abs(np.subtract(reg_stats['max'],  init_stats['max']))
    print(f"\nDVH difference registered vs other contour [Gy]")
    print(f"{'DVH stat':<10}{'Min diff':<10}{'Median':<10}"
          f"{'Mean':<10}{'Max diff':<10}")
    print(f"{'D98%':<10}{np.min(diff_min_dose):<10.3f}"
          f"{np.median(diff_min_dose):<10.3f}{np.mean(diff_min_dose):<10.3f}"
          f"{np.max(diff_min_dose):<10.3f}")
    print(f"{'D_mean':<10}{np.min(diff_mean_dose):<10.3f}"
          f"{np.median(diff_mean_dose):<10.3f}{np.mean(diff_mean_dose):<10.3f}"
          f"{np.max(diff_mean_dose):<10.3f}")
    print(f"{'D2%':<10}{np.min(diff_max_dose):<10.3f}"
          f"{np.median(diff_max_dose):<10.3f}{np.mean(diff_max_dose):<10.3f}"
          f"{np.max(diff_max_dose):<10.3f}")



def print_outcome_comparison_stats(description, dvhs_compl, dvhs_no_compl,
                                   vol_compl, vol_no_compl):
    COMPL = np.size(dvhs_compl)
    NO_COMPL = np.size(dvhs_no_compl)
    N_TOT = COMPL + NO_COMPL
    PERCENT_WITH = 100 * COMPL / N_TOT
    PERCENT_WITHOUT = 100 * NO_COMPL / N_TOT

    print("\n" + description)
    print("-" * 50)
    print(f"{'Complication?':<16}{'#':<6}{'%':<7}{'Mean volume (cm3)':<14}")
    print(f"{'With':<16}{COMPL:<6}{PERCENT_WITH:<7.1f}"
          f"{np.mean(vol_compl):<14.1f}")
    print(f"{'Without':<16}{NO_COMPL:<6}{PERCENT_WITHOUT:<7.1f}"
          f"{np.mean(vol_no_compl):<14.1f}")
    print(f"Volume t-test:\tp = "
          f"{indep_student_t(vol_no_compl, vol_compl)[1]:.4f}")


def plot_dvh_comparison(dose_values, avg1, std1, avg2, std2,
                        group_labels, p_values, head_title=None):
    """
    Plot mean DVH and STD cloud to compare two groups
    """
    import matplotlib.pyplot as plt

    MIN_X = 0
    MAX_X = 80  #dose_values[-1]
    FS_TITLE = 18     # title fontsize
    FS_AX_LABEL = 18  # axis label fontsize
    FS_AXTICKS = 14   # tick marks fontsize
    FS_LEGEND = 15    # legend fontsize

    fig = plt.figure(figsize=(8, 5))
    if head_title is not None:
        plt.title(head_title, fontsize=FS_TITLE)
    ax = plt.gca()

    # Plot mean and STD cloud on primary axis
    mean1 = ax.plot(dose_values, avg1, label=group_labels[0],
                   linestyle='-', c='b', linewidth=1)
    ax.fill_between(dose_values, avg1 - std1, avg1 + std1,
                    color='b', alpha=0.2)
    mean2 = ax.plot(dose_values, avg2, label=group_labels[1],
                   linestyle='-', c='r', linewidth=1)
    ax.fill_between(dose_values, avg2 - std2, avg2 + std2,
                    color='r', alpha=0.2)
    ax.set_xlabel("Dose [Gy]", fontsize=FS_AX_LABEL)
    ax.set_ylabel("Volume [%]", fontsize=FS_AX_LABEL)
    ax.tick_params(axis="x", labelsize=FS_AXTICKS)
    ax.tick_params(axis="y", labelsize=FS_AXTICKS)
    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(0, 105)

    # Plot p-value (and reference line at p=0.05) on secondary axis
    ax2 = ax.twinx()
    p_val = ax2.plot(dose_values, p_values, label='p-value', c='gray')
    ax2.axhline(0.05, 0, MAX_X, linestyle='--', c='k')
    ax2.set_ylabel("p-value", fontsize=FS_AX_LABEL)
    ax2.tick_params(axis="y", labelsize=FS_AXTICKS)
    ax2.set_xlim(MIN_X, MAX_X)
    ax2.set_ylim(0, 1.05)

    # Add all axes to legend
    plots = mean1 + mean2 + p_val
    labels = [p.get_label() for p in plots]
    #ax.legend(plots, labels, fontsize=FS_LEGEND)#, loc=6)

    return fig


def plot_with_without_complication(
        dose_values, dvhs_with_compl, dvhs_without_compl, title=None):

    import matplotlib.pyplot as plt
    from utilities.util import get_dvh_counts

    MAX_X = 80  #dose_values[-1]
    FS_TITLE = 18     # title fontsize
    FS_AX_LABEL = 16  # axis label fontsize
    FS_LEGEND = 14    # legend fontsize

    LINE_COMPL = dict(linestyle='-', linewidth=1, c='red')
    LINE_NO_COMPL = dict(linestyle='-', linewidth=1, c='k')

    counts_with = get_dvh_counts(dvhs_with_compl, pad_to=TOTAL_BINS)
    counts_without = get_dvh_counts(dvhs_without_compl, pad_to=TOTAL_BINS)


    # ======= PLOT FIGURE ========
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Plot DVHs with and without complication
    for dvh in counts_without:
        ax.plot(dose_values, dvh, **LINE_NO_COMPL)
    for dvh in counts_with:
        ax.plot(dose_values, dvh, **LINE_COMPL)

    # Adjust plot settings and appearance
    ax.set_xlabel("Dose [Gy]", fontsize=FS_AX_LABEL)
    ax.set_ylabel("Volume [%]", fontsize=FS_AX_LABEL)
    ax.set_xlim(0, MAX_X)
    #ax.set_ylim(0, 110)
    if title is not None:
        plt.title(title, fontsize=FS_TITLE)

    # Generate legend
    from matplotlib.lines import Line2D
    N_COMPL = np.size(dvhs_with_compl)
    N_NO_COMPL = np.size(dvhs_without_compl)
    N_TOT = N_COMPL + N_NO_COMPL

    labels = ["No complication: {}/{}".format(N_NO_COMPL, N_TOT),
              "Has complication: {}/{}".format(N_COMPL, N_TOT)]
    legend_elements = [Line2D([], [], label=labels[0], **LINE_NO_COMPL),
                       Line2D([], [], label=labels[1], **LINE_COMPL)]
    ax.legend(handles=legend_elements, fontsize=FS_LEGEND)#, loc=6)

    return fig


def plot_dvhs_with_id_labels(dose_values, dvhs, patient_ids,
                             title=None, dvhs2=None):

    import matplotlib.pyplot as plt
    from utilities.util import get_dvh_counts

    MIN_X = 0
    MAX_X = 80  #dose_values[-1]
    FS_TITLE = 18     # title fontsize
    FS_AX_LABEL = 18  # axis label fontsize
    FS_AXTICKS = 14   # tick marks fontsize
    FS_LEGEND = 15    # legend fontsize

    counts = get_dvh_counts(dvhs, pad_to=TOTAL_BINS)
    if dvhs2 is not None:
        counts2 = get_dvh_counts(dvhs2, pad_to=TOTAL_BINS)


    # ======= PLOT FIGURE ========
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Plot DVHs with and without complication
    for i, dvh in enumerate(counts):
        ax.plot(dose_values, dvh, label=str(patient_ids[i]) + ' (transformed)')
        c = plt.gca().lines[-1].get_color()
        if dvhs2 is not None:
            ax.plot(dose_values, counts2[i], c=c, linestyle='dashed',
                    label=patient_ids[i])

    # Adjust plot settings and appearance
    ax.set_xlabel("Dose [Gy]", fontsize=FS_AX_LABEL)
    ax.set_ylabel("Volume [%]", fontsize=FS_AX_LABEL)
    ax.set_xlim(0, MAX_X)
    ax.set_ylim(0, 110)
    if title is not None:
        plt.title(title, fontsize=FS_TITLE)

    ax.legend(fontsize=FS_LEGEND)#, loc=6)
    ax.tick_params(axis="x", labelsize=FS_AXTICKS)
    ax.tick_params(axis="y", labelsize=FS_AXTICKS)

    return fig



if __name__ == "__main__":
    from time import time
    t1 = time()

    #dvh_analysis_treatment_arms()
    dvh_analysis_outcome()
    #compare_initial_and_registered_dvhs()


    t2 = time()
    time_str = "Process took {:.6f} s".format(t2 - t1)
    print("\n--------------------------\n" + time_str)
