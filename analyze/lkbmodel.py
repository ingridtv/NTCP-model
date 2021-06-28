"""
Created on 12/11/2020
@author: ingridtveten

Abbreviations
-------------
NTCP : Normal Tissue Complication Probability
LKB  : Lyman-Kutcher-Burman (model)
gEUD : Generalized Equivalent Uniform Dose (EUD)
TD50 : 50% tolerance dose (dose where 50% experience effect (complications))

LSQ  : Least-squares
MLE  : Maximum likelihood estimation
"""

import logging
from time import time
import numpy as np

from constants import AVAILABLE_PATIENTS
from utilities.util import StructureEnum, timestamp


logger = logging.getLogger()

LKB_INTEGRAL_PREFACTOR = 1/np.sqrt(2 * np.pi)
NTCP_LOWER_LIMIT = 1E-10
NTCP_UPPER_LIMIT = 1 - 1E-10


""" =============================================
* CALCULATION OF NTCP
============================================="""

def lkb_calculate_ntcp(voxel_doses, td50, n, m):

    gEUD = generalized_EUD(voxel_doses, n)
    ntcp = lkb_ntcp_from_gEUD(gEUD, td50, m)
    return ntcp


def lkb_ntcp_from_gEUD(gEUD, td50, m):
    from scipy.integrate import quad

    def ntcp_integrand(x):
        return np.exp(-np.square(x) / 2)
    lower_limit = -np.inf

    if isinstance(gEUD, np.ndarray):
        ntcp_predictions = []
        for eud in gEUD:
            upper_limit = integral_limit_upper(eud, td50, m)
            quad_result = quad(ntcp_integrand, lower_limit, upper_limit)
            integral = quad_result[0]

            ntcp = LKB_INTEGRAL_PREFACTOR * integral
            ntcp_predictions.append(ntcp)
        return ntcp_predictions

    # If gEUD is not ndarray, but e.g. float
    upper_limit = integral_limit_upper(gEUD, td50, m)
    quad_result = quad(ntcp_integrand, lower_limit, upper_limit)
    integral = quad_result[0]

    ntcp = LKB_INTEGRAL_PREFACTOR * integral
    return ntcp


def integral_limit_upper(gEUD, td50, m):
    """
    Calculates the upper integral limit in the cumulative integral in the LKB
        NTCP model

    Parameters:
    -----------
    gEUD : float
        Generalized EUD for volume, as calculated by the gEUD function below
    td50 : int/float
        50% tolerance dose (i.e. dose where 50 % experience complications)
    m : float
        Standard deviation of the probability distribution
    """

    return float(gEUD - td50) / (m * td50)


def generalized_EUD(voxel_doses, n):
    """
    Calculates the gEUD in the LKB model for NTCP from dose-voxel data for ROI

    Parameters:
    -----------
    voxel_doses : ndarray (1D/2D)
        Ndarray containing dose for all voxels in the region of interest (ROI)
    num_voxels : int/float
        Number of voxels in the ROI
    n : float
        Volume weighting parameter (indicates organ seriality)
    """

    if type(voxel_doses[0]) == np.float64:   # Assume array for a single patient
        gEUD = _calculate_gEUD(voxel_doses, n)
        return gEUD

    else:   # Assume more than one patient/set of observations
        gEUDs = []
        for patient_dose in voxel_doses:
            gEUD = _calculate_gEUD(patient_dose, n)
            gEUDs.append(gEUD)

        return np.asarray(gEUDs, dtype=float)


def _calculate_gEUD(voxel_doses, n):
    """
    Calculates the gEUD in the LKB model from dose-voxel data for single
        patient. See: Liu et al. (2010) Acta Oncol., 49(7): 1040–1044

    Parameters:
    -----------
    voxel_doses : ndarray (1D)
        ndarray containing dose for all voxels in the region of interest (ROI)
    n : float
        Volume weighting parameter (indicates organ seriality)
    """

    num_voxels = np.size(voxel_doses)
    relative_voxel_volume = 1 / float(num_voxels)  # v_voxel / V_total
    inverse_n = 1 / float(n)

    gEUD = 0
    for dose in voxel_doses:
        if dose != 0:
            gEUD += relative_voxel_volume * np.power(dose, inverse_n)

    gEUD = np.power(gEUD, n)
    return gEUD



""" =============================================
* OPTIMIZE USING scipy.optimize.curve_fit
============================================="""
def optimize_lkb_parameters_from_voxels(voxel_data, has_complication,
        init_guess=(70, 0.1, 0.1), bounds=([0., 1E-2, 1E-2], [150., 2., 2.])):
    """
    Calculates the optimal LKB model parameters from voxel data

    Parameters:
    -----------
    voxel_data : ndarray (2D)
        Patients are distributed along primary axis, and the corresponding
        voxel_dose arrays are along the secondary axis
    has_complication : ndarray (1D) of int/bool
        Array of booleans indicating whether the patients experienced
        complications or not
    init_guess : size-3 tuple
        Initial guess for the model parameters (TD50, n, m)

    Returns:
    --------
    See documentation for 'optimize_lkb_parameters_from_gEUD' below
        td50 : int/float
        m : float
    """
    from scipy.optimize import curve_fit
    t0 = time()

    params, cov = curve_fit(lkb_calculate_ntcp, voxel_data, has_complication,
                            p0=init_guess, bounds=bounds)
    t = time()
    logger.info("Optimization of LKB parameters took {:.5f} s".format(t-t0))
    logger.info("Params – TD50: {:.8f}, n: {:.8f}, m: {:.8f}".format(*params))

    # ====== FIND MAXIMUM LIKELIHOOD ======
    gEUDs = generalized_EUD(voxel_data, params[1])
    max_log_likelihood = calculate_log_likelihood(
        gEUDs, has_complication, params[0], params[2])
    return params, cov, max_log_likelihood


""" =============================================
* OPTIMIZE USING LEAST SQUARES + BINS/COMPLICATION FREQUENCY
============================================="""
def optimize_lkb_by_least_squares(voxel_data, has_complication):
    t_init = time()
    NUM_SAMPLES = np.size(has_complication)

    td50_values = np.linspace(70., 80., 41)
    n_values = np.geomspace(7E-3, 0.1, 10)
    m_values = np.geomspace(5E-3, 0.2, 20)
    squared_diff = np.zeros(
        shape=(np.size(td50_values), np.size(n_values), np.size(m_values)))

    i, j, k = 0, 0, 0
    least_squares = np.inf
    idx_least_squares = (i, j, k)
    t0 = t_init

    for n in n_values:
        # Calculate gEUDs and put into 1 Gy bins
        gEUDs = generalized_EUD(voxel_data, n)
        bins, frac = group_gEUDs(gEUDs, has_complication,
            min_patients_per_bin=6, #binsize=0.5
            )
        i = 0
        for td50 in td50_values:
            k = 0
            for m in m_values:

                '''Calculate squared diff and compare to current minimum by'''
                # (1) Grouping gEUDs and consider fraction with complication
                #sq_diff = calculate_absolute_square_difference(bins, frac, td50, m)
                # or (2) Minimize abs diff between 0/1 complication and gEUD
                sq_diff = calculate_absolute_square_difference(
                    gEUDs, has_complication, td50, m)

                squared_diff[i, j, k] = sq_diff
                if sq_diff < least_squares:
                    least_squares = sq_diff
                    idx_least_squares = (i, j, k)

                k += 1  # END M LOOP
            i += 1  # END TD50 LOOP

        t = time()
        print("Up to n = {} took {:.3f} s ({:.1f} s in total)".format(
            n_values[j], t-t0, t-t_init))
        t0 = t
        j += 1  # END N LOOP

    t = time()
    logger.info("Optimization of LKB parameters took {:.3f}".format(t-t0))

    # ====== FIND PARAMS MINIMIZING ABS SQ ======
    i = idx_least_squares[0]
    j = idx_least_squares[1]
    k = idx_least_squares[2]
    params = (td50_values[i], n_values[j], m_values[k])
    logger.info(
        "Found optimal parameters - TD50: {}, n: {}, m: {}".format(*params))

    # ====== SAVE PARAMS AND LIKELIHOOD TO FILE ======
    write_lsq_to_file(td50_values, n_values, m_values, squared_diff)

    # ====== PLOT MAX LIKELIHOOD CURVES ======
    from constants import LSQ_PLOT_PATH
    from utilities.util import save_figure, timestamp

    fig = plot_lsq_profiles(
        td50_values, n_values, m_values, squared_diff, idx_least_squares)
    fn = 'least_squares_plot_' + timestamp() + '.png'
    save_figure(fig, LSQ_PLOT_PATH, fn)


    cov = (-1, -1, -1)
    return params, cov, least_squares


def group_gEUDs(gEUDs, has_complication, min_patients_per_bin=10, binsize=1):

    avail_gEUDs = np.arange(np.floor(np.min(gEUDs)),
                            np.ceil(np.max(gEUDs)), binsize)
    NUM_BINS = np.size(avail_gEUDs) - 1

    count = [[] for n in range(NUM_BINS)]   # Track complications for each bin
    frac = [[] for n in range(NUM_BINS)]    # Fraction with complication per bin
    for i in range(NUM_BINS):
        for j in range(np.size(gEUDs)):
            gEUD = gEUDs[j]
            if (gEUD > avail_gEUDs[i]) and (gEUD < avail_gEUDs[i+1]):
                count[i].append(has_complication[j])

        # If >= x patients in the bin, calculate fraction with complications
        num_patients = np.size(count[i])
        if num_patients >= min_patients_per_bin:
            frac[i] = np.sum(count[i]) / np.size(count[i])
        else:
            # Remove/mark this dose bin for exclusion
            frac[i] = -1

    invalid_idx = np.nonzero(np.array(frac) < 0)
    avail_gEUDs = np.delete(avail_gEUDs, invalid_idx)
    frac = np.delete(frac, invalid_idx)
    return avail_gEUDs, frac


def calculate_absolute_square_difference(gEUDs, complication_freq, td50, m):
    """ Calculate log-likelihood of observation for given td50 and m """
    num_observations = np.size(complication_freq)

    sq_diff = 0
    for i in range(num_observations):
        ntcp = lkb_ntcp_from_gEUD(gEUDs[i], td50, m)
        sq_diff += np.square(ntcp - complication_freq[i])

    return sq_diff


def plot_lsq_profiles(td50_values, n_values, m_values, lsq, idx_min_lsq):
    import matplotlib.pyplot as plt

    (i, j, k) = idx_min_lsq

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    subtitle_size = 16
    #yaxis_title = r'$(NTCP_\mathrm{model} - NTCP_\mathrm{observed})^2$'
    yaxis_title = '$(\mathrm{model} - \mathrm{observed})^2$'
    # TD50, index i
    ax1.set_title('TD50', fontsize=subtitle_size)
    ax1.scatter(td50_values, lsq[:, j, k])
    ax1.set_xlim(np.min(td50_values), np.max(td50_values))
    ax1.set_xlabel('TD50')
    ax1.set_ylabel(yaxis_title)
    # n, index j
    ax2.set_title('n', fontsize=subtitle_size)
    ax2.scatter(n_values, lsq[i, :, k])
    ax2.set_xlim(np.min(n_values), np.max(n_values))
    ax2.set_xscale('log')
    ax2.set_xlabel('n')
    ax2.set_ylabel(yaxis_title)
    # m, index k
    ax3.set_title('m', fontsize=subtitle_size)
    ax3.scatter(m_values, lsq[i, j, :])
    ax3.set_xlim(np.min(m_values), np.max(m_values))
    ax3.set_xscale('log')
    ax3.set_xlabel('m')
    ax3.set_ylabel(yaxis_title)

    plt.show()
    return fig


def write_lsq_to_file(td50_values, n_values, m_values, lsq):
    from os.path import join
    from constants import LSQ_PLOT_PATH
    from utilities.util import timestamp

    fn = 'lsq_diff-' + timestamp() + '.txt'
    f = open(file=join(LSQ_PLOT_PATH, fn), mode='w')

    f.write('\nTD50\t')
    for v in td50_values: f.write(str(v) + '\t')
    f.write('\nn\t')
    for v in n_values: f.write(str(v) + '\t')
    f.write('\nm\t')
    for v in m_values: f.write(str(v) + '\t')

    ax_size = np.shape(lsq)
    for i in range(ax_size[0]):
        f.write('\n')
        for j in range(ax_size[1]):
            for k in range(ax_size[2]):
                val = lsq[i, j, k]
                f.write(str(val) + '\t')
            f.write('\n')
    f.close()


""" =============================================
* OPTIMIZE USING MLE (maximize log-likelihood)
============================================="""
def optimize_lkb_by_maximum_likelihood(voxel_data, has_complication):
    t_init = time()

    td50_values = np.linspace(75., 85., 41)
    n_values = np.geomspace(7E-3, 0.05, 8)
    m_values = np.geomspace(5E-3, 0.1, 15)
    likelihood_of_observation = np.zeros(
        shape=(np.size(td50_values), np.size(n_values), np.size(m_values)))

    i, j, k = 0, 0, 0
    max_log_likelihood = -np.inf
    idx_max_likelihood = (i, j, k)
    t0 = t_init
    for n in n_values:
        gEUDs = generalized_EUD(voxel_data, n)

        # Bin gEUDs into 1 Gy bins
        bins, frac = group_gEUDs(gEUDs, has_complication,
                                 min_patients_per_bin=10, #binsize=1
                                 )
        # Perform optimization on these parameters

        i = 0
        for td50 in td50_values:
            k = 0
            for m in m_values:

                # ====== CALCULATE (LOG-)LIKELIHOOD OF OBSERVATION ======
                log_likelihood = calculate_log_likelihood(
                    gEUDs, has_complication, td50, m)
                likelihood_of_observation[i, j, k] = log_likelihood

                # ====== COMPARE TO MAX LIKELIHOOD ======
                if log_likelihood > max_log_likelihood:
                    max_log_likelihood = log_likelihood
                    idx_max_likelihood = (i, j, k)

                k += 1  # END M LOOP
            i += 1  # END TD50 LOOP

        t = time()
        print("Up to n = {} took {:.5f} s ({:.1f} s in total)".format(
            n_values[j], t-t0, t-t_init))
        t0 = t
        j += 1  # END N LOOP

    t = time()
    logger.info("Optimization of LKB parameters took {:.5f} s".format(t-t0))

    # ====== FIND PARAMS MAXIMIZING LIKELIHOOD ======
    max_likelihood = np.max(likelihood_of_observation)
    '''assert (max_likelihood == max_log_likelihood)'''
    i = idx_max_likelihood[0]
    j = idx_max_likelihood[1]
    k = idx_max_likelihood[2]
    params = (td50_values[i], n_values[j], m_values[k])
    logger.info(
        "Found optimal parameters - TD50: {}, n: {}, m: {}".format(*params))

    # ====== SAVE PARAMS AND LIKELIHOOD TO FILE ======
    write_likelihood_to_file(td50_values, n_values, m_values,
                             likelihood_of_observation)

    # ====== PLOT MAX LIKELIHOOD CURVES ======
    from constants import MLE_PLOT_PATH
    from utilities.util import save_figure, timestamp

    fig = plot_likelihood_profiles(td50_values, n_values, m_values,
                            likelihood_of_observation, idx_max_likelihood)
    fn = 'mle_plot_' + timestamp() + '.png'
    save_figure(fig, MLE_PLOT_PATH, fn)


    cov = (-1, -1, -1)
    return params, cov, max_log_likelihood


def calculate_log_likelihood(gEUDs, has_complication, td50, m):
    """ Calculate log-likelihood of observation for given td50 and m """
    num_observations = np.size(has_complication)

    log_likelihood = 0
    for i in range(num_observations):
        ntcp = lkb_ntcp_from_gEUD(gEUDs[i], td50, m)
        ntcp = max(ntcp, NTCP_LOWER_LIMIT)  # To ensure not zero to log
        ntcp = min(ntcp, NTCP_UPPER_LIMIT)  # To ensure not zero to log
        if has_complication[i]:
            log_likelihood += np.log(ntcp)
        else: #no complication
            log_likelihood += np.log(1 - ntcp)

    return log_likelihood


def plot_likelihood_profiles(td50_values, n_values, m_values,
                             likelihood, idx_max_likelihood):
    import matplotlib.pyplot as plt

    (i, j, k) = idx_max_likelihood

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    subtitle_size = 16
    # TD50, index i
    ax1.set_title('TD50', fontsize=subtitle_size)
    ax1.scatter(td50_values, likelihood[:, j, k])
    ax1.set_xlim(np.min(td50_values), np.max(td50_values))
    ax1.set_xlabel('TD50')
    ax1.set_ylabel('LogLikelihood')
    # n, index j
    ax2.set_title('n', fontsize=subtitle_size)
    ax2.scatter(n_values, likelihood[i, :, k])
    ax2.set_xlim(np.min(n_values), np.max(n_values))
    ax2.set_xscale('log')
    ax2.set_xlabel('n')
    ax2.set_ylabel('LogLikelihood')
    # m, index k
    ax3.set_title('m', fontsize=subtitle_size)
    ax3.scatter(m_values, likelihood[i, j, :])
    ax3.set_xlim(np.min(m_values), np.max(m_values))
    ax3.set_xscale('log')
    ax3.set_xlabel('m')
    ax3.set_ylabel('LogLikelihood')

    plt.show()
    return fig


def write_likelihood_to_file(td50_values, n_values, m_values, likelihood):
    from os.path import join
    from constants import MLE_PLOT_PATH
    from utilities.util import timestamp

    fn = 'TEST-mle_likelihoods-' + timestamp() + '.txt'
    f = open(file=join(MLE_PLOT_PATH, fn), mode='w')

    f.write('\nTD50\t')
    for v in td50_values: f.write(str(v) + '\t')
    f.write('\nn\t')
    for v in n_values: f.write(str(v) + '\t')
    f.write('\nm\t')
    for v in m_values: f.write(str(v) + '\t')

    ax_size = np.shape(likelihood)
    for i in range(ax_size[0]):
        f.write('\n')
        for j in range(ax_size[1]):
            for k in range(ax_size[2]):
                val = likelihood[i, j, k]
                f.write(str(val) + '\t')
            f.write('\n')
    f.close()


""" =============================================
* OPTIMIZE USING gEUD (no optimization on n)
============================================="""
def optimize_lkb_parameters_from_gEUD(gEUDs, has_complication):
    """
    Calculates the optimal LKB model parameters from patient gEUDs

    Parameters:
    -----------
    gEUDs : ndarray (1D) of floats
        Array of gEUDs for the patients
    has_complication : ndarray (1D) of int/bool
        Boolean array indicating whether the patients experienced complications
    n : float
        Dose/volume weighting parameter (measure of organ seriality)

    Returns:
    --------
    params : tuple
        Tuple containing values for the below parameters optimized for dataset
        * td50 : int/float
            50% tolerance dose (i.e. dose where 50 % experience complications)
        * m : float
            Standard deviation of the probability distribution
    cov : list (of lists)
        The covariance matrix for the optimized parameters
    """
    from scipy.optimize import curve_fit

    gEUDs = np.asarray(gEUDs, dtype=float)  # In case array is not ndarray
    if not np.size(gEUDs) == np.size(has_complication):
        print("Unmatched sizes for gEUDs ({}) and complication array ({})"
              "".format(np.size(gEUDs), np.size(has_complication)))

    params, cov = curve_fit(lkb_ntcp_from_gEUD, gEUDs, has_complication,
                            p0=(70, 0.1)  # Initial guess for (TD50, m)
                            )
    return params, cov


""" =============================================
* OPTIMIZATION, PARAMETER FITTING
============================================="""

# QUANTEC issue:  Michalski et al., "[...] Radiation-Induced Rectal Injury"
#   n = 0.09 (0.04–0.14)
#   m = 0.13 (0.10–0.17)
#   TD50 = 76.9 (73.7–80.1) Gy
LKB_TD50 = 76.9
LKB_N = 0.09
LKB_M = 0.13

LKB_INIT_GUESS = (76.9, 0.09, 0.13)
LKB_PARAM_BOUNDS = ([0., 7E-3, 5E-3],   # lower bounds
                    [150., 1., 1.])     # upper bounds


def calculate_lkb_ntcp():
    """
    Calculates NTCP value for desired patients and structures, as calculated by
        LKB model with QUANTEC parameters. Can define other parameter values.
    """

    from fileio.readwritedata import read_voxel_data_from_file
    from utilities.util import print_structure_name

    # ======= Define patients and structures ========
    patients = np.arange(191, 211, 1, dtype=int)
    #structs = StructureEnum
    structs = [StructureEnum.RECTUM]#, StructureEnum.RECTAL_MUCOSA, StructureEnum.ANAL_CANAL]
    # ===============================================

    for patient_id in patients:

        for struct in structs:
            if struct == StructureEnum.VOID: continue
            print_structure_name(struct)

            # ==== Use QUANTEC params or define other values ====
            LKB_TD50 = 76.9
            LKB_N = 0.09
            LKB_M = 0.13

            try:
                voxel_data = read_voxel_data_from_file(patient_id, struct)
                voxel_doses = voxel_data['dose']
                gEUD = generalized_EUD(voxel_doses, n=LKB_N)
                ntcp = lkb_ntcp_from_gEUD(gEUD, LKB_TD50, LKB_M)
                print("{:<10}\tNTCP = {:.5f}".format(patient_id, ntcp))

            except FileNotFoundError as err:
                print("{:<10}\tNo voxel data file found for ROI {}. Error: {}"
                      "".format(patient_id, struct.name, err))
            '''except Exception as err:
                print("{:<10}\tCould not get voxel data for ROI {}. {}: {}"
                      "".format(patient_id, struct.name,
                                err.__class__.__name__, err))'''


def optimize_lkb_parameters():
    """
    Optimizes parameters for the LKB model for the patient datasets. Define the
        investigated structure, outcome measure and threshold, and which study
        arm(s) to investigate.
    """

    from complicationdata import PROMEnum, get_patient_outcomes
    from utilities.util import count_patients_exceeding_V50Gy_V60Gy, \
        filter_study_arms, read_voxel_data, sort_voxel_and_complication_data
    import matplotlib.pyplot as plt

    t0 = timestamp()
    logger.info("Running optimize_lkb_parameters at {}".format(t0))

    # =========== PARAMETERS/SETTINGS =============
    struct = StructureEnum.RECTUM
    # OUTCOME: See PROMEnum class for options + description
    OUTCOME = PROMEnum.RECTAL_BOTHER_SCORE
    AFTER_MONTHS = 36
    GRADING_THRESHOLD = 3.5

    # STUDY_ARM: 1, 2 or 'both'
    STUDY_ARM = 'both'
    # OPTIM_METHOD: 'curve_fit' (recommended), 'MLE' or 'LSQ'
    OPTIM_METHOD = 'curve_fit'
    # =============================================
    if (STUDY_ARM==1): ARM = 'A'
    elif (STUDY_ARM==2): ARM = 'B'
    else: ARM = STUDY_ARM

    # Filter only patients in wanted study arm(s)
    patients = filter_study_arms(AVAILABLE_PATIENTS, STUDY_ARM)

    # Read voxel and complication data for patients, and sort/format data
    voxel_data = read_voxel_data(patients, struct)
    complication_data = get_patient_outcomes(OUTCOME, months=AFTER_MONTHS)
    if complication_data is None: exit()

    voxels, outcome = sort_voxel_and_complication_data(
        voxel_data, complication_data, threshold=GRADING_THRESHOLD)

    count_patients_exceeding_V50Gy_V60Gy(voxel_data, struct) # For rectum

    # Print/log number of patients exceeding threshold
    N_SAMPLES = np.size(outcome)
    N_COMPL = np.sum(outcome)
    grading_txt = "Arm:\t\t{}\nOutcome:\t{} >= {}\n" \
        "Rate:\t\t{} / {} = {:.4f} %".format(ARM, OUTCOME.name,
            GRADING_THRESHOLD, N_COMPL, N_SAMPLES, 100 * np.average(outcome))
    print("\n" + grading_txt)
    logger.info(grading_txt)

    # Plot outcomes(0/1) vs gEUD for visual inspection before next calculations
    plt.figure()
    plt.title(f"gEUD vs. NTCP (0/1 for </> cut-off {GRADING_THRESHOLD})"
              f"\nusing QUANTEC parameters, arm = {ARM}")
    gEUDs = generalized_EUD(voxels, 0.11)
    plt.scatter(gEUDs, outcome)
    plt.xlabel('gEUD [Gy]', fontsize=14)
    plt.ylabel('NTCP', fontsize=14)
    plt.show()


    # ========== FIT ALL LKB PARAMETERS ===========

    if OPTIM_METHOD == 'curve_fit':
        params, cov, max_LL = optimize_lkb_parameters_from_voxels(
            voxels, outcome, init_guess=LKB_INIT_GUESS, bounds=LKB_PARAM_BOUNDS)

        std_dev = np.sqrt(np.diag(cov))
        logger.info("Covariance matrix")
        for elem in cov:
            logger.info(elem)

        td50_fit, n_fit, m_fit = params[0], params[1], params[2]
        td50_std, n_std, m_std = std_dev[0], std_dev[1], std_dev[2]

    elif OPTIM_METHOD == 'MLE':
        params, cov, max_LL = optimize_lkb_by_maximum_likelihood(voxels, outcome)
        td50_fit, n_fit, m_fit = params[0], params[1], params[2]
        # cov = (-1, -1, -1)
        td50_std, n_std, m_std = cov[0], cov[1], cov[2]

    elif OPTIM_METHOD == 'LSQ':
        params, cov, lsq = optimize_lkb_by_least_squares(voxels, outcome)
        td50_fit, n_fit, m_fit = params[0], params[1], params[2]
        # cov = (-1, -1, -1)
        td50_std, n_std, m_std = cov[0], cov[1], cov[2]

    else:
        print("'OPTIM_METHOD' must be one of 'curve_fit', 'MLE' or 'LSQ'")
        logger.warning("'OPTIM_METHOD' must be 'curve_fit', 'MLE' or 'LSQ'")
        exit()

    # Print when optimization is completed
    txt_string = "\nFitted (TD50, n, m) with {} for {}, study arm {}"
    print(txt_string.format(OPTIM_METHOD, struct, ARM))
    logger.info(txt_string.format(OPTIM_METHOD, struct, ARM))


    # ========= SAVE/PRINT/PLOT LKB RESULTS ==========
    from utilities.util import save_figure, print_lkb_params, \
        save_lkb_params
    from constants import NTCP_PATH, NTCP_LKB_PLOT_FILE_NAME, \
        NTCP_LKB_TEXT_FILE_NAME

    CONFIDENCE_LEVEL = 0.95   # Means alpha = 0.05 = 5%

    # ========= Print and save parameters =========
    print_lkb_params(td50_fit, td50_std, n_fit, n_std, m_fit, m_std,
                     N_SAMPLES, confidence_level=CONFIDENCE_LEVEL)
    textfile_name = '{}-arm-{}-'.format(OPTIM_METHOD, STUDY_ARM) + \
                    NTCP_LKB_TEXT_FILE_NAME.format(organ=struct.name, time=t0)
    save_lkb_params(td50_fit, td50_std, n_fit, n_std, m_fit, m_std,
                    N_SAMPLES, grading_txt, CONFIDENCE_LEVEL,
                    NTCP_PATH, textfile_name)

    gEUDs = generalized_EUD(voxels, n_fit)
    print("Number of gEUDs above TD50 = {:.1f} Gy: {}/{}".format(
        td50_fit, np.size(np.where(gEUDs > td50_fit)), np.size(gEUDs)))

    # ========= Plot patient data and best fit NTCP =========
    do_plot = True
    do_save = True
    if do_plot:
        fig = plot_lkb_model(voxels, outcome, struct, td50_fit, n_fit, m_fit)
        if do_save:
            plot_filename ='{}-arm-{}-'.format(OPTIM_METHOD, STUDY_ARM) + \
                           NTCP_LKB_PLOT_FILE_NAME.format(organ=struct.name, time=t0)
            save_figure(fig, NTCP_PATH, plot_filename)
        # Save figure BEFORE showing since plt.show() begins object destruction
        plt.show()


def plot_lkb_model(voxel_doses, has_complication, struct,
                   td50_fit, n_fit, m_fit):
    import matplotlib.pyplot as plt
    from constants import ORANGE_RGB, BLUE_RGB

    gEUDs = generalized_EUD(voxel_doses, n_fit)
    if gEUDs.any() > 78: print("gEUD > 78 at {}".format(np.where(gEUDs > 78)))

    bins, frac = group_gEUDs(gEUDs, has_complication, min_patients_per_bin=6,
                             # binsize=1
                             )
    binsize = bins[1]-bins[0]
    bincenters = [bins[i] + 0.5 * binsize for i in range(len(bins)-1)]

    # ===== Calculate curves =====
    min_gEUD_plot = np.min(gEUDs) - 5
    max_gEUD_plot = np.max(gEUDs) + 5
    gEUD_curve = np.linspace(min_gEUD_plot, max_gEUD_plot, 100)
    quantec_ntcp_curve = lkb_curve(gEUD_curve, LKB_TD50, LKB_M)
    fitted_ntcp_curve = lkb_curve(gEUD_curve, td50_fit, m_fit)

    # ===== Plot curves and observations =====
    fig = plt.figure(figsize=(8, 5))
    plt.plot(gEUD_curve, fitted_ntcp_curve, label='NTCP (best fit)',
             linestyle='solid', linewidth=2, color=ORANGE_RGB)  # 'darkorange')
    plt.plot(gEUD_curve, quantec_ntcp_curve, label='NTCP (QUANTEC)',
             linestyle='dashed', linewidth=2, color='darkgray')

    plt.scatter(gEUDs, has_complication, label='Patient outcome',
                marker='x', color=BLUE_RGB, s=75)
    """plt.scatter(bincenters, frac, label='Patient outcome',
                marker='x', color=BLUE_RGB, s=75)"""

    # ===== Helper curves and display settings =====
    plt.hlines(0, min_gEUD_plot, max_gEUD_plot, colors='k')
    plt.xlim(min_gEUD_plot, max_gEUD_plot)
    #plt.ylim(-0.1, 1.1)

    # ===== Plot settings =====
    #plt.title('Lyman-Kutcher-Burman (LKB) model', fontsize=18)
    plt.xlabel('gEUD [Gy]', fontsize=14)
    plt.ylabel('NTCP', fontsize=14)
    plt.legend(loc='center left', fontsize=12)

    return fig


""" =============================================
* HELPER FUNCTIONS
============================================="""

def lkb_curve(gEUD, td50, m):
    ntcp = []
    for dose in gEUD:
        ntcp.append(lkb_ntcp_from_gEUD(dose, td50, m))
    return ntcp


def test_lkb_curve():
    import matplotlib.pyplot as plt

    gEUD = np.linspace(0, 100, 1000)
    NTCP = lkb_curve(gEUD, td50=50, m=0.2)

    plt.figure()
    plt.plot(gEUD, NTCP)
    plt.xlabel("gEUD [Gy]")
    plt.ylabel("NTCP [%]")
    plt.show()


if __name__ == "__main__":

    #test_lkb_curve()

    # See function for setting parameters (structure, outcome, threshold, ...)
    optimize_lkb_parameters()


