"""
Created on 16/11/2020
@author: ingridtveten
"""

from enum import Enum



class PROMEnum(Enum):
    """ PROM = Patient Recorded Outcome Measure
        * name : The name/description of the outcome measure
        * value : (str) Question number in the QUFW94 (Fransson et al., 1994)
            questionnaire (Prostate Cancer Symptom Scale)

    Rectum:
        S47: Overall bother from all bowel symptoms
        S48: Stool frequency
        S50: Stool leakage
        S51: Planning of toilet visits
        S58: Limitations in daily activity caused by bowel symptoms
        RECTAL BOTHER SCORE (Tondel, PhD, 2019): The above five items regarding
            intestinal problems represent the rectal bother scale:

    Sexual function:
        SEXUAL_BOTHER "S60"
        SEXUAL_DESIRE "S62"
        HAD_ERECTION "S63A"
        ERECTION_WITHOUT_AID "S63B"
        ERECTION_WITH_AID "S64D"

    Bladder:
        URINARY_BOTHER "S35"
        URINARY_SMART_PAIN "S38"
        URINARY_START_TIME "S39"
        URINARY_LEAKAGE "S40"*
        URINARY_URGENCY "S42"*
        URINARY_BLOOD "S43"
        URINARY_DAILY_ACTIVITY "S45"
    * Important for side-effect (Fransson and Widmark, 1999, CUTTOFF <2.5)
    """

    # Questions relating to rectum
    BOTHER_BOWEL_SYMPTOMS = 'S47'
    STOOL_FREQUENCY = 'S48'
    STOOL_LEAKAGE = 'S50'
    PLANNING_TOILET_VISITS = 'S51'
    LIMIT_DAILY_ACTIVITY = 'S58'
    RECTAL_BOTHER_SCORE = 'RBS'

    # Questions relating to sexual function
    SEXUAL_BOTHER = "S60"
    SEXUAL_DESIRE = "S62"
    HAD_ERECTION = "S63A"
    ERECTION_WITHOUT_AID = "S63B" 
    ERECTION_WITH_AID = "S64D"

    # Questions relating to bladder
    URINARY_BOTHER = "S35"
    URINARY_SMART_PAIN = "S38"
    URINARY_START_TIME = "S39"
    URINARY_LEAKAGE = "S40"
    URINARY_URGENCY ="S42"
    URINARY_BLOOD ="S43"
    URINARY_DAILY_ACTIVITY = "S45"
    URINARY_COMPOSITE_SCORE = "UBS"


RBS_PROMS = [PROMEnum.BOTHER_BOWEL_SYMPTOMS,
             PROMEnum.STOOL_FREQUENCY,
             PROMEnum.STOOL_LEAKAGE,
             PROMEnum.PLANNING_TOILET_VISITS,
             PROMEnum.LIMIT_DAILY_ACTIVITY]

SEXUAL_FUNCTION_PROMS = [PROMEnum.SEXUAL_BOTHER,
                         PROMEnum.SEXUAL_DESIRE,
                         PROMEnum.HAD_ERECTION,
                         PROMEnum.ERECTION_WITHOUT_AID,
                         PROMEnum.ERECTION_WITH_AID]

GU_PROMS = [PROMEnum.URINARY_BOTHER,
            PROMEnum.URINARY_SMART_PAIN,
            PROMEnum.URINARY_START_TIME,
            PROMEnum.URINARY_LEAKAGE,
            PROMEnum.URINARY_URGENCY,
            PROMEnum.URINARY_BLOOD,
            PROMEnum.URINARY_DAILY_ACTIVITY]


def get_patient_outcomes(outcome_enum, months=36):
    """
    Reads PROMs from .xlsx file in the project's 'data' folder and returns data
        for the desired outcome as a dictionary

    Parameters
    ----------
    outcome_enum : PROMEnum
        The PROM for which outcome data should be retrieved

    Returns
    ----------
    proms : dictionary
        Dictionary of outcome scores for the patients, indexed by patient ID
    """
    from fileio.readwritexlsx import read_xlsx_complication_data
    from constants import QOL_FILE_NAME, VERIFIED_FOLLOW_UP, \
        UNVERIFIED_FOLLOW_UP

    if months in VERIFIED_FOLLOW_UP:
        pass
    elif months in UNVERIFIED_FOLLOW_UP:
        print(f"Warning: Using unverified follow-up data for {months} months.")
    else:
        print(f"Error: No follow-up data for {months} months exists. "
              f"Try one of {VERIFIED_FOLLOW_UP}.")
        return None

    filename = QOL_FILE_NAME.format(months=months)
    complication_data = read_xlsx_complication_data(filename)
    proms = {}
    for idx, patient_id in enumerate(complication_data['NR']):

        # Calculate RBS or get outcome item for each patient
        if outcome_enum == PROMEnum.RECTAL_BOTHER_SCORE:
            outcome = calculate_rectal_bother_score(complication_data, idx)
        elif outcome_enum == PROMEnum.URINARY_COMPOSITE_SCORE:
            outcome = calculate_urinary_composite_score(complication_data, idx)
        else:
            outcome = complication_data[outcome_enum.value][idx]

        if outcome is not None:
            proms[patient_id] = outcome
        # Else: Skip patients missing/with invalid outcome measures

    return proms


def get_cumulative_outcomes(patients, outcome_enum, up_to_months=36):
    """
    Reads PROMs from .xlsx file in the project's 'data' folder and returns data
        for the desired outcome as a dictionary

    Parameters
    ----------
    outcome_enum : PROMEnum
        The PROM for which outcome data should be retrieved

    Returns
    ----------
    proms : dictionary
        Dictionary of outcome scores for the patients, indexed by patient ID
    """

    from constants import VERIFIED_FOLLOW_UP

    if up_to_months < min(VERIFIED_FOLLOW_UP):
        print(f"Error: {up_to_months} months is before earliest follow-up "
              f"data. Try one of {VERIFIED_FOLLOW_UP}.")
        return None

    cumulative_proms = {}
    for id in patients:
        cumulative_proms[id] = []

    for months in VERIFIED_FOLLOW_UP:
        if months <= up_to_months:

            # Get PROMs for current month
            proms = get_patient_outcomes(outcome_enum, months)
            if proms is None:
                continue

            # Add proms for each patient to
            for id in patients:
                if id not in proms.keys():
                    cumulative_proms[id].append(-1)
                else:
                    cumulative_proms[id].append(proms[id])

    return cumulative_proms


def calculate_rectal_bother_score(complication_data, idx):
    """
    Parameters
    ----------
    complication_data : dictionary
        Dictionary containing the complication data  for all patients
    idx : int
        Index of the current patient in the complication_data container

    Returns
    -------
    bother_score : float
        Returns rectal bother score (RBS) if patient indexed by 'idx' has all
        RBS measures, otherwise returns None
    """

    bother_score = 0

    for prom in RBS_PROMS:  # RBS_PROMS = list of QUFW94 question no's
        outcome = complication_data[prom.value][idx]
        if outcome == None:
            # Invalid without all 5 measures, break out and return None
            return None
        bother_score += outcome

    bother_score = float(bother_score) / len(RBS_PROMS)
    return bother_score


def calculate_urinary_composite_score(complication_data, idx):
    """
    Parameters
    ----------
    complication_data : dictionary
        Dictionary containing the complication data for all patients
    idx : int
        Index of the current patient in the complication_data container

    Returns
    -------
    score : float
        Returns composite urinary toxicity score if patient indexed by 'idx'
        has all included measures, otherwise returns None
    """

    score = 0

    for prom in GU_PROMS:  # GU_PROMS: List of PROMs defined above
        outcome = complication_data[prom.value][idx]
        if outcome == None:
            # Invalid without all measures, break out and return None
            return None
        score += outcome

    score = float(score) / len(GU_PROMS)
    return score



