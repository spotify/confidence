from typing import Tuple, Dict

import numpy as np
from pandas import DataFrame, Series
from statsmodels.stats.weightstats import _tconfint_generic, _tstat_generic

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import (
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    DENOMINATOR,
    INTERVAL_SIZE,
    POINT_ESTIMATE,
    CI_LOWER,
    CI_UPPER,
    VARIANCE,
    TWO_SIDED,
    SFX1,
    SFX2,
    STD_ERR,
    PREFERENCE_TEST,
    NULL_HYPOTHESIS,
    DIFFERENCE,
)


def point_estimate(df: DataFrame, arg_dict: Dict[str, str]) -> float:
    numerator = arg_dict[NUMERATOR]
    denominator = arg_dict[DENOMINATOR]
    if (df[denominator] == 0).any():
        raise ValueError("""Can't compute point estimate: denominator is 0""")
    return df[numerator] / df[denominator]


def variance(df: DataFrame, arg_dict: Dict[str, str]) -> float:
    numerator = arg_dict[NUMERATOR]
    denominator = arg_dict[DENOMINATOR]
    numerator_sumsq = arg_dict[NUMERATOR_SUM_OF_SQUARES]
    binary = df[numerator_sumsq] == df[numerator]
    if binary.all():
        # This equals row[POINT_ESTIMATE]*(1-row[POINT_ESTIMATE]) when the data is binary,
        # and also gives a robust fallback in case it's not
        variance = df[numerator_sumsq] / df[denominator] - df[POINT_ESTIMATE] ** 2
    else:
        variance = (df[numerator_sumsq] - np.power(df[numerator], 2) / df[denominator]) / (df[denominator] - 1)
    if (variance < 0).any():
        raise ValueError("Computed variance is negative. " "Please check your inputs.")
    return variance


def std_err(df: DataFrame, arg_dict: Dict[str, str]) -> Series:
    denominator = arg_dict[DENOMINATOR]
    return np.sqrt(df[VARIANCE + SFX1] / df[denominator + SFX1] + df[VARIANCE + SFX2] / df[denominator + SFX2])


def add_point_estimate_ci(df: DataFrame, arg_dict: Dict[str, str]) -> Series:
    denominator = arg_dict[DENOMINATOR]
    interval_size = arg_dict[INTERVAL_SIZE]
    df[CI_LOWER], df[CI_UPPER] = _tconfint_generic(
        mean=df[POINT_ESTIMATE],
        std_mean=np.sqrt(df[VARIANCE] / df[denominator]),
        dof=df[denominator] - 1,
        alpha=1 - interval_size,
        alternative=TWO_SIDED,
    )
    return df


def _dof(row: Series, arg_dict: Dict[str, str]) -> float:
    denominator = arg_dict[DENOMINATOR]
    v1, v2 = row[VARIANCE + SFX1], row[VARIANCE + SFX2]
    n1, n2 = row[denominator + SFX1], row[denominator + SFX2]
    return (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))


def p_value(df: Series, arg_dict: Dict[str, str]) -> Series:
    _, p_value = _tstat_generic(
        value1=df[POINT_ESTIMATE + SFX2],
        value2=df[POINT_ESTIMATE + SFX1],
        std_diff=df[STD_ERR],
        dof=_dof(df, arg_dict),
        alternative=df[PREFERENCE_TEST].values[0],
        diff=df[NULL_HYPOTHESIS],
    )
    return p_value


def ci(df: DataFrame, alpha_column: str, arg_dict: Dict[str, str]) -> Tuple[Series, Series]:
    return _tconfint_generic(
        mean=df[DIFFERENCE],
        std_mean=df[STD_ERR],
        dof=_dof(df, arg_dict),
        alpha=df[alpha_column],
        alternative=df[PREFERENCE_TEST].values[0],
    )


def achieved_power(df: DataFrame, mde: float, alpha: float, arg_dict: Dict[str, str]) -> DataFrame:
    v1, v2 = df[VARIANCE + SFX1], df[VARIANCE + SFX2]
    d1, d2 = arg_dict[DENOMINATOR] + SFX1, arg_dict[DENOMINATOR] + SFX2
    n1, n2 = df[d1], df[d2]

    var_pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)

    return power_calculation(mde, var_pooled, alpha, n1, n2)
