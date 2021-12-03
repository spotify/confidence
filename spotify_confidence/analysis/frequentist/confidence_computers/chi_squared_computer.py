from typing import Tuple, Dict

import numpy as np
from pandas import DataFrame, Series
from statsmodels.stats.proportion import proportion_confint, proportions_chisquare, confint_proportions_2indep

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import (
    NUMERATOR,
    DENOMINATOR,
    INTERVAL_SIZE,
    ALPHA,
    POINT_ESTIMATE,
    VARIANCE,
    CI_LOWER,
    CI_UPPER,
    SFX1,
    SFX2,
)


def point_estimate(df: DataFrame, arg_dict: Dict[str, str]) -> float:
    numerator = arg_dict[NUMERATOR]
    denominator = arg_dict[DENOMINATOR]
    if (df[denominator] == 0).any():
        raise ValueError("""Can't compute point estimate: denominator is 0""")
    return df[numerator] / df[denominator]


def variance(df: DataFrame, arg_dict: Dict[str, str]) -> Series:
    variance = df[POINT_ESTIMATE] * (1 - df[POINT_ESTIMATE])
    if (variance < 0).any():
        raise ValueError(f"Computed variance is negative: {variance}. " "Please check your inputs.")
    return variance


def std_err(df: DataFrame, arg_dict: Dict[str, str]) -> Series:
    denominator = arg_dict[DENOMINATOR]
    return np.sqrt(df[VARIANCE + SFX1] / df[denominator + SFX1] + df[VARIANCE + SFX2] / df[denominator + SFX2])


def add_point_estimate_ci(df: DataFrame, arg_dict: Dict[str, str]) -> Series:
    numerator = arg_dict[NUMERATOR]
    denominator = arg_dict[DENOMINATOR]
    interval_size = arg_dict[INTERVAL_SIZE]
    df[CI_LOWER], df[CI_UPPER] = proportion_confint(
        count=df[numerator],
        nobs=df[denominator],
        alpha=1 - interval_size,
    )
    return df


def p_value(row: Series, arg_dict: Dict[str, str]) -> float:
    n1, n2 = arg_dict[NUMERATOR] + SFX1, arg_dict[NUMERATOR] + SFX2
    d1, d2 = arg_dict[DENOMINATOR] + SFX1, arg_dict[DENOMINATOR] + SFX2
    _, p_value, _ = proportions_chisquare(
        count=[row[n1], row[n2]],
        nobs=[row[d1], row[d2]],
    )
    return p_value


def ci(row: Series, arg_dict: Dict[str, str]) -> Tuple[float, float]:
    n1, n2 = arg_dict[NUMERATOR] + SFX1, arg_dict[NUMERATOR] + SFX2
    d1, d2 = arg_dict[DENOMINATOR] + SFX1, arg_dict[DENOMINATOR] + SFX2
    alpha = arg_dict[ALPHA]
    return confint_proportions_2indep(
        count1=row[n2],
        nobs1=row[d2],
        count2=row[n1],
        nobs2=row[d1],
        alpha=row[alpha],
        compare="diff",
        method="wald",
    )


def achieved_power(df: DataFrame, mde: float, alpha: float, arg_dict: Dict[str, str]) -> DataFrame:
    n1, n2 = arg_dict[NUMERATOR] + SFX1, arg_dict[NUMERATOR] + SFX2
    d1, d2 = arg_dict[DENOMINATOR] + SFX1, arg_dict[DENOMINATOR] + SFX2

    pooled_prop = (df[n1] + df[n2]) / (df[d1] + df[d2])
    var_pooled = pooled_prop * (1 - pooled_prop)

    return power_calculation(mde, var_pooled, alpha, df[d1], df[d2])
