from typing import Tuple, Dict

import numpy as np
from pandas import DataFrame, Series
from statsmodels.stats.proportion import proportion_confint, proportions_chisquare, confint_proportions_2indep

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import (
    NUMERATOR,
    DENOMINATOR,
    INTERVAL_SIZE,
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


def p_value(df: DataFrame, arg_dict: Dict[str, str]) -> Series:
    n1, n2 = arg_dict[NUMERATOR] + SFX1, arg_dict[NUMERATOR] + SFX2
    d1, d2 = arg_dict[DENOMINATOR] + SFX1, arg_dict[DENOMINATOR] + SFX2

    def p_value_row(row):
        _, p_value, _ = proportions_chisquare(
            count=[row[n1], row[n2]],
            nobs=[row[d1], row[d2]],
        )
        return p_value

    return df.apply(p_value_row, axis=1)


def ci(df: DataFrame, alpha_column: str, arg_dict: Dict[str, str]) -> Tuple[Series, Series]:
    n1, n2 = arg_dict[NUMERATOR] + SFX1, arg_dict[NUMERATOR] + SFX2
    d1, d2 = arg_dict[DENOMINATOR] + SFX1, arg_dict[DENOMINATOR] + SFX2
    return confint_proportions_2indep(
        count1=df[n2],
        nobs1=df[d2],
        count2=df[n1],
        nobs2=df[d1],
        alpha=df[alpha_column],
        compare="diff",
        method="wald",
    )


def achieved_power(df: DataFrame, mde: float, alpha: float, arg_dict: Dict[str, str]) -> DataFrame:
    n1, n2 = arg_dict[NUMERATOR] + SFX1, arg_dict[NUMERATOR] + SFX2
    d1, d2 = arg_dict[DENOMINATOR] + SFX1, arg_dict[DENOMINATOR] + SFX2

    pooled_prop = (df[n1] + df[n2]) / (df[d1] + df[d2])
    var_pooled = pooled_prop * (1 - pooled_prop)

    return power_calculation(mde, var_pooled, alpha, df[d1], df[d2])
