from typing import Any, Tuple, Union

import numpy as np
from pandas import DataFrame, Series
from statsmodels.stats.weightstats import _tconfint_generic, _tstat_generic

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import (
    CI_LOWER,
    CI_UPPER,
    DENOMINATOR,
    DIFFERENCE,
    INTERVAL_SIZE,
    NULL_HYPOTHESIS,
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    POINT_ESTIMATE,
    PREFERENCE_TEST,
    SFX1,
    SFX2,
    STD_ERR,
    TWO_SIDED,
    VARIANCE,
)


def point_estimate(df: DataFrame, **kwargs: Any) -> float:
    numerator = kwargs[NUMERATOR]
    denominator = kwargs[DENOMINATOR]
    if (df[denominator] == 0).any():
        raise ValueError("""Can't compute point estimate: denominator is 0""")
    return df[numerator] / df[denominator]


def variance(df: DataFrame, **kwargs: Any) -> float:
    numerator = kwargs[NUMERATOR]
    denominator = kwargs[DENOMINATOR]
    numerator_sumsq = kwargs[NUMERATOR_SUM_OF_SQUARES]
    binary = df[numerator_sumsq] == df[numerator]
    if binary.all():
        # This equals row[POINT_ESTIMATE]*(1-row[POINT_ESTIMATE]) when the data is binary,
        # and also gives a robust fallback in case it's not
        variance = df[numerator_sumsq] / df[denominator] - df[POINT_ESTIMATE] ** 2
    else:
        variance = (df[numerator_sumsq] - np.power(df[numerator], 2) / df[denominator]) / (df[denominator] - 1)
    if (variance < 0).any():
        raise ValueError("Computed variance is negative. Please check your inputs.")
    return variance


def std_err(df: DataFrame, **kwargs: Any) -> Series:
    denominator = kwargs[DENOMINATOR]
    return np.sqrt(df[VARIANCE + SFX1] / df[denominator + SFX1] + df[VARIANCE + SFX2] / df[denominator + SFX2])


def add_point_estimate_ci(df: DataFrame, **kwargs: Any) -> DataFrame:
    denominator = kwargs[DENOMINATOR]
    interval_size = kwargs[INTERVAL_SIZE]
    df[CI_LOWER], df[CI_UPPER] = _tconfint_generic(
        mean=df[POINT_ESTIMATE],
        std_mean=np.sqrt(df[VARIANCE] / df[denominator]),
        dof=df[denominator] - 1,
        alpha=1 - interval_size,
        alternative=TWO_SIDED,
    )
    return df


def _dof(df: DataFrame, **kwargs: Any) -> float:
    denominator = kwargs[DENOMINATOR]
    v1, v2 = df[VARIANCE + SFX1], df[VARIANCE + SFX2]
    n1, n2 = df[denominator + SFX1], df[denominator + SFX2]
    return (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))


def p_value(df: DataFrame, **kwargs: Any) -> Series:
    _, p_value = _tstat_generic(
        value1=df[POINT_ESTIMATE + SFX2],
        value2=df[POINT_ESTIMATE + SFX1],
        std_diff=df[STD_ERR],
        dof=_dof(df, **kwargs),
        alternative=df[PREFERENCE_TEST].values[0],
        diff=df[NULL_HYPOTHESIS],
    )
    return p_value


def ci(df: DataFrame, alpha_column: str, **kwargs: Any) -> Tuple[Series, Series]:
    return _tconfint_generic(
        mean=df[DIFFERENCE],
        std_mean=df[STD_ERR],
        dof=_dof(df, **kwargs),
        alpha=df[alpha_column],
        alternative=df[PREFERENCE_TEST].values[0],
    )


def achieved_power(df: DataFrame, mde: float, alpha: float, **kwargs: Any) -> Union[int, float]:
    v1, v2 = df[VARIANCE + SFX1], df[VARIANCE + SFX2]
    d1, d2 = kwargs[DENOMINATOR] + SFX1, kwargs[DENOMINATOR] + SFX2
    n1, n2 = df[d1], df[d2]

    var_pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)

    return power_calculation(mde, var_pooled, alpha, n1, n2)
