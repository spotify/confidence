from typing import Tuple

import numpy as np
from pandas import DataFrame, Series
from statsmodels.stats.proportion import proportion_confint, proportions_chisquare, confint_proportions_2indep

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import POINT_ESTIMATE, VARIANCE, CI_LOWER, CI_UPPER, SFX1, SFX2


class ChiSquaredComputer(object):
    def __init__(self, numerator, numerator_sumsq, denominator, ordinal_group_column, interval_size):
        self._numerator = numerator
        self._numerator_sumsq = numerator_sumsq
        self._denominator = denominator
        self._ordinal_group_column = ordinal_group_column
        self._interval_size = interval_size

    def _point_estimate(self, row: Series) -> float:
        if row[self._denominator] == 0:
            raise ValueError("""Can't compute point estimate: denominator is 0""")
        return row[self._numerator] / row[self._denominator]

    def _variance(self, row: Series) -> float:
        variance = row[POINT_ESTIMATE] * (1 - row[POINT_ESTIMATE])
        if variance < 0:
            raise ValueError("Computed variance is negative. " "Please check your inputs.")
        return variance

    def _std_err(self, row: Series) -> float:
        return np.sqrt(
            row[VARIANCE + SFX1] / row[self._denominator + SFX1] + row[VARIANCE + SFX2] / row[self._denominator + SFX2]
        )

    def _add_point_estimate_ci(self, row: DataFrame) -> Series:
        row[CI_LOWER], row[CI_UPPER] = proportion_confint(
            count=row[self._numerator],
            nobs=row[self._denominator],
            alpha=1 - self._interval_size,
        )
        return row

    def _p_value(self, row: Series) -> float:
        _, p_value, _ = proportions_chisquare(
            count=[row[self._numerator + SFX1], row[self._numerator + SFX2]],
            nobs=[row[self._denominator + SFX1], row[self._denominator + SFX2]],
        )
        return p_value

    def _ci(self, row: Series, alpha_column: str) -> Tuple[float, float]:
        return confint_proportions_2indep(
            count1=row[self._numerator + SFX2],
            nobs1=row[self._denominator + SFX2],
            count2=row[self._numerator + SFX1],
            nobs2=row[self._denominator + SFX1],
            alpha=row[alpha_column],
            compare="diff",
            method="wald",
        )

    def _achieved_power(self, df: DataFrame, mde: float, alpha: float) -> DataFrame:
        s1, s2 = df[self._numerator + SFX1], df[self._numerator + SFX2]
        n1, n2 = df[self._denominator + SFX1], df[self._denominator + SFX2]

        pooled_prop = (s1 + s2) / (n1 + n2)
        var_pooled = pooled_prop * (1 - pooled_prop)

        return power_calculation(mde, var_pooled, alpha, n1, n2)
