from typing import Tuple

import numpy as np
from pandas import DataFrame, Series
from statsmodels.stats.weightstats import _tconfint_generic, _tstat_generic

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import POINT_ESTIMATE, CI_LOWER, CI_UPPER, VARIANCE, TWO_SIDED, SFX1, SFX2, \
    STD_ERR, PREFERENCE_TEST, NULL_HYPOTHESIS, DIFFERENCE
from spotify_confidence.analysis.frequentist.generic_computer import GenericComputer


class TTestComputer(GenericComputer):
    def _variance(self, df: DataFrame) -> Series:
        variance = (
                df[self._numerator_sumsq] / df[self._denominator] -
                df[POINT_ESTIMATE] ** 2)
        if (variance < 0).any():
            raise ValueError('Computed variance is negative. '
                             'Please check your inputs.')
        return variance

    def _add_point_estimate_ci(self, df: DataFrame):
        df[CI_LOWER], df[CI_UPPER] = _tconfint_generic(
            mean=df[POINT_ESTIMATE],
            std_mean=np.sqrt(df[VARIANCE] / df[self._denominator]),
            dof=df[self._denominator] - 1,
            alpha=1-self._interval_size,
            alternative=TWO_SIDED
        )
        return df

    def _dof(self, row):
        v1, v2 = row[VARIANCE + SFX1], row[VARIANCE + SFX2]
        n1, n2 = row[self._denominator + SFX1], row[self._denominator + SFX2]
        return ((v1 / n1 + v2 / n2) ** 2 /
                ((v1 / n1) ** 2 / (n1 - 1) +
                 (v2 / n2) ** 2 / (n2 - 1)))

    def _p_value(self, row) -> float:
        _, p_value = _tstat_generic(value1=row[POINT_ESTIMATE + SFX2],
                                    value2=row[POINT_ESTIMATE + SFX1],
                                    std_diff=row[STD_ERR],
                                    dof=self._dof(row),
                                    alternative=row[PREFERENCE_TEST],
                                    diff=row[NULL_HYPOTHESIS])
        return p_value

    def _ci(self, row, alpha_column: str) -> Tuple[float, float]:
        return _tconfint_generic(
            mean=row[DIFFERENCE],
            std_mean=row[STD_ERR],
            dof=self._dof(row),
            alpha=row[alpha_column],
            alternative=row[PREFERENCE_TEST])

    def _achieved_power(self,
                        df: DataFrame,
                        mde: float,
                        alpha: float) -> DataFrame:
        v1, v2 = df[VARIANCE + SFX1], df[VARIANCE + SFX2]
        n1, n2 = df[self._denominator + SFX1], df[self._denominator + SFX2]

        var_pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)

        power = power_calculation(mde, var_pooled, alpha, n1, n2)

        return (
            df.assign(achieved_power=power)
              .loc[:, ['level_1', 'level_2', 'achieved_power']]
              .reset_index()
        )
