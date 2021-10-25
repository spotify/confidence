from typing import Tuple

from pandas import DataFrame, Series
from statsmodels.stats.proportion import proportion_confint, proportions_chisquare, confint_proportions_2indep

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import POINT_ESTIMATE, CI_LOWER, CI_UPPER, SFX1, SFX2
from spotify_confidence.analysis.frequentist.generic_computer import GenericComputer


class ChiSquaredComputer(GenericComputer):
    def _variance(self, df: DataFrame) -> Series:
        variance = df[POINT_ESTIMATE] * (1 - df[POINT_ESTIMATE])
        if (variance < 0).any():
            raise ValueError('Computed variance is negative. '
                             'Please check your inputs.')
        return variance

    def _add_point_estimate_ci(self, df: DataFrame):
        df[CI_LOWER], df[CI_UPPER] = proportion_confint(
            count=df[self._numerator],
            nobs=df[self._denominator],
            alpha=1-self._interval_size,
        )
        return df

    def _p_value(self, row):
        _, p_value, _ = (
            proportions_chisquare(count=[row[self._numerator + SFX1],
                                         row[self._numerator + SFX2]],
                                  nobs=[row[self._denominator + SFX1],
                                        row[self._denominator + SFX2]])
        )
        return p_value

    def _ci(self, row, alpha_column: str) -> Tuple[float, float]:
        return confint_proportions_2indep(
            count1=row[self._numerator + SFX2],
            nobs1=row[self._denominator + SFX2],
            count2=row[self._numerator + SFX1],
            nobs2=row[self._denominator + SFX1],
            alpha=row[alpha_column],
            compare='diff',
            method='wald'
        )

    def _achieved_power(self,
                        df: DataFrame,
                        mde: float,
                        alpha: float) -> DataFrame:
        s1, s2 = df[self._numerator + SFX1], df[self._numerator + SFX2]
        n1, n2 = df[self._denominator + SFX1], df[self._denominator + SFX2]

        pooled_prop = (s1 + s2) / (n1 + n2)
        var_pooled = pooled_prop * (1 - pooled_prop)

        power = power_calculation(mde, var_pooled, alpha, n1, n2)

        return (
            df.assign(achieved_power=power)
              .loc[:, ['level_1', 'level_2', 'achieved_power']]
              .reset_index()
        )
