from typing import Tuple

import numpy as np
from pandas import DataFrame, Series
from scipy.stats import norm
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic

from spotify_confidence.analysis.confidence_utils import power_calculation, _search_MDE_binary_local_search, \
    _get_hypothetical_treatment_var
from spotify_confidence.analysis.constants import POINT_ESTIMATE, CI_LOWER, CI_UPPER, VARIANCE, TWO_SIDED, SFX2, SFX1, \
    STD_ERR, PREFERENCE, NULL_HYPOTHESIS, DIFFERENCE, NIM
from spotify_confidence.analysis.frequentist.generic_computer import GenericComputer


class ZTestComputer(GenericComputer):
    def _variance(self, df: DataFrame) -> Series:
        variance = (
                df[self._numerator_sumsq] / df[self._denominator] -
                df[POINT_ESTIMATE] ** 2)
        if (variance < 0).any():
            raise ValueError('Computed variance is negative. '
                             'Please check your inputs.')
        return variance

    def _add_point_estimate_ci(self, df: DataFrame):
        df[CI_LOWER], df[CI_UPPER] = _zconfint_generic(
            mean=df[POINT_ESTIMATE],
            std_mean=np.sqrt(df[VARIANCE] / df[self._denominator]),
            alpha=1-self._interval_size,
            alternative=TWO_SIDED
        )
        return df

    def _p_value(self, row) -> float:
        _, p_value = _zstat_generic(value1=row[POINT_ESTIMATE + SFX2],
                                    value2=row[POINT_ESTIMATE + SFX1],
                                    std_diff=row[STD_ERR],
                                    alternative=row[PREFERENCE],
                                    diff=row[NULL_HYPOTHESIS])
        return p_value

    def _ci(self, row, alpha_column: str) -> Tuple[float, float]:
        return _zconfint_generic(
            mean=row[DIFFERENCE],
            std_mean=row[STD_ERR],
            alpha=row[alpha_column],
            alternative=row[PREFERENCE])

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

    def _powered_effect(self,
                                df: DataFrame,
                                power: float,
                                alpha: float) -> DataFrame:

        proportion_of_total = 1 #TODO
        z_alpha = norm.ppf(1 - alpha)
        z_power = norm.ppf(power)
        n1, n2 = df[self._denominator + SFX1], df[self._denominator + SFX2]
        binary = self._numerator_sumsq == self._numerator
        kappa = n1/n2
        current_number_of_units = n1 + n2
        if binary and df[NIM] is None:
            effect = _search_MDE_binary_local_search(
                control_avg=df[POINT_ESTIMATE + SFX1],
                control_var=df[VARIANCE + SFX1],
                non_inferiority=df[NIM],
                kappa=kappa,
                proportion_of_total=proportion_of_total,
                current_number_of_units=current_number_of_units,
                z_alpha=z_alpha,
                z_power=z_power,
            )[0]
        else:
            treatment_var = _get_hypothetical_treatment_var(
                binary_metric=binary, non_inferiority = df[NIM] is not None, control_avg = df[POINT_ESTIMATE + SFX1], control_var = df[VARIANCE + SFX1], hypothetical_effect=0
            )
            n2_partial = np.power((z_alpha + z_power), 2) * (df[VARIANCE + SFX1] / kappa + treatment_var)
            effect = np.sqrt((1 / (current_number_of_units * proportion_of_total)) * (
                        n2_partial + kappa * n2_partial))

        return (
            df.assign(powered_effect=effect)
              .loc[:, ['level_1', 'level_2', 'powered_effect']]
              .reset_index()
        )