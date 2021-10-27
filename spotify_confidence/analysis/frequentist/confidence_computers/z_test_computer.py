from typing import Tuple, Union

import numpy as np
from pandas import DataFrame, Series
from scipy import stats as st
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import POINT_ESTIMATE, CI_LOWER, CI_UPPER, VARIANCE, TWO_SIDED, SFX2, SFX1, \
    STD_ERR, PREFERENCE_TEST, NULL_HYPOTHESIS, DIFFERENCE, ALPHA, IS_SIGNIFICANT, HOLM, SPOT_1_HOLM, HOMMEL, \
    SIMES_HOCHBERG, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG
from spotify_confidence.analysis.frequentist.sequential_bound_solver import bounds


def sequential_bounds(t: np.array, alpha: float, sides: int):
    return bounds(t, alpha, rho=2, ztrun=8, sides=sides, max_nints=1000)


class ZTestComputer(object):
    def __init__(self, numerator, numerator_sumsq, denominator, ordinal_group_column, interval_size):
        self._numerator = numerator
        self._numerator_sumsq = numerator_sumsq
        self._denominator = denominator
        self._ordinal_group_column = ordinal_group_column
        self._interval_size = interval_size

    def _point_estimate(self, row: Series) -> float:
        if row[self._denominator] == 0:
            raise ValueError('''Can't compute point estimate: denominator is 0''')
        return row[self._numerator] / row[self._denominator]

    def _variance(self, row: Series) -> float:
        variance = (
                row[self._numerator_sumsq] / row[self._denominator] -
                row[POINT_ESTIMATE] ** 2)
        if variance < 0:
            raise ValueError('Computed variance is negative. '
                             'Please check your inputs.')
        return variance

    def _std_err(self, row: Series) -> float:
        return np.sqrt(row[VARIANCE + SFX1] / row[self._denominator + SFX1] +
                       row[VARIANCE + SFX2] / row[self._denominator + SFX2])

    def _add_point_estimate_ci(self, row: Series) -> Series:
        row[CI_LOWER], row[CI_UPPER] = _zconfint_generic(
            mean=row[POINT_ESTIMATE],
            std_mean=np.sqrt(row[VARIANCE] / row[self._denominator]),
            alpha=1-self._interval_size,
            alternative=TWO_SIDED
        )
        return row

    def _p_value(self, row: Series) -> float:
        _, p_value = _zstat_generic(value1=row[POINT_ESTIMATE + SFX2],
                                    value2=row[POINT_ESTIMATE + SFX1],
                                    std_diff=row[STD_ERR],
                                    alternative=row[PREFERENCE_TEST],
                                    diff=row[NULL_HYPOTHESIS])
        return p_value

    def _ci(self, row: Series, alpha_column: str) -> Tuple[float, float]:
        return _zconfint_generic(
            mean=row[DIFFERENCE],
            std_mean=row[STD_ERR],
            alpha=row[alpha_column],
            alternative=row[PREFERENCE_TEST])

    def _achieved_power(self,
                        df: DataFrame,
                        mde: float,
                        alpha: float) -> DataFrame:
        v1, v2 = df[VARIANCE + SFX1], df[VARIANCE + SFX2]
        n1, n2 = df[self._denominator + SFX1], df[self._denominator + SFX2]

        var_pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)

        return power_calculation(mde, var_pooled, alpha, n1, n2)

    def _compute_sequential_adjusted_alpha(self,
                                           df: DataFrame,
                                           final_expected_sample_size_column: str,
                                           filtered_sufficient_statistics: DataFrame,
                                           num_comparisons: int):
        total_sample_size = (
            filtered_sufficient_statistics.groupby(df.index.names)
                                          .agg({self._denominator: sum, final_expected_sample_size_column: np.mean})
                                          .rename(columns={self._denominator: f'total_{self._denominator}'})
        )
        groups_except_ordinal = [
            column for column in df.index.names if column != self._ordinal_group_column]
        max_sample_size_by_group = (
            total_sample_size[f'total_{self._denominator}'].max() if len(groups_except_ordinal) == 0
            else total_sample_size.groupby(groups_except_ordinal)[f'total_{self._denominator}'].max())

        if type(max_sample_size_by_group) is not Series:
            total_sample_size = total_sample_size.assign(**{f'total_{self._denominator}_max': max_sample_size_by_group})
        else:
            total_sample_size = total_sample_size.merge(right=max_sample_size_by_group,
                                                        left_index=True,
                                                        right_index=True,
                                                        suffixes=('', '_max'))

        total_sample_size = (
            total_sample_size
            .assign(final_expected_sample_size=lambda df: df[[f'total_{self._denominator}_max',
                                                              final_expected_sample_size_column]].max(axis=1))
            .assign(
                    sample_size_proportions=lambda df: df['total_' + self._denominator]/df['final_expected_sample_size']
            )
        )

        def adjusted_alphas_for_group(grp: DataFrame) -> Series:
            return (
                sequential_bounds(
                    t=grp['sample_size_proportions'].values,
                    alpha=grp[ALPHA].values[0]/num_comparisons,
                    sides=2 if (grp[PREFERENCE_TEST] == TWO_SIDED).all() else 1
                ).df
                 .set_index(grp.index)
                 .assign(adjusted_alpha=lambda df: df.apply(
                    lambda row: 2 * (1 - st.norm.cdf(row['zb'])) if (grp[PREFERENCE_TEST] == TWO_SIDED).all()
                    else 1 - st.norm.cdf(row['zb']), axis=1))
            )[['zb', 'adjusted_alpha']]

        return (
            df.merge(total_sample_size, left_index=True, right_index=True)
              .groupby(groups_except_ordinal + ['level_1', 'level_2'])[['sample_size_proportions',
                                                                        PREFERENCE_TEST,
                                                                        ALPHA]]
              .apply(adjusted_alphas_for_group)
              .reset_index().set_index(df.index.names)
        )['adjusted_alpha']

    def _ci_for_multiple_comparison_methods(
            self,
            df: DataFrame,
            correction_method: str,
            alpha: float,
            w: float = 1.0,
    ) -> Tuple[Union[Series, float], Union[Series, float]]:
        if TWO_SIDED in df[PREFERENCE_TEST]:
            raise ValueError(
                "CIs can only be produced for one-sided tests when other multiple test corrections "
                "methods than bonferroni are applied"
            )
        m_scal = len(df)
        num_significant = sum(df[IS_SIGNIFICANT])
        r = m_scal - num_significant

        def _aw(W: float, alpha: float, m_scal: float, r: int):
            return alpha * (1 - (1 - W) * (m_scal - r) / m_scal)

        def _bw(W: float, alpha: float, m_scal: float, r: int):
            return 1 - (1 - alpha) / np.power((1 - (1 - W) * (1 - np.power((1 - alpha), (1 / m_scal)))), (m_scal - r))

        if correction_method in [HOLM, SPOT_1_HOLM]:
            adjusted_alpha_rej_equal_m = 1 - alpha / m_scal
            adjusted_alpha_rej_less_m = 1 - (1 - w) * (alpha / m_scal)
            adjusted_alpha_accept = 1 - _aw(w, alpha, m_scal, r) / r if r != 0 else 0
        elif correction_method in [HOMMEL, SIMES_HOCHBERG, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG]:
            adjusted_alpha_rej_equal_m = np.power((1 - alpha), (1 / m_scal))
            adjusted_alpha_rej_less_m = 1 - (1 - w) * (1 - np.power((1 - alpha), (1 / m_scal)))
            adjusted_alpha_accept = 1 - _bw(w, alpha, m_scal, r) / r if r != 0 else 0
        else:
            raise ValueError("CIs not supported for correction method. "
                             f"Supported methods: {HOMMEL}, {HOLM}, {SIMES_HOCHBERG},"
                             f"{SPOT_1_HOLM}, {SPOT_1_HOMMEL} and {SPOT_1_SIMES_HOCHBERG}")

        def _compute_ci_for_row(row: Series) -> Tuple[float, float]:
            if row[IS_SIGNIFICANT] and num_significant == m_scal:
                alpha_adj = adjusted_alpha_rej_equal_m
            elif row[IS_SIGNIFICANT] and num_significant < m_scal:
                alpha_adj = adjusted_alpha_rej_less_m
            else:
                alpha_adj = adjusted_alpha_accept

            ci_sign = -1 if row[PREFERENCE_TEST] == "larger" else 1
            bound1 = row[DIFFERENCE] + ci_sign * st.norm.ppf(alpha_adj) * row[STD_ERR]
            if ci_sign == -1:
                bound2 = max(row[NULL_HYPOTHESIS], bound1)
            else:
                bound2 = min(row[NULL_HYPOTHESIS], bound1)

            bound = bound2 if row[IS_SIGNIFICANT] else bound1

            lower = bound if row[PREFERENCE_TEST] == "larger" else -np.inf
            upper = bound if row[PREFERENCE_TEST] == "smaller" else np.inf

            return lower, upper

        return df.apply(_compute_ci_for_row, axis=1)
