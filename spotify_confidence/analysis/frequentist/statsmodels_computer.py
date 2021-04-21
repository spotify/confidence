# Copyright 2017-2020 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pandas import DataFrame, Series
import numpy as np
import scipy.stats as st
from statsmodels.stats.proportion import (
    proportions_chisquare, proportion_confint, confint_proportions_2indep)
from statsmodels.stats.weightstats import (
    _zstat_generic, _zconfint_generic, _tstat_generic, _tconfint_generic)
from typing import (Union, Iterable, List, Tuple, Dict)
from abc import abstractmethod

from ..abstract_base_classes.confidence_computer_abc import \
    ConfidenceComputerABC
from .sequential_bound_solver import bounds
from ..constants import (POINT_ESTIMATE, VARIANCE, CI_LOWER, CI_UPPER,
                         DIFFERENCE, P_VALUE, SFX1, SFX2, STD_ERR, ALPHA,
                         ADJUSTED_ALPHA, ADJUSTED_P, ADJUSTED_LOWER, ADJUSTED_UPPER,
                         NULL_HYPOTHESIS, NIM, PREFERENCE, TWO_SIDED,
                         PREFERENCE_DICT, NIM_TYPE, BONFERRONI, BONFERRONI_ONLY_COUNT_TWOSIDED)
from ..confidence_utils import (get_remaning_groups, validate_levels,
                                level2str, listify, get_all_group_columns,
                                power_calculation, validate_nims, signed_nims)


def sequential_bounds(t: np.array, alpha: float, sides: int):
    return bounds(t, alpha, rho=2, ztrun=8, sides=sides, max_nints=1000)


class StatsmodelsComputer(ConfidenceComputerABC):

    def __init__(self, data_frame: DataFrame, numerator_column: str,
                 numerator_sum_squares_column: str, denominator_column: str,
                 categorical_group_columns: Union[str, Iterable],
                 ordinal_group_column: str, interval_size: float,
                 correction_method: str):

        self._df = data_frame
        self._numerator = numerator_column
        self._numerator_sumsq = numerator_sum_squares_column
        if self._numerator_sumsq is None or \
                self._numerator_sumsq == self._numerator:
            if (data_frame[numerator_column] <=
                    data_frame[denominator_column]).all():
                # Treat as binomial data
                self._numerator_sumsq = self._numerator
            else:
                raise ValueError(
                    f'numerator_sum_squares_column missing or same as '
                    f'numerator_column, but since {numerator_column} is not '
                    f'always smaller than {denominator_column} it can\'t be '
                    f'binomial data. Please check your data.')

        self._denominator = denominator_column
        self._categorical_group_columns = categorical_group_columns
        self._ordinal_group_column = ordinal_group_column
        self._interval_size = interval_size

        correction_methods = [BONFERRONI, BONFERRONI_ONLY_COUNT_TWOSIDED]
        if correction_method.lower() not in correction_methods:
            raise ValueError(f'Use one of the correction methods ' +
                             f'in {correction_methods}')
        self._correction_method = correction_method

        self._all_group_columns = get_all_group_columns(
            self._categorical_group_columns, self._ordinal_group_column)
        self._sufficient = None

    def compute_summary(self) -> DataFrame:
        return self._sufficient_statistics[
              self._all_group_columns +
              [self._numerator, self._denominator,
               POINT_ESTIMATE, CI_LOWER, CI_UPPER]]

    @property
    def _sufficient_statistics(self) -> DataFrame:
        if self._sufficient is None:
            self._sufficient = (
                self._df
                    .assign(**{POINT_ESTIMATE: self._point_estimate})
                    .assign(**{VARIANCE: self._variance})
                    .pipe(self._add_point_estimate_ci)
            )
        return self._sufficient

    def _point_estimate(self, df: DataFrame) -> Series:
        if (df[self._denominator] == 0).any():
            raise ValueError('''Can't compute point estimate:
                                denominator is 0''')
        return df[self._numerator] / df[self._denominator]

    def compute_difference(self,
                           level_1: Union[str, Iterable],
                           level_2: Union[str, Iterable],
                           absolute: bool,
                           groupby: str,
                           nims: NIM_TYPE,
                           final_expected_sample_size: float
                           ) -> DataFrame:
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        difference_df = self._compute_differences(level_columns,
                                                  level_1,
                                                  [level_2],
                                                  absolute,
                                                  groupby,
                                                  level_as_reference=True,
                                                  nims=nims,
                                                  final_expected_sample_size=final_expected_sample_size)
        return difference_df[listify(groupby) +
                             ['level_1', 'level_2', 'absolute_difference',
                              DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE] +
                             [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P] +
                             ([NIM, NULL_HYPOTHESIS, PREFERENCE]
                              if nims is not None else [])]

    def compute_multiple_difference(self,
                                    level: Union[str, Iterable],
                                    absolute: bool,
                                    groupby: Union[str, Iterable],
                                    level_as_reference: bool,
                                    nims: NIM_TYPE,
                                    final_expected_sample_size: float
                                    ) -> DataFrame:
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        other_levels = [other for other in self._sufficient_statistics
                        .groupby(level_columns).groups.keys() if other != level]
        difference_df = self._compute_differences(level_columns,
                                                  level,
                                                  other_levels,
                                                  absolute,
                                                  groupby,
                                                  level_as_reference,
                                                  nims,
                                                  final_expected_sample_size)
        return difference_df[listify(groupby) +
                             ['level_1', 'level_2', 'absolute_difference',
                              DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE] +
                             [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P] +
                             ([NIM, NULL_HYPOTHESIS, PREFERENCE]
                              if nims is not None else [])]

    def _compute_differences(self,
                             level_columns: Iterable,
                             level: Union[str, Iterable],
                             other_levels: Iterable,
                             absolute: bool,
                             groupby: Union[str, Iterable],
                             level_as_reference: bool,
                             nims: NIM_TYPE,
                             final_expected_sample_size: float):
        groupby = listify(groupby)
        validate_levels(self._sufficient_statistics,
                        level_columns,
                        [level] + other_levels)
        validate_nims(self._sufficient_statistics, groupby, nims)
        levels = [(level2str(level), level2str(other))
                  if level_as_reference
                  else (level2str(other), level2str(level))
                  for other in other_levels]
        str2level = {level2str(lv): lv for lv in [level] + other_levels}
        return (
            self._sufficient_statistics
                .assign(level=self._sufficient_statistics[level_columns]
                        .agg(level2str, axis='columns'))
                .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
                .pipe(self._create_comparison_df,
                      groups_to_compare=levels,
                      absolute=absolute,
                      nims=nims,
                      final_expected_sample_size=final_expected_sample_size)
                .assign(level_1=lambda df:
                        df['level_1'].map(lambda s: str2level[s]))
                .assign(level_2=lambda df:
                        df['level_2'].map(lambda s: str2level[s]))
                .reset_index()
                .sort_values(by=groupby + ['level_1', 'level_2'])
        )

    def _create_comparison_df(self,
                              df: DataFrame,
                              groups_to_compare: List[Tuple[str, str]],
                              absolute: bool,
                              nims: NIM_TYPE,
                              final_expected_sample_size: float
                              ) -> DataFrame:

        def join(df: DataFrame) -> DataFrame:
            has_index = not all(idx is None for idx in df.index.names)
            if has_index:
                # self-join on index (the index will typically model the date,
                # i.e., rows with the same date are joined)
                return df.merge(df,
                                left_index=True,
                                right_index=True,
                                suffixes=(SFX1, SFX2))
            else:
                # join on dummy column, i.e. conduct a cross join
                return (
                    df.assign(dummy_join_column=1)
                      .merge(right=df.assign(dummy_join_column=1),
                             on='dummy_join_column',
                             suffixes=(SFX1, SFX2))
                      .drop(columns='dummy_join_column')
                )

        comparison_df = (
            df.pipe(join)
              .query(f'level_1 in {[l1 for l1,l2 in groups_to_compare]} and ' +
                     f'level_2 in {[l2 for l1,l2 in groups_to_compare]}')
              .assign(**{DIFFERENCE: lambda df: df[POINT_ESTIMATE + SFX2] -
                      df[POINT_ESTIMATE + SFX1]})
              .assign(**{STD_ERR: self._std_err})
              .assign(**{NIM: lambda df: self._nims_2_series(df, nims)[NIM]})
              .assign(**{NULL_HYPOTHESIS: lambda df:
                         self._nims_2_series(df, nims)[NULL_HYPOTHESIS]})
              .assign(**{PREFERENCE: lambda df:
                         self._nims_2_series(df, nims)[PREFERENCE]})
              .pipe(self._add_p_value_and_ci, final_expected_sample_size=final_expected_sample_size)
              .pipe(self._adjust_if_absolute, absolute=absolute)
              .assign(**{PREFERENCE: lambda df:
                         df[PREFERENCE].map(PREFERENCE_DICT)})
        )
        return comparison_df

    @staticmethod
    def _adjust_if_absolute(df: DataFrame, absolute: bool) -> DataFrame:
        if absolute:
            return df.assign(absolute_difference=absolute)
        else:
            return (
                df.assign(absolute_difference=absolute)
                  .assign(**{DIFFERENCE:
                          df[DIFFERENCE] / df[POINT_ESTIMATE + SFX1]})
                  .assign(**{CI_LOWER:
                          df[CI_LOWER] / df[POINT_ESTIMATE + SFX1]})
                  .assign(**{CI_UPPER:
                          df[CI_UPPER] / df[POINT_ESTIMATE + SFX1]})
                  .assign(**{ADJUSTED_LOWER:
                          df[ADJUSTED_LOWER] / df[POINT_ESTIMATE + SFX1]})
                  .assign(**{ADJUSTED_UPPER:
                          df[ADJUSTED_UPPER] / df[POINT_ESTIMATE + SFX1]})
                  .assign(**{NULL_HYPOTHESIS:
                          df[NULL_HYPOTHESIS] / df[POINT_ESTIMATE + SFX1]})
            )

    def _nims_2_series(self,
                       df: DataFrame,
                       nims: NIM_TYPE
                       ) -> Union[DataFrame, Dict[float, str]]:

        sgnd_nims = signed_nims(nims)
        if nims is None or type(nims) is tuple:
            return {NIM: sgnd_nims[0],
                    NULL_HYPOTHESIS: df[POINT_ESTIMATE + SFX1]*sgnd_nims[1],
                    PREFERENCE: np.repeat(sgnd_nims[2], len(df))}
        else:
            return (
                DataFrame(index=df.index,
                          columns=[NIM, NULL_HYPOTHESIS, PREFERENCE],
                          data=list(df.index.to_series().map(sgnd_nims)))
                .assign(**{NULL_HYPOTHESIS: lambda d:
                           d[NULL_HYPOTHESIS]*df[POINT_ESTIMATE + SFX1]})
            )

    def _std_err(self, df: DataFrame) -> Series:
        return np.sqrt(df[VARIANCE + SFX1] / df[self._denominator + SFX1] +
                       df[VARIANCE + SFX2] / df[self._denominator + SFX2])

    def _add_p_value_and_ci(self, df: DataFrame, final_expected_sample_size: float) -> DataFrame:
        df[ALPHA] = 1 - self._interval_size

        if(final_expected_sample_size is None):
            df[ADJUSTED_ALPHA] = (1-self._interval_size)/len(df)
        else:
            df[ADJUSTED_ALPHA] = self._compute_sequential_adjusted_alpha(df, final_expected_sample_size)

        ci = df.apply(self._ci, axis=1, alpha_column=ALPHA)
        ci_df = DataFrame(index=ci.index,
                          columns=[CI_LOWER, CI_UPPER],
                          data=list(ci.values))
        adjusted_ci = df.apply(self._ci, axis=1, alpha_column=ADJUSTED_ALPHA)
        adjusted_ci_df = DataFrame(index=adjusted_ci.index,
                                   columns=[ADJUSTED_LOWER, ADJUSTED_UPPER],
                                   data=list(adjusted_ci.values))

        return (
            df.assign(**{P_VALUE: df.apply(self._p_value, axis=1)})
              .assign(**{ADJUSTED_P: lambda df:
                      df[P_VALUE].map(lambda p: min(p * len(df), 1))})
              .assign(**{CI_LOWER: ci_df[CI_LOWER]})
              .assign(**{CI_UPPER: ci_df[CI_UPPER]})
              .assign(**{ADJUSTED_LOWER: adjusted_ci_df[ADJUSTED_LOWER]})
              .assign(**{ADJUSTED_UPPER: adjusted_ci_df[ADJUSTED_UPPER]})
        )

    def _compute_sequential_adjusted_alpha(self, df, final_expected_sample_size):
        sample_size_by_ordinal = (
                df[self._denominator + SFX1].groupby(self._ordinal_group_column).sum() +
                df[self._denominator + SFX2].groupby(self._ordinal_group_column).sum()
        )
        final_expected_sample_size = max(final_expected_sample_size, sample_size_by_ordinal.max())
        sample_size_proportions = sample_size_by_ordinal / final_expected_sample_size

        def get_num_comparisons(df):
            if self._correction_method == BONFERRONI:
                return len(df)
            elif self._correction_method == BONFERRONI_ONLY_COUNT_TWOSIDED:
                return max(1, len(df.query(f'{PREFERENCE} == "{TWO_SIDED}"')))
            else:
                raise ValueError("Unsupported correction method")

        alpha = (1 - self._interval_size) / (get_num_comparisons(df) / len(sample_size_proportions))

        z_crit_one_sided = (
            sequential_bounds(
                t=sample_size_proportions.values, alpha=alpha, sides=1
            ).df.set_index(sample_size_proportions.index)['zb']
        ) if not (df[PREFERENCE] == TWO_SIDED).all() else None

        z_crit_two_sided = (
            sequential_bounds(
                t=sample_size_proportions.values, alpha=alpha, sides=2
            ).df.set_index(sample_size_proportions.index)['zb']
        ) if not (df[PREFERENCE] != TWO_SIDED).all() else None

        def z_crit(row):
            has_multi_index = len(df.index.names) > 1
            ordinal = row.name[df.index.names.index(self._ordinal_group_column)] if has_multi_index else row.name
            return z_crit_two_sided.loc[ordinal] if row[PREFERENCE] == TWO_SIDED else z_crit_one_sided.loc[ordinal]

        def alpha_from_z_crit(row):
            return 2 * (1 - st.norm.cdf(z_crit(row))) if row[PREFERENCE] == TWO_SIDED else 1 - st.norm.cdf(z_crit(row))

        return df.apply(alpha_from_z_crit, axis=1)

    def achieved_power(self, level_1, level_2, mde, alpha, groupby):
        """Calculated the achieved power of test of differences between
        level 1 and level 2 given a targeted MDE.

        Args:
            level_1 (str, tuple of str): Name of first level.
            level_2 (str, tuple of str): Name of second level.
            mde (float): Absolute minimal detectable effect size.
            alpha (float): Type I error rate, cutoff value for determining
                statistical significance.
            groupby (str): Name of column.
                If specified, will return the difference for each level
                of the grouped dimension.

        Returns:
            Pandas DataFrame with the following columns:
            - level_1: Name of level 1.
            - level_2: Name of level 2.
            - power: 1 - B, where B is the likelihood of a Type II (false
                negative) error.

        """
        groupby = listify(groupby)
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        return (
            self._compute_differences(level_columns,
                                      level_1,
                                      [level_2],
                                      True,
                                      groupby,
                                      level_as_reference=True,
                                      nims=None,  # TODO: IS this right?
                                      final_expected_sample_size=None)  # TODO: IS this right?
                .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
                .pipe(self._achieved_power, mde=mde, alpha=alpha)
        )

    @staticmethod
    def _variance(self, df: DataFrame) -> Series:
        pass

    @abstractmethod
    def _add_point_estimate_ci(self, df: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def _p_value(self, row) -> float:
        pass

    @abstractmethod
    def _ci(self, row, alpha_column: str) -> Tuple[float, float]:
        pass

    @abstractmethod
    def _achieved_power(self,
                        df: DataFrame,
                        mde: float,
                        alpha: float) -> DataFrame:
        pass


class ChiSquaredComputer(StatsmodelsComputer):
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


class TTestComputer(StatsmodelsComputer):
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
                                    alternative=row[PREFERENCE],
                                    diff=row[NULL_HYPOTHESIS])
        return p_value

    def _ci(self, row, alpha_column: str) -> Tuple[float, float]:
        return _tconfint_generic(
            mean=row[DIFFERENCE],
            std_mean=row[STD_ERR],
            dof=self._dof(row),
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


class ZTestComputer(StatsmodelsComputer):
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
