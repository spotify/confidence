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

from warnings import warn
from pandas import DataFrame, Series, concat
import numpy as np
from statsmodels.stats.multitest import multipletests
from typing import (Union, Iterable, List, Tuple)
from abc import abstractmethod

from ..abstract_base_classes.confidence_computer_abc import \
    ConfidenceComputerABC
from .sequential_bound_solver import bounds
from ..constants import (POINT_ESTIMATE, VARIANCE, CI_LOWER, CI_UPPER,
                         DIFFERENCE, P_VALUE, SFX1, SFX2, STD_ERR, ALPHA,
                         ADJUSTED_ALPHA, POWER, POWERED_EFFECT, ADJUSTED_POWER, ADJUSTED_P,
                         ADJUSTED_LOWER, ADJUSTED_UPPER, IS_SIGNIFICANT, REQUIRED_SAMPLE_SIZE,
                         NULL_HYPOTHESIS, NIM, PREFERENCE, PREFERENCE_TEST, TWO_SIDED,
                         PREFERENCE_DICT, NIM_TYPE, BONFERRONI, CORRECTION_METHODS,
                         HOLM, HOMMEL, SIMES_HOCHBERG, SIDAK, HOLM_SIDAK, FDR_BH, FDR_BY, FDR_TSBH,
                         FDR_TSBKY,
                         SPOT_1_HOLM, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG,
                         SPOT_1_SIDAK, SPOT_1_HOLM_SIDAK, SPOT_1_FDR_BH,
                         SPOT_1_FDR_BY, SPOT_1_FDR_TSBH, SPOT_1_FDR_TSBKY,
                         BONFERRONI_ONLY_COUNT_TWOSIDED, BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
                         SPOT_1)
from ..confidence_utils import (get_remaning_groups, validate_levels,
                                level2str, listify, get_all_group_columns,
                                add_nim_columns,
                                validate_and_rename_nims,
                                validate_and_rename_final_expected_sample_sizes,
                                get_all_categorical_group_columns,
                                add_mde_columns)


def sequential_bounds(t: np.array, alpha: float, sides: int):
    return bounds(t, alpha, rho=2, ztrun=8, sides=sides, max_nints=1000)


class GenericComputer(ConfidenceComputerABC):

    def __init__(self, data_frame: DataFrame, numerator_column: str,
                 numerator_sum_squares_column: str, denominator_column: str,
                 categorical_group_columns: Union[str, Iterable],
                 metric_column: Union[str, None],
                 treatment_column: Union[str, None],
                 ordinal_group_column: str,
                 interval_size: float,
                 power: float, correction_method: str):

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
        self._categorical_group_columns = get_all_categorical_group_columns(
                                            categorical_group_columns, metric_column,
                                            treatment_column)
        self._segments = list(set(self._categorical_group_columns) - set([metric_column]) - set([treatment_column]))
        self._ordinal_group_column = ordinal_group_column
        self._metric_column = metric_column
        self._interval_size = interval_size
        self._power = power
        self._treatment_column = treatment_column

        if correction_method.lower() not in CORRECTION_METHODS:
            raise ValueError(f'Use one of the correction methods ' +
                             f'in {CORRECTION_METHODS}')
        self._correction_method = correction_method

        self._single_metric = True
        if self._metric_column is not None:
            if data_frame.groupby(self._metric_column).ngroups == 1:
                self._categorical_group_columns = list(
                    set(self._categorical_group_columns) - set([self._metric_column]))
            else:
                self._single_metric = False

        self._all_group_columns = get_all_group_columns(
            self._categorical_group_columns, self._ordinal_group_column)
        self._sufficient = None

    def compute_summary(self, verbose: bool) -> DataFrame:
        return (
            self._sufficient_statistics if verbose else
            self._sufficient_statistics[
                self._all_group_columns + [self._numerator, self._denominator, POINT_ESTIMATE,
                                           CI_LOWER, CI_UPPER]
                ]
        )

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
                           groupby: Union[str, Iterable],
                           nims: NIM_TYPE,
                           mdes: bool,
                           final_expected_sample_size_column: str,
                           verbose: bool) -> DataFrame:
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        difference_df = self._compute_differences(level_columns,
                                                  [(level_1, level_2)],
                                                  absolute,
                                                  groupby,
                                                  level_as_reference=True,
                                                  nims=nims,
                                                  mdes=mdes,
                                                  final_expected_sample_size_column=final_expected_sample_size_column)
        return (difference_df if verbose else
                difference_df[listify(groupby) +
                              ['level_1', 'level_2', 'absolute_difference',
                               DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE] +
                              [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT,
                               POWERED_EFFECT, REQUIRED_SAMPLE_SIZE] +
                              ([NIM, NULL_HYPOTHESIS, PREFERENCE]
                               if nims is not None else [])])


    def compute_multiple_difference(self,
                                    level: Union[str, Iterable],
                                    absolute: bool,
                                    groupby: Union[str, Iterable],
                                    level_as_reference: bool,
                                    nims: NIM_TYPE,
                                    minimum_detectable_effect:bool,
                                    final_expected_sample_size_column: str,
                                    verbose: bool) -> DataFrame:
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        other_levels = [other for other in self._sufficient_statistics
            .groupby(level_columns).groups.keys() if other != level]
        levels = [(level, other) for other in other_levels]
        difference_df = self._compute_differences(level_columns,
                                                  levels,
                                                  absolute,
                                                  groupby,
                                                  level_as_reference,
                                                  nims,
                                                  minimum_detectable_effect,
                                                  final_expected_sample_size_column)
        return (difference_df if verbose else
                difference_df[listify(groupby) +
                              ['level_1', 'level_2', 'absolute_difference',
                               DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE, POWERED_EFFECT, REQUIRED_SAMPLE_SIZE] +
                              [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT] +
                              ([NIM, NULL_HYPOTHESIS, PREFERENCE]
                               if nims is not None else [])])

    def compute_differences(self,
                            levels: List[Tuple],
                            absolute: bool,
                            groupby: Union[str, Iterable],
                            nims: NIM_TYPE,
                            final_expected_sample_size_column: str,
                            verbose: bool
                            ) -> DataFrame:
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        difference_df = self._compute_differences(
            level_columns,
            [levels] if type(levels) == tuple else levels,
            absolute,
            groupby,
            level_as_reference=True,
            nims=nims,
            final_expected_sample_size_column=final_expected_sample_size_column)
        return (difference_df if verbose else
                difference_df[listify(groupby) +
                              ['level_1', 'level_2', 'absolute_difference',
                               DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE] +
                              [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT,
                               POWERED_EFFECT, REQUIRED_SAMPLE_SIZE] +
                              ([NIM, NULL_HYPOTHESIS, PREFERENCE]
                               if nims is not None else [])])

    def _compute_differences(self,
                             level_columns: Iterable,
                             levels: Union[str, Iterable],
                             absolute: bool,
                             groupby: Union[str, Iterable],
                             level_as_reference: bool,
                             nims: NIM_TYPE,
                             mdes:bool,
                             final_expected_sample_size_column: str):
        if type(level_as_reference) is not bool:
            raise ValueError(
                f'level_is_reference must be either True or False, but is {level_as_reference}.')
        groupby = listify(groupby)
        unique_levels = set([l[0] for l in levels] + [l[1] for l in levels])
        validate_levels(self._sufficient_statistics,
                        level_columns,
                        unique_levels)
        str2level = {level2str(lv): lv for lv in unique_levels}
        filtered_sufficient_statistics = concat(
            [self._sufficient_statistics.groupby(level_columns).get_group(group) for group in
             unique_levels])
        levels = [(level2str(l[0]), level2str(l[1]))
                  if level_as_reference
                  else (level2str(l[1]), level2str(l[0]))
                  for l in levels]
        return (
            self._sufficient_statistics
                .assign(level=self._sufficient_statistics[level_columns]
                        .agg(level2str, axis='columns'))
                .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
                .pipe(self._create_comparison_df,
                      groups_to_compare=levels,
                      absolute=absolute,
                      nims=nims,
                      mdes=mdes,
                      final_expected_sample_size_column=final_expected_sample_size_column,
                      filtered_sufficient_statistics=filtered_sufficient_statistics)
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
                              mdes:bool,
                              final_expected_sample_size_column: str,
                              filtered_sufficient_statistics: DataFrame
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
            df.pipe(add_nim_columns, nims=nims)
                .pipe(add_mde_columns, mdes=mdes)
                .pipe(join)
                .query(f'level_1 in {[l1 for l1, l2 in groups_to_compare]} and ' +
                       f'level_2 in {[l2 for l1, l2 in groups_to_compare]}' +
                       'and level_1 != level_2')
                .assign(**{DIFFERENCE: lambda df: df[POINT_ESTIMATE + SFX2] -
                                                  df[POINT_ESTIMATE + SFX1]})
                .assign(**{STD_ERR: self._std_err})
                .pipe(validate_and_rename_nims)
                .pipe(validate_and_rename_final_expected_sample_sizes,
                      final_expected_sample_size_column)
                .pipe(self._add_p_value_and_ci,
                      final_expected_sample_size_column=final_expected_sample_size_column,
                      filtered_sufficient_statistics=filtered_sufficient_statistics)
                .pipe(self._adjust_if_absolute, absolute=absolute)
                .pipe(self._add_adjusted_power)
                .apply(self._powered_effect_and_required_sample_size, axis=1)
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

    def _std_err(self, df: DataFrame) -> Series:
        return np.sqrt(df[VARIANCE + SFX1] / df[self._denominator + SFX1] +
                       df[VARIANCE + SFX2] / df[self._denominator + SFX2])

    def _corrections_power(self, number_of_success_metrics: int,
                           number_of_guardrail_metrics: int) -> int:
        return number_of_guardrail_metrics if number_of_success_metrics == 0 else \
            number_of_guardrail_metrics + 1

    def _add_adjusted_power(self, df: DataFrame) -> DataFrame:
        groupby = ['level_1', 'level_2'] + [column for column in df.index.names if
                                            column is not None]
        power_correction = 1
        if self._correction_method in [BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY, SPOT_1,
                                       HOLM, HOMMEL, SIMES_HOCHBERG,
                                       SIDAK, HOLM_SIDAK, FDR_BH, FDR_BY, FDR_TSBH, FDR_TSBKY,
                                       SPOT_1_HOLM, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG,
                                       SPOT_1_SIDAK, SPOT_1_HOLM_SIDAK, SPOT_1_FDR_BH,
                                       SPOT_1_FDR_BY, SPOT_1_FDR_TSBH, SPOT_1_FDR_TSBKY]:

            self._number_total_metrics = 1 if self._single_metric else df.groupby(
                self._metric_column).ngroups
            if self._single_metric:
                if df[df[NIM].isnull()].shape[0] > 0:
                    self._number_success_metrics = 1
                else:
                    self._number_success_metrics = 0
            else:
                self._number_success_metrics = df[df[NIM].isnull()].groupby(
                    self._metric_column).ngroups

            self._number_guardrail_metrics = self._number_total_metrics - \
                                             self._number_success_metrics
            power_correction = self._corrections_power(
                number_of_guardrail_metrics=self._number_guardrail_metrics,
                number_of_success_metrics=self._number_success_metrics)

        return df.assign(**{ADJUSTED_POWER: 1 - (1 - df[POWER]) / power_correction})

    def _add_p_value_and_ci(self,
                            df: DataFrame,
                            final_expected_sample_size_column: str,
                            filtered_sufficient_statistics: DataFrame) -> DataFrame:

        def set_alpha_and_adjust_preference(df: DataFrame) -> DataFrame:
            alpha_0 = 1 - self._interval_size
            return (
                df.assign(
                    **{ALPHA: df.apply(lambda row: 2 * alpha_0 if self._correction_method == SPOT_1
                                                                  and row[PREFERENCE] != TWO_SIDED
                    else alpha_0, axis=1)})
                    .assign(**{POWER: self._power})
                    .assign(**{PREFERENCE_TEST: df.apply(
                    lambda row: TWO_SIDED if self._correction_method == SPOT_1
                    else row[PREFERENCE], axis=1)})
            )

        def _add_adjusted_p_and_is_significant(df: DataFrame) -> DataFrame:
            if (final_expected_sample_size_column is not None):
                if self._correction_method not in [BONFERRONI, BONFERRONI_ONLY_COUNT_TWOSIDED,
                                                   BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
                                                   SPOT_1]:
                    raise ValueError(
                        f"{self._correction_method} not supported for sequential tests. Use one of"
                        f"{BONFERRONI}, {BONFERRONI_ONLY_COUNT_TWOSIDED}, "
                        f"{BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY}, {SPOT_1}")

                df[ADJUSTED_ALPHA] = self._compute_sequential_adjusted_alpha(df,
                                                                             final_expected_sample_size_column,
                                                                             filtered_sufficient_statistics)
                df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
                df[P_VALUE] = None
                df[ADJUSTED_P] = None
            elif self._correction_method in [HOLM, HOMMEL, SIMES_HOCHBERG,
                                             SIDAK, HOLM_SIDAK, FDR_BH, FDR_BY, FDR_TSBH,
                                             FDR_TSBKY,
                                             SPOT_1_HOLM, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG,
                                             SPOT_1_SIDAK, SPOT_1_HOLM_SIDAK, SPOT_1_FDR_BH,
                                             SPOT_1_FDR_BY, SPOT_1_FDR_TSBH, SPOT_1_FDR_TSBKY]:
                if self._correction_method.startswith('spot-'):
                    correction_method = self._correction_method[7:]
                else:
                    correction_method = self._correction_method

                groupby = ['level_1', 'level_2'] + [column for column in df.index.names if
                                                    column is not None]
                df[ADJUSTED_ALPHA] = df[ALPHA] / self._get_num_comparisons(df,
                                                                           self._correction_method,
                                                                           groupby)
                is_significant, adjusted_p, _, _ = multipletests(pvals=df[P_VALUE],
                                                                 alpha=1 - self._interval_size,
                                                                 method=correction_method)
                df[ADJUSTED_P] = adjusted_p
                df[IS_SIGNIFICANT] = is_significant
            elif self._correction_method in [BONFERRONI, BONFERRONI_ONLY_COUNT_TWOSIDED,
                                             BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY, SPOT_1]:
                groupby = ['level_1', 'level_2'] + [column for column in df.index.names if
                                                    column is not None]
                n_comparisons = self._get_num_comparisons(df, self._correction_method, groupby)
                df[ADJUSTED_ALPHA] = df[ALPHA] / n_comparisons / (1 + (df[PREFERENCE_TEST]=='two-sided').astype(int))
                df[ADJUSTED_P] = df.apply(lambda row: min(row[P_VALUE] * n_comparisons * (1 + (row[PREFERENCE_TEST]=='two-sided')), 1), axis=1)
                #df[ADJUSTED_P] = df[P_VALUE].map(lambda p: min(p * n_comparisons , 1))
                df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
            else:
                raise ValueError("Can't figure out which correction method to use :(")

            return df

        def _add_ci(df: DataFrame) -> DataFrame:
            ci = df.apply(self._ci, axis=1, alpha_column=ALPHA)
            ci_df = DataFrame(index=ci.index,
                              columns=[CI_LOWER, CI_UPPER],
                              data=list(ci.values))

            if self._correction_method in [HOLM, HOMMEL, SIMES_HOCHBERG,
                                           SPOT_1_HOLM, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG] \
                    and all(df[PREFERENCE_TEST] != TWO_SIDED):
                adjusted_ci = self._ci_for_multiple_comparison_methods(
                    df,
                    correction_method=self._correction_method,
                    alpha=1 - self._interval_size,
                )
            elif self._correction_method in [BONFERRONI, BONFERRONI_ONLY_COUNT_TWOSIDED,
                                             BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY, SPOT_1,
                                             SPOT_1_HOLM, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG,
                                             SPOT_1_SIDAK, SPOT_1_HOLM_SIDAK, SPOT_1_FDR_BH,
                                             SPOT_1_FDR_BY, SPOT_1_FDR_TSBH, SPOT_1_FDR_TSBKY]:
                adjusted_ci = df.apply(self._ci, axis=1, alpha_column=ADJUSTED_ALPHA)
            else:
                warn(f"Confidence intervals not supported for {self._correction_method}")
                adjusted_ci = Series(index=df.index, data=[(None, None) for i in range(len(df))])

            adjusted_ci_df = DataFrame(index=adjusted_ci.index,
                                       columns=[ADJUSTED_LOWER, ADJUSTED_UPPER],
                                       data=list(adjusted_ci.values))

            return (
                df.assign(**{CI_LOWER: ci_df[CI_LOWER]})
                    .assign(**{CI_UPPER: ci_df[CI_UPPER]})
                    .assign(**{ADJUSTED_LOWER: adjusted_ci_df[ADJUSTED_LOWER]})
                    .assign(**{ADJUSTED_UPPER: adjusted_ci_df[ADJUSTED_UPPER]})
            )

        return (
            df.pipe(set_alpha_and_adjust_preference)
                .assign(**{P_VALUE: lambda df: df.apply(self._p_value, axis=1)})
                .pipe(_add_adjusted_p_and_is_significant)
                .pipe(_add_ci)
        )

    def _get_num_comparisons(self, df: DataFrame, correction_method: str,
                             groupby: Iterable) -> int:
        if correction_method == BONFERRONI:
            return max(1, df.groupby(groupby).ngroups)
        elif correction_method == BONFERRONI_ONLY_COUNT_TWOSIDED:
            return max(df.query(f'{PREFERENCE_TEST} == "{TWO_SIDED}"').groupby(groupby).ngroups, 1)
        elif correction_method in [HOLM, HOMMEL, SIMES_HOCHBERG,
                                       SIDAK, HOLM_SIDAK, FDR_BH, FDR_BY, FDR_TSBH, FDR_TSBKY]:
            return 1
        elif correction_method in [BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY, SPOT_1,
                                   SPOT_1_HOLM, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG,
                                   SPOT_1_SIDAK, SPOT_1_HOLM_SIDAK, SPOT_1_FDR_BH,
                                   SPOT_1_FDR_BY, SPOT_1_FDR_TSBH, SPOT_1_FDR_TSBKY]:

            if self._single_metric:
                if df[df[NIM].isnull()].shape[0] > 0:
                    self._number_success_metrics = 1
                else:
                    self._number_success_metrics = 0
            else:
                self._number_success_metrics = df[df[NIM].isnull()].groupby(
                    self._metric_column).ngroups

            number_comparions = len((df[self._treatment_column+SFX1]+df[self._treatment_column+SFX2] ).unique())
            number_segments = (1 if len(self._segments) is 0 else df.groupby(self._segments).ngroups)



            return max(1, number_comparions * max(1, self._number_success_metrics)* number_segments)
        else:
            raise ValueError(f"Unsupported correction method: {correction_method}.")

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
                                      [(level_1, level_2)],
                                      True,
                                      groupby,
                                      level_as_reference=True,
                                      nims=None,  # TODO: IS this right?
                                      final_expected_sample_size_column=None)  # TODO: IS this
                # right?
                .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
                .pipe(self._achieved_power, mde=mde, alpha=alpha)
        )

    @abstractmethod
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

    def _compute_sequential_adjusted_alpha(self,
                                           df: DataFrame,
                                           final_expected_sample_size_column: str,
                                           filtered_sufficient_statistics: DataFrame) -> Series:
        raise NotImplementedError("Sequential tests are only supported for ZTests")

    def _ci_for_multiple_comparison_methods(
            self,
            df: DataFrame,
            correction_method: str,
            alpha: float,
            w: float = 1.0,
    ) -> Tuple[Union[Series, float], Union[Series, float]]:
        raise NotImplementedError(f"{self._correction_method} is only supported for ZTests")

    def _powered_effect_and_required_sample_size(self,
                        df: DataFrame,
                        ) -> DataFrame:
        pass
