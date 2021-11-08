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

from typing import Union, Iterable, List, Tuple
from warnings import warn

import numpy as np
from pandas import DataFrame, Series, concat
from statsmodels.stats.multitest import multipletests

from spotify_confidence.analysis.abstract_base_classes.confidence_computer_abc import ConfidenceComputerABC
from spotify_confidence.analysis.confidence_utils import (
    get_remaning_groups,
    validate_levels,
    level2str,
    listify,
    add_nim_columns,
    validate_and_rename_columns,
    add_mde_columns,
    get_all_categorical_group_columns,
    get_all_group_columns,
    validate_data,
    remove_group_columns,
)
from spotify_confidence.analysis.constants import (
    POINT_ESTIMATE,
    VARIANCE,
    CI_LOWER,
    CI_UPPER,
    DIFFERENCE,
    P_VALUE,
    SFX1,
    SFX2,
    STD_ERR,
    ALPHA,
    ADJUSTED_ALPHA,
    ADJUSTED_ALPHA_POWER_SAMPLE_SIZE,
    POWER,
    POWERED_EFFECT,
    ADJUSTED_POWER,
    ADJUSTED_P,
    ADJUSTED_LOWER,
    ADJUSTED_UPPER,
    IS_SIGNIFICANT,
    REQUIRED_SAMPLE_SIZE,
    NULL_HYPOTHESIS,
    ALTERNATIVE_HYPOTHESIS,
    NIM,
    PREFERENCE,
    PREFERENCE_TEST,
    TWO_SIDED,
    PREFERENCE_DICT,
    BONFERRONI,
    HOLM,
    HOMMEL,
    SIMES_HOCHBERG,
    SIDAK,
    HOLM_SIDAK,
    FDR_BH,
    FDR_BY,
    FDR_TSBH,
    FDR_TSBKY,
    SPOT_1_HOLM,
    SPOT_1_HOMMEL,
    SPOT_1_SIMES_HOCHBERG,
    SPOT_1_SIDAK,
    SPOT_1_HOLM_SIDAK,
    SPOT_1_FDR_BH,
    SPOT_1_FDR_BY,
    SPOT_1_FDR_TSBH,
    SPOT_1_FDR_TSBKY,
    BONFERRONI_ONLY_COUNT_TWOSIDED,
    BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
    SPOT_1,
    CORRECTION_METHODS,
    BOOTSTRAP,
    CHI2,
    TTEST,
    ZTEST,
    NIM_TYPE,
    CORRECTION_METHODS_THAT_REQUIRE_METRIC_INFO,
)
from spotify_confidence.analysis.frequentist.confidence_computers.bootstrap_computer import BootstrapComputer
from spotify_confidence.analysis.frequentist.confidence_computers.chi_squared_computer import ChiSquaredComputer
from spotify_confidence.analysis.frequentist.confidence_computers.t_test_computer import TTestComputer
from spotify_confidence.analysis.frequentist.confidence_computers.z_test_computer import ZTestComputer
from spotify_confidence.analysis.frequentist.sequential_bound_solver import bounds


def sequential_bounds(t: np.array, alpha: float, sides: int):
    return bounds(t, alpha, rho=2, ztrun=8, sides=sides, max_nints=1000)


class GenericComputer(ConfidenceComputerABC):
    def __init__(
        self,
        data_frame: DataFrame,
        numerator_column: str,
        numerator_sum_squares_column: str,
        denominator_column: str,
        categorical_group_columns: Union[str, Iterable],
        ordinal_group_column: str,
        interval_size: float,
        correction_method: str,
        method_column: str,
        bootstrap_samples_column: str,
        metric_column: Union[str, None],
        treatment_column: Union[str, None],
        power: float,
    ):

        self._df = data_frame
        self._numerator = numerator_column
        self._numerator_sumsq = numerator_sum_squares_column
        if self._numerator is not None and (self._numerator_sumsq is None or self._numerator_sumsq == self._numerator):
            if (data_frame[numerator_column] <= data_frame[denominator_column]).all():
                # Treat as binomial data
                self._numerator_sumsq = self._numerator
            else:
                raise ValueError(
                    f"numerator_sum_squares_column missing or same as "
                    f"numerator_column, but since {numerator_column} is not "
                    f"always smaller than {denominator_column} it can't be "
                    f"binomial data. Please check your data."
                )

        self._denominator = denominator_column
        self._categorical_group_columns = get_all_categorical_group_columns(
            categorical_group_columns, metric_column, treatment_column
        )
        self._segments = remove_group_columns(self._categorical_group_columns, metric_column)
        self._segments = remove_group_columns(self._segments, treatment_column)
        self._ordinal_group_column = ordinal_group_column
        self._metric_column = metric_column
        self._interval_size = interval_size
        self._power = power
        self._treatment_column = treatment_column

        if correction_method.lower() not in CORRECTION_METHODS:
            raise ValueError(f"Use one of the correction methods " + f"in {CORRECTION_METHODS}")
        self._correction_method = correction_method
        self._method_column = method_column

        self._single_metric = False
        if self._metric_column is not None and data_frame.groupby(self._metric_column).ngroups == 1:
            self._single_metric = True
            self._categorical_group_columns = remove_group_columns(
                self._categorical_group_columns, self._metric_column
            )

        self._all_group_columns = get_all_group_columns(self._categorical_group_columns, self._ordinal_group_column)

        self._bootstrap_samples_column = bootstrap_samples_column

        columns_that_must_exist = []
        if (
            CHI2 in self._df[self._method_column]
            or TTEST in self._df[self._method_column]
            or ZTEST in self._df[self._method_column]
        ):
            columns_that_must_exist += [self._numerator, self._denominator]
            columns_that_must_exist += [] if self._numerator_sumsq is None else [self._numerator_sumsq]
        if BOOTSTRAP in self._df[self._method_column]:
            columns_that_must_exist += [self._bootstrap_samples_column]

        validate_data(self._df, columns_that_must_exist, self._all_group_columns, self._ordinal_group_column)

        self._sufficient = None
        self._computers = {
            CHI2: ChiSquaredComputer(
                self._numerator,
                self._numerator_sumsq,
                self._denominator,
                self._ordinal_group_column,
                self._interval_size,
            ),
            TTEST: TTestComputer(
                self._numerator,
                self._numerator_sumsq,
                self._denominator,
                self._ordinal_group_column,
                self._interval_size,
            ),
            ZTEST: ZTestComputer(
                self._numerator,
                self._numerator_sumsq,
                self._denominator,
                self._ordinal_group_column,
                self._interval_size,
            ),
            BOOTSTRAP: BootstrapComputer(self._bootstrap_samples_column, self._interval_size),
        }

    @property
    def _confidence_computers(self):
        return self._computers

    def compute_summary(self, verbose: bool) -> DataFrame:
        return (
            self._sufficient_statistics
            if verbose
            else self._sufficient_statistics[
                self._all_group_columns
                + ([self._metric_column] if self._metric_column is not None and self._single_metric else [])
                + [c for c in [self._numerator, self._denominator] if c is not None]
                + [POINT_ESTIMATE, CI_LOWER, CI_UPPER]
            ]
        )

    @property
    def _sufficient_statistics(self) -> DataFrame:
        if self._sufficient is None:
            self._sufficient = (
                self._df.assign(**{POINT_ESTIMATE: self._point_estimate})
                .assign(**{VARIANCE: self._variance})
                .pipe(self._add_point_estimate_ci)
            )
        return self._sufficient

    def compute_difference(
        self,
        level_1: Union[str, Iterable],
        level_2: Union[str, Iterable],
        absolute: bool,
        groupby: Union[str, Iterable],
        nims: NIM_TYPE,
        final_expected_sample_size_column: str,
        verbose: bool,
        mde_column: str,
    ) -> DataFrame:
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        difference_df = self._compute_differences(
            level_columns=level_columns,
            levels=[(level_1, level_2)],
            absolute=absolute,
            groupby=groupby,
            level_as_reference=True,
            nims=nims,
            final_expected_sample_size_column=final_expected_sample_size_column,
            mde_column=mde_column,
        )
        return (
            difference_df
            if verbose
            else difference_df[
                listify(groupby)
                + ["level_1", "level_2", "absolute_difference", DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE]
                + [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT, POWERED_EFFECT, REQUIRED_SAMPLE_SIZE]
                + ([NIM, NULL_HYPOTHESIS, PREFERENCE] if nims is not None else [])
            ]
        )

    def compute_multiple_difference(
        self,
        level: Union[str, Iterable],
        absolute: bool,
        groupby: Union[str, Iterable],
        level_as_reference: bool,
        nims: NIM_TYPE,
        final_expected_sample_size_column: str,
        verbose: bool,
        mde_column: str,
    ) -> DataFrame:
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        other_levels = [
            other for other in self._sufficient_statistics.groupby(level_columns).groups.keys() if other != level
        ]
        levels = [(level, other) for other in other_levels]
        difference_df = self._compute_differences(
            level_columns=level_columns,
            levels=levels,
            absolute=absolute,
            groupby=groupby,
            level_as_reference=level_as_reference,
            nims=nims,
            final_expected_sample_size_column=final_expected_sample_size_column,
            mde_column=mde_column,
        )
        return (
            difference_df
            if verbose
            else difference_df[
                listify(groupby)
                + [
                    "level_1",
                    "level_2",
                    "absolute_difference",
                    DIFFERENCE,
                    CI_LOWER,
                    CI_UPPER,
                    P_VALUE,
                    POWERED_EFFECT,
                    REQUIRED_SAMPLE_SIZE,
                ]
                + [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT]
                + ([NIM, NULL_HYPOTHESIS, PREFERENCE] if nims is not None else [])
            ]
        )

    def compute_differences(
        self,
        levels: List[Tuple],
        absolute: bool,
        groupby: Union[str, Iterable],
        nims: NIM_TYPE,
        final_expected_sample_size_column: str,
        verbose: bool,
        mde_column: str,
    ) -> DataFrame:
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        difference_df = self._compute_differences(
            level_columns=level_columns,
            levels=[levels] if type(levels) == tuple else levels,
            absolute=absolute,
            groupby=groupby,
            level_as_reference=True,
            nims=nims,
            final_expected_sample_size_column=final_expected_sample_size_column,
            mde_column=mde_column,
        )
        return (
            difference_df
            if verbose
            else difference_df[
                listify(groupby)
                + ["level_1", "level_2", "absolute_difference", DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE]
                + [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT, POWERED_EFFECT, REQUIRED_SAMPLE_SIZE]
                + ([NIM, NULL_HYPOTHESIS, PREFERENCE] if nims is not None else [])
            ]
        )

    def _compute_differences(
        self,
        level_columns: Iterable,
        levels: Union[str, Iterable],
        absolute: bool,
        groupby: Union[str, Iterable],
        level_as_reference: bool,
        nims: NIM_TYPE,
        final_expected_sample_size_column: str,
        mde_column: str,
    ):
        if type(level_as_reference) is not bool:
            raise ValueError(f"level_is_reference must be either True or False, but is {level_as_reference}.")
        groupby = listify(groupby)
        unique_levels = set([l[0] for l in levels] + [l[1] for l in levels])
        validate_levels(self._sufficient_statistics, level_columns, unique_levels)
        str2level = {level2str(lv): lv for lv in unique_levels}
        filtered_sufficient_statistics = concat(
            [self._sufficient_statistics.groupby(level_columns).get_group(group) for group in unique_levels]
        )
        levels = [
            (level2str(l[0]), level2str(l[1])) if level_as_reference else (level2str(l[1]), level2str(l[0]))
            for l in levels
        ]

        def assign_total_denominator(df, groupby):
            if self._denominator is None:
                return df.assign(**{f"current_total_{self._denominator}": None})

            if len(groupby) == 0:
                return df.assign(
                    **{f"current_total_{self._denominator}": self._sufficient_statistics[self._denominator].sum()}
                )
            else:
                return df.merge(
                    df.groupby(groupby)[self._denominator]
                    .sum()
                    .reset_index()
                    .rename(columns={self._denominator: f"current_total_{self._denominator}"})
                )

        return (
            self._sufficient_statistics.assign(
                level=self._sufficient_statistics[level_columns].agg(level2str, axis="columns")
            )
            .pipe(assign_total_denominator, groupby)
            .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
            .pipe(
                self._create_comparison_df,
                groups_to_compare=levels,
                absolute=absolute,
                nims=nims,
                mde_column=mde_column,
                final_expected_sample_size_column=final_expected_sample_size_column,
                filtered_sufficient_statistics=filtered_sufficient_statistics,
            )
            .assign(level_1=lambda df: df["level_1"].map(lambda s: str2level[s]))
            .assign(level_2=lambda df: df["level_2"].map(lambda s: str2level[s]))
            .reset_index()
            .sort_values(by=groupby + ["level_1", "level_2"])
        )

    def _create_comparison_df(
        self,
        df: DataFrame,
        groups_to_compare: List[Tuple[str, str]],
        absolute: bool,
        nims: NIM_TYPE,
        mde_column: bool,
        final_expected_sample_size_column: str,
        filtered_sufficient_statistics: DataFrame,
    ) -> DataFrame:
        def join(df: DataFrame) -> DataFrame:
            has_index = not all(idx is None for idx in df.index.names)
            if has_index:
                # self-join on index (the index will typically model the date,
                # i.e., rows with the same date are joined)
                return df.merge(df, left_index=True, right_index=True, suffixes=(SFX1, SFX2))
            else:
                # join on dummy column, i.e. conduct a cross join
                return (
                    df.assign(dummy_join_column=1)
                    .merge(right=df.assign(dummy_join_column=1), on="dummy_join_column", suffixes=(SFX1, SFX2))
                    .drop(columns="dummy_join_column")
                )

        comparison_df = (
            df.pipe(add_nim_columns, nims=nims)
            .pipe(add_mde_columns, mde_column=mde_column)
            .pipe(join)
            .query(
                f"level_1 in {[l1 for l1, l2 in groups_to_compare]} and "
                + f"level_2 in {[l2 for l1, l2 in groups_to_compare]}"
                + "and level_1 != level_2"
            )
            # TODO: validate_and_rename_mdes
            .pipe(validate_and_rename_columns, NIM)
            .pipe(validate_and_rename_columns, mde_column)
            .pipe(validate_and_rename_columns, PREFERENCE)
            .pipe(validate_and_rename_columns, final_expected_sample_size_column)
            .pipe(validate_and_rename_columns, self._method_column)
            .rename(
                columns={
                    NULL_HYPOTHESIS + SFX1: NULL_HYPOTHESIS,
                    ALTERNATIVE_HYPOTHESIS + SFX1: ALTERNATIVE_HYPOTHESIS,
                    f"current_total_{self._denominator}{SFX1}": f"current_total_{self._denominator}",
                }
            )
            .drop(
                columns=[
                    NULL_HYPOTHESIS + SFX2,
                    ALTERNATIVE_HYPOTHESIS + SFX2,
                    f"current_total_{self._denominator}{SFX2}",
                ]
            )
            .assign(**{DIFFERENCE: lambda df: df[POINT_ESTIMATE + SFX2] - df[POINT_ESTIMATE + SFX1]})
            .assign(**{STD_ERR: self._std_err})
            .pipe(
                self._add_p_value_and_ci,
                final_expected_sample_size_column=final_expected_sample_size_column,
                filtered_sufficient_statistics=filtered_sufficient_statistics,
            )
            .pipe(self._add_adjusted_power)
            .apply(self._powered_effect_and_required_sample_size, mde_column=mde_column, axis=1)
            .pipe(self._adjust_if_absolute, absolute=absolute)
            .assign(**{PREFERENCE: lambda df: df[PREFERENCE].map(PREFERENCE_DICT)})
        )

        return comparison_df

    @staticmethod
    def _adjust_if_absolute(df: DataFrame, absolute: bool) -> DataFrame:
        if absolute:
            return df.assign(absolute_difference=absolute)
        else:
            return (
                df.assign(absolute_difference=absolute)
                .assign(**{DIFFERENCE: df[DIFFERENCE] / df[POINT_ESTIMATE + SFX1]})
                .assign(**{CI_LOWER: df[CI_LOWER] / df[POINT_ESTIMATE + SFX1]})
                .assign(**{CI_UPPER: df[CI_UPPER] / df[POINT_ESTIMATE + SFX1]})
                .assign(**{ADJUSTED_LOWER: df[ADJUSTED_LOWER] / df[POINT_ESTIMATE + SFX1]})
                .assign(**{ADJUSTED_UPPER: df[ADJUSTED_UPPER] / df[POINT_ESTIMATE + SFX1]})
                .assign(**{NULL_HYPOTHESIS: df[NULL_HYPOTHESIS] / df[POINT_ESTIMATE + SFX1]})
                .assign(**{POWERED_EFFECT: df[POWERED_EFFECT] / df[POINT_ESTIMATE + SFX1]})
            )

    def _corrections_power(self, number_of_success_metrics: int, number_of_guardrail_metrics: int) -> int:
        return number_of_guardrail_metrics if number_of_success_metrics == 0 else number_of_guardrail_metrics + 1

    def _add_adjusted_power(self, df: DataFrame) -> DataFrame:
        if self._correction_method in CORRECTION_METHODS_THAT_REQUIRE_METRIC_INFO:
            if self._metric_column is None or self._treatment_column is None:
                return df.assign(**{ADJUSTED_POWER: None})
            else:
                self._number_total_metrics = 1 if self._single_metric else df.groupby(self._metric_column).ngroups
                if self._single_metric:
                    if df[df[NIM].isnull()].shape[0] > 0:
                        self._number_success_metrics = 1
                    else:
                        self._number_success_metrics = 0
                else:
                    self._number_success_metrics = df[df[NIM].isnull()].groupby(self._metric_column).ngroups

                self._number_guardrail_metrics = self._number_total_metrics - self._number_success_metrics
            power_correction = self._corrections_power(
                number_of_guardrail_metrics=self._number_guardrail_metrics,
                number_of_success_metrics=self._number_success_metrics,
            )
            return df.assign(**{ADJUSTED_POWER: 1 - (1 - df[POWER]) / power_correction})
        else:
            return df.assign(**{ADJUSTED_POWER: df[POWER]})

    def _add_p_value_and_ci(
        self, df: DataFrame, final_expected_sample_size_column: str, filtered_sufficient_statistics: DataFrame
    ) -> DataFrame:
        def set_alpha_and_adjust_preference(df: DataFrame) -> DataFrame:
            alpha_0 = 1 - self._interval_size
            return (
                df.assign(
                    **{
                        ALPHA: df.apply(
                            lambda row: 2 * alpha_0
                            if self._correction_method == SPOT_1 and row[PREFERENCE] != TWO_SIDED
                            else alpha_0,
                            axis=1,
                        )
                    }
                )
                .assign(**{POWER: self._power})
                .assign(
                    **{
                        PREFERENCE_TEST: df.apply(
                            lambda row: TWO_SIDED if self._correction_method == SPOT_1 else row[PREFERENCE], axis=1
                        )
                    }
                )
            )

        def _add_adjusted_p_and_is_significant(df: DataFrame) -> DataFrame:
            if final_expected_sample_size_column is not None:
                if self._correction_method not in [
                    BONFERRONI,
                    BONFERRONI_ONLY_COUNT_TWOSIDED,
                    BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
                    SPOT_1,
                ]:
                    raise ValueError(
                        f"{self._correction_method} not supported for sequential tests. Use one of"
                        f"{BONFERRONI}, {BONFERRONI_ONLY_COUNT_TWOSIDED}, "
                        f"{BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY}, {SPOT_1}"
                    )

                groups_except_ordinal = [column for column in df.index.names if column != self._ordinal_group_column]
                n_comparisons = self._get_num_comparisons(
                    df, self._correction_method, ["level_1", "level_2"] + groups_except_ordinal
                )

                df[ADJUSTED_ALPHA] = self._compute_sequential_adjusted_alpha(
                    df, final_expected_sample_size_column, filtered_sufficient_statistics, n_comparisons
                )
                df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE] = df[ALPHA] / n_comparisons
                df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
                df[P_VALUE] = None
                df[ADJUSTED_P] = None
            elif self._correction_method in [
                HOLM,
                HOMMEL,
                SIMES_HOCHBERG,
                SIDAK,
                HOLM_SIDAK,
                FDR_BH,
                FDR_BY,
                FDR_TSBH,
                FDR_TSBKY,
                SPOT_1_HOLM,
                SPOT_1_HOMMEL,
                SPOT_1_SIMES_HOCHBERG,
                SPOT_1_SIDAK,
                SPOT_1_HOLM_SIDAK,
                SPOT_1_FDR_BH,
                SPOT_1_FDR_BY,
                SPOT_1_FDR_TSBH,
                SPOT_1_FDR_TSBKY,
            ]:
                if self._correction_method.startswith("spot-"):
                    correction_method = self._correction_method[7:]
                else:
                    correction_method = self._correction_method

                groupby = ["level_1", "level_2"] + [column for column in df.index.names if column is not None]
                df[ADJUSTED_ALPHA] = df[ALPHA] / self._get_num_comparisons(df, self._correction_method, groupby)
                df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE] = df[ADJUSTED_ALPHA]
                is_significant, adjusted_p, _, _ = multipletests(
                    pvals=df[P_VALUE], alpha=1 - self._interval_size, method=correction_method
                )
                df[ADJUSTED_P] = adjusted_p
                df[IS_SIGNIFICANT] = is_significant
            elif self._correction_method in [
                BONFERRONI,
                BONFERRONI_ONLY_COUNT_TWOSIDED,
                BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
                SPOT_1,
            ]:
                groupby = ["level_1", "level_2"] + [column for column in df.index.names if column is not None]
                n_comparisons = self._get_num_comparisons(df, self._correction_method, groupby)
                df[ADJUSTED_ALPHA] = df[ALPHA] / n_comparisons
                df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE] = df[ADJUSTED_ALPHA]
                df[ADJUSTED_P] = df[P_VALUE].map(lambda p: min(p * n_comparisons, 1))
                df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
            else:
                raise ValueError("Can't figure out which correction method to use :(")

            return df

        def _add_ci(df: DataFrame) -> DataFrame:
            ci = df.apply(self._ci, axis=1, alpha_column=ALPHA)
            ci_df = DataFrame(index=ci.index, columns=[CI_LOWER, CI_UPPER], data=list(ci.values))

            if (
                self._correction_method
                in [
                    HOLM,
                    HOMMEL,
                    SIMES_HOCHBERG,
                    SPOT_1_HOLM,
                    SPOT_1_HOMMEL,
                    SPOT_1_SIMES_HOCHBERG,
                ]
                and all(df[PREFERENCE_TEST] != TWO_SIDED)
            ):
                adjusted_ci = self._ci_for_multiple_comparison_methods(
                    df,
                    correction_method=self._correction_method,
                    alpha=1 - self._interval_size,
                )
            elif self._correction_method in [
                BONFERRONI,
                BONFERRONI_ONLY_COUNT_TWOSIDED,
                BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
                SPOT_1,
                SPOT_1_HOLM,
                SPOT_1_HOMMEL,
                SPOT_1_SIMES_HOCHBERG,
                SPOT_1_SIDAK,
                SPOT_1_HOLM_SIDAK,
                SPOT_1_FDR_BH,
                SPOT_1_FDR_BY,
                SPOT_1_FDR_TSBH,
                SPOT_1_FDR_TSBKY,
            ]:
                adjusted_ci = df.apply(self._ci, axis=1, alpha_column=ADJUSTED_ALPHA)
            else:
                warn(f"Confidence intervals not supported for {self._correction_method}")
                adjusted_ci = Series(index=df.index, data=[(None, None) for i in range(len(df))])

            adjusted_ci_df = DataFrame(
                index=adjusted_ci.index, columns=[ADJUSTED_LOWER, ADJUSTED_UPPER], data=list(adjusted_ci.values)
            )

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

    def _get_num_comparisons(self, df: DataFrame, correction_method: str, groupby: Iterable) -> int:
        if correction_method == BONFERRONI:
            return max(1, df.groupby(groupby).ngroups)
        elif correction_method == BONFERRONI_ONLY_COUNT_TWOSIDED:
            return max(df.query(f'{PREFERENCE_TEST} == "{TWO_SIDED}"').groupby(groupby).ngroups, 1)
        elif correction_method in [
            HOLM,
            HOMMEL,
            SIMES_HOCHBERG,
            SIDAK,
            HOLM_SIDAK,
            FDR_BH,
            FDR_BY,
            FDR_TSBH,
            FDR_TSBKY,
        ]:
            return 1
        elif correction_method in [
            BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
            SPOT_1,
            SPOT_1_HOLM,
            SPOT_1_HOMMEL,
            SPOT_1_SIMES_HOCHBERG,
            SPOT_1_SIDAK,
            SPOT_1_HOLM_SIDAK,
            SPOT_1_FDR_BH,
            SPOT_1_FDR_BY,
            SPOT_1_FDR_TSBH,
            SPOT_1_FDR_TSBKY,
        ]:
            if self._metric_column is None or self._treatment_column is None:
                return max(1, df[df[NIM].isnull()].groupby(groupby).ngroups)
            else:
                if self._single_metric:
                    if df[df[NIM].isnull()].shape[0] > 0:
                        self._number_success_metrics = 1
                    else:
                        self._number_success_metrics = 0
                else:
                    self._number_success_metrics = df[df[NIM].isnull()].groupby(self._metric_column).ngroups

                number_comparions = len(
                    (df[self._treatment_column + SFX1] + df[self._treatment_column + SFX2]).unique()
                )
                number_segments = (
                    1
                    if len(self._segments) == 0 or not all(item in df.index.names for item in self._segments)
                    else df.groupby(self._segments).ngroups
                )

                return max(1, number_comparions * max(1, self._number_success_metrics) * number_segments)
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
            self._compute_differences(
                level_columns,
                [(level_1, level_2)],
                True,
                groupby,
                level_as_reference=True,
                nims=None,
                final_expected_sample_size_column=None,
                mde_column=None,
            )  # TODO: IS this right?
            .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
            .assign(achieved_power=lambda df: df.apply(self._achieved_power, mde=mde, alpha=alpha, axis=1))
        )[["level_1", "level_2", "achieved_power"]]

    def _point_estimate(self, df: DataFrame) -> Series:
        return df.apply(lambda row: self._confidence_computers[row[self._method_column]]._point_estimate(row), axis=1)

    def _variance(self, df: DataFrame) -> Series:
        return df.apply(lambda row: self._confidence_computers[row[self._method_column]]._variance(row), axis=1)

    def _std_err(self, df: DataFrame) -> Series:
        return df.apply(lambda row: self._confidence_computers[row[self._method_column]]._std_err(row), axis=1)

    def _add_point_estimate_ci(self, df: DataFrame) -> DataFrame:
        return df.apply(
            lambda row: self._confidence_computers[row[self._method_column]]._add_point_estimate_ci(row), axis=1
        )

    def _p_value(self, row) -> float:
        if row[self._method_column] == CHI2 and row[NIM] is not None:
            raise ValueError(
                "Non-inferiority margins not supported in ChiSquared. Use StudentsTTest or ZTest instead."
            )
        return self._confidence_computers[row[self._method_column]]._p_value(row)

    def _ci(self, row, alpha_column: str) -> Tuple[float, float]:
        return self._confidence_computers[row[self._method_column]]._ci(row, alpha_column=alpha_column)

    def _powered_effect_and_required_sample_size(self, row: Series, mde_column: str) -> DataFrame:
        if row[self._method_column] != ZTEST and mde_column in row:
            raise ValueError("Minimum detectable effects only supported for ZTest.")
        elif row[self._method_column] != ZTEST or row[ADJUSTED_POWER] is None:
            row[POWERED_EFFECT] = None
            row[REQUIRED_SAMPLE_SIZE] = None
            return row
        else:
            return self._confidence_computers[row[self._method_column]]._powered_effect_and_required_sample_size(row)

    def _achieved_power(self, row: Series, mde: float, alpha: float) -> DataFrame:
        return self._confidence_computers[row[self._method_column]]._achieved_power(row, mde, alpha)

    def _compute_sequential_adjusted_alpha(
        self,
        df: DataFrame,
        final_expected_sample_size_column: str,
        filtered_sufficient_statistics: DataFrame,
        n_comparisons: int,
    ) -> Series:
        if all(df[self._method_column] == "z-test"):
            return self._confidence_computers["z-test"]._compute_sequential_adjusted_alpha(
                df, final_expected_sample_size_column, filtered_sufficient_statistics, n_comparisons
            )
        else:
            raise NotImplementedError("Sequential testing is only supported for z-tests")

    def _ci_for_multiple_comparison_methods(
        self,
        df: DataFrame,
        correction_method: str,
        alpha: float,
        w: float = 1.0,
    ) -> Tuple[Union[Series, float], Union[Series, float]]:
        if all(df[self._method_column] == "z-test"):
            return self._confidence_computers["z-test"]._ci_for_multiple_comparison_methods(
                df, correction_method, alpha, w
            )
        else:
            raise NotImplementedError(f"{self._correction_method} is only supported for ZTests")
