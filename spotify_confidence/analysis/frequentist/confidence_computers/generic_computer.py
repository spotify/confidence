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

from typing import Union, Iterable, List, Tuple, Dict
from warnings import warn

import numpy as np
from numpy import isnan
from pandas import DataFrame, Series
from scipy import stats as st
from statsmodels.stats.multitest import multipletests

import spotify_confidence.analysis.frequentist.confidence_computers.bootstrap_computer as bootstrap_computer
import spotify_confidence.analysis.frequentist.confidence_computers.chi_squared_computer as chi_squared_computer
import spotify_confidence.analysis.frequentist.confidence_computers.t_test_computer as t_test_computer
import spotify_confidence.analysis.frequentist.confidence_computers.z_test_computer as z_test_computers
import spotify_confidence.analysis.frequentist.confidence_computers.z_test_linreg_computer as z_test_linreg_computer
from spotify_confidence.analysis.abstract_base_classes.confidence_computer_abc import ConfidenceComputerABC
from spotify_confidence.analysis.confidence_utils import (
    get_remaning_groups,
    validate_levels,
    level2str,
    listify,
    validate_and_rename_columns,
    drop_and_rename_columns,
    get_all_categorical_group_columns,
    get_all_group_columns,
    validate_data,
    remove_group_columns,
    groupbyApplyParallel,
    is_non_inferiority,
    reset_named_indices,
)
from spotify_confidence.analysis.constants import (
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    DENOMINATOR,
    BOOTSTRAPS,
    INTERVAL_SIZE,
    POINT_ESTIMATE,
    FINAL_EXPECTED_SAMPLE_SIZE,
    ORDINAL_GROUP_COLUMN,
    MDE,
    METHOD,
    CORRECTION_METHOD,
    ABSOLUTE,
    VARIANCE,
    NUMBER_OF_COMPARISONS,
    TREATMENT_WEIGHTS,
    IS_BINARY,
    FEATURE,
    FEATURE_SUMSQ,
    FEATURE_CROSS,
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
    REQUIRED_SAMPLE_SIZE_METRIC,
    OPTIMAL_KAPPA,
    OPTIMAL_WEIGHTS,
    CI_WIDTH,
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
    NIM_COLUMN_DEFAULT,
    PREFERRED_DIRECTION_COLUMN_DEFAULT,
    INCREASE_PREFFERED,
    DECREASE_PREFFERED,
    ZTESTLINREG,
    ORIGINAL_POINT_ESTIMATE,
    ORIGINAL_VARIANCE,
    VARIANCE_REDUCTION,
)

confidence_computers = {
    CHI2: chi_squared_computer,
    TTEST: t_test_computer,
    ZTEST: z_test_computers,
    BOOTSTRAP: bootstrap_computer,
    ZTESTLINREG: z_test_linreg_computer,
}


class GenericComputer(ConfidenceComputerABC):
    def __init__(
        self,
        data_frame: DataFrame,
        numerator_column: Union[str, None],
        numerator_sum_squares_column: Union[str, None],
        denominator_column: Union[str, None],
        categorical_group_columns: Union[str, Iterable],
        ordinal_group_column: Union[str, None],
        interval_size: float,
        correction_method: str,
        method_column: str,
        bootstrap_samples_column: Union[str, None],
        metric_column: Union[str, None],
        treatment_column: Union[str, None],
        power: float,
        point_estimate_column: str,
        var_column: str,
        is_binary_column: str,
        feature_column: Union[str, None],
        feature_sum_squares_column: Union[str, None],
        feature_cross_sum_column: Union[str, None],
    ):

        self._df = data_frame.reset_index(drop=True)
        self._point_estimate_column = point_estimate_column
        self._var_column = var_column
        self._is_binary = is_binary_column
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
        self._feature = feature_column
        self._feature_ssq = feature_sum_squares_column
        self._feature_cross = feature_cross_sum_column

        if correction_method.lower() not in CORRECTION_METHODS:
            raise ValueError(f"Use one of the correction methods " + f"in {CORRECTION_METHODS}")
        self._correction_method = correction_method
        self._method_column = method_column

        self._single_metric = False
        if self._metric_column is not None and data_frame.groupby(self._metric_column, sort=False).ngroups == 1:
            self._single_metric = True

        self._all_group_columns = get_all_group_columns(self._categorical_group_columns, self._ordinal_group_column)
        self._bootstrap_samples_column = bootstrap_samples_column

        columns_that_must_exist = []
        if (
            CHI2 in self._df[self._method_column]
            or TTEST in self._df[self._method_column]
            or ZTEST in self._df[self._method_column]
        ):
            if not self._point_estimate_column or not self._var_column:
                columns_that_must_exist += [self._numerator, self._denominator]
                columns_that_must_exist += [] if self._numerator_sumsq is None else [self._numerator_sumsq]
            else:
                columns_that_must_exist += [self._point_estimate_column, self._var_column]
        if BOOTSTRAP in self._df[self._method_column]:
            columns_that_must_exist += [self._bootstrap_samples_column]
        if ZTESTLINREG in self._df[self._method_column]:
            columns_that_must_exist += [self._feature, self._feature_ssq, self._feature_cross]

        validate_data(self._df, columns_that_must_exist, self._all_group_columns, self._ordinal_group_column)

        self._sufficient = None

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
            arg_dict = {
                NUMERATOR: self._numerator,
                NUMERATOR_SUM_OF_SQUARES: self._numerator_sumsq,
                DENOMINATOR: self._denominator,
                BOOTSTRAPS: self._bootstrap_samples_column,
                INTERVAL_SIZE: self._interval_size,
                FEATURE: self._feature,
                FEATURE_SUMSQ: self._feature_ssq,
                FEATURE_CROSS: self._feature_cross,
            }
            groupby = [col for col in [self._method_column, self._metric_column] if col is not None]
            self._sufficient = (
                self._df.groupby(groupby, sort=False)
                .apply(
                    lambda df: df.assign(
                        **{
                            POINT_ESTIMATE: lambda df: df[self._point_estimate_column]
                            if self._point_estimate_column is not None
                            else confidence_computers[df[self._method_column].values[0]].point_estimate(df, arg_dict)
                        }
                    )
                    .assign(
                        **{
                            VARIANCE: lambda df: df[self._var_column]
                            if self._var_column is not None
                            else confidence_computers[df[self._method_column].values[0]].variance(df, arg_dict)
                        }
                    )
                    .pipe(
                        lambda df: df
                        if self._point_estimate_column is not None
                        else confidence_computers[df[self._method_column].values[0]].add_point_estimate_ci(
                            df, arg_dict
                        )
                    )
                )
                .pipe(reset_named_indices)
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
            other
            for other in self._sufficient_statistics.groupby(level_columns, sort=False).groups.keys()
            if other != level
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
                    df.groupby(groupby, sort=False)[self._denominator]
                    .sum()
                    .reset_index()
                    .rename(columns={self._denominator: f"current_total_{self._denominator}"})
                )

        return (
            self._sufficient_statistics.assign(
                level=self._sufficient_statistics[level_columns].agg(level2str, axis="columns")
            )
            .pipe(assign_total_denominator, groupby)
            .query(f"level in {[l1 for l1,l2 in levels] + [l2 for l1,l2 in levels]}")
            .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
            .pipe(
                self._create_comparison_df,
                groups_to_compare=levels,
                absolute=absolute,
                nims=nims,
                mde_column=mde_column,
                final_expected_sample_size_column=final_expected_sample_size_column,
            )
            .assign(level_1=lambda df: df["level_1"].map(lambda s: str2level[s]))
            .assign(level_2=lambda df: df["level_2"].map(lambda s: str2level[s]))
            .pipe(lambda df: df.reset_index([name for name in df.index.names if name is not None]))
            .reset_index(drop=True)
            .sort_values(by=groupby + ["level_1", "level_2"])
            .reset_index(drop=True)
        )

    def _create_comparison_df(
        self,
        df: DataFrame,
        groups_to_compare: List[Tuple[str, str]],
        absolute: bool,
        nims: NIM_TYPE,
        mde_column: bool,
        final_expected_sample_size_column: str,
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
            df.pipe(add_nim_input_columns_from_tuple_or_dict, nims=nims, mde_column=mde_column)
            .pipe(
                add_nims_and_mdes,
                mde_column=mde_column,
                nim_column=NIM_COLUMN_DEFAULT,
                preferred_direction_column=PREFERRED_DIRECTION_COLUMN_DEFAULT,
            )
            .pipe(join)
            .query(
                "("
                + " or ".join([f"(level_1=='{l1}' and level_2=='{l2}')" for l1, l2 in groups_to_compare])
                + ")"
                + "and level_1 != level_2"
            )
            .pipe(
                validate_and_rename_columns,
                [NIM, mde_column, PREFERENCE, final_expected_sample_size_column, self._method_column],
            )
            .pipe(
                drop_and_rename_columns,
                [NULL_HYPOTHESIS, ALTERNATIVE_HYPOTHESIS, f"current_total_{self._denominator}"]
                + ([ORIGINAL_POINT_ESTIMATE] if ORIGINAL_POINT_ESTIMATE in df.columns else []),
            )
            .assign(**{PREFERENCE_TEST: lambda df: TWO_SIDED if self._correction_method == SPOT_1 else df[PREFERENCE]})
            .assign(**{POWER: self._power})
            .pipe(self._add_adjusted_power)
        )

        groups_except_ordinal = [
            column
            for column in df.index.names
            if column is not None
            and (column != self._ordinal_group_column or final_expected_sample_size_column is None)
        ]
        n_comparisons = self._get_num_comparisons(
            comparison_df,
            self._correction_method,
            number_of_level_comparisons=comparison_df.groupby(["level_1", "level_2"], sort=False).ngroups,
            groupby=groups_except_ordinal,
        )

        arg_dict = {
            NUMERATOR: self._numerator,
            NUMERATOR_SUM_OF_SQUARES: self._numerator_sumsq,
            DENOMINATOR: self._denominator,
            BOOTSTRAPS: self._bootstrap_samples_column,
            FINAL_EXPECTED_SAMPLE_SIZE: final_expected_sample_size_column,
            ORDINAL_GROUP_COLUMN: self._ordinal_group_column,
            MDE: mde_column,
            METHOD: self._method_column,
            CORRECTION_METHOD: self._correction_method,
            INTERVAL_SIZE: self._interval_size,
            ABSOLUTE: absolute,
            NUMBER_OF_COMPARISONS: n_comparisons,
        }
        comparison_df = groupbyApplyParallel(
            comparison_df.groupby(groups_except_ordinal + [self._method_column], as_index=False, sort=False),
            lambda df: _compute_comparisons(df, arg_dict=arg_dict),
        )
        return comparison_df

    def compute_sample_size(
        self,
        treatment_weights: Iterable,
        mde_column: str,
        nim_column: str,
        preferred_direction_column: str,
        final_expected_sample_size_column: str,
    ) -> DataFrame:
        arg_dict, group_columns, sample_size_df = self._initialise_sample_size_and_power_computation(
            final_expected_sample_size_column, mde_column, nim_column, preferred_direction_column, treatment_weights
        )
        sample_size_df = groupbyApplyParallel(
            sample_size_df.pipe(set_alpha_and_adjust_preference, arg_dict=arg_dict).groupby(
                group_columns + [self._method_column],
                as_index=False,
                sort=False,
            ),
            lambda df: _compute_sample_sizes_and_ci_widths(df, arg_dict=arg_dict),
        )

        return sample_size_df.reset_index()

    def compute_powered_effect(
        self,
        treatment_weights: Iterable,
        mde_column: str,
        nim_column: str,
        preferred_direction_column: str,
        sample_size: float,
    ) -> DataFrame:
        arg_dict, group_columns, powered_effect_df = self._initialise_sample_size_and_power_computation(
            sample_size, mde_column, nim_column, preferred_direction_column, treatment_weights
        )
        powered_effect_df = groupbyApplyParallel(
            powered_effect_df.pipe(set_alpha_and_adjust_preference, arg_dict=arg_dict).groupby(
                group_columns + [self._method_column],
                as_index=False,
                sort=False,
            ),
            lambda df: _compute_powered_effects(df, arg_dict=arg_dict),
        )

        return powered_effect_df.reset_index()

    def _initialise_sample_size_and_power_computation(
        self, final_expected_sample_size_column, mde_column, nim_column, preferred_direction_column, treatment_weights
    ):
        sample_size_df = (
            self._sufficient_statistics.pipe(
                lambda df: df if self._all_group_columns == [] else df.set_index(self._all_group_columns)
            )
            .pipe(
                add_nims_and_mdes,
                mde_column=mde_column,
                nim_column=nim_column,
                preferred_direction_column=preferred_direction_column,
            )
            .assign(**{PREFERENCE_TEST: lambda df: TWO_SIDED if self._correction_method == SPOT_1 else df[PREFERENCE]})
            .assign(**{POWER: self._power})
            .pipe(self._add_adjusted_power)
        )
        group_columns = [column for column in sample_size_df.index.names if column is not None]
        n_comparisons = self._get_num_comparisons(
            sample_size_df,
            self._correction_method,
            number_of_level_comparisons=len(treatment_weights) - 1,
            groupby=group_columns,
        )
        arg_dict = {
            MDE: mde_column,
            METHOD: self._method_column,
            NUMBER_OF_COMPARISONS: n_comparisons,
            TREATMENT_WEIGHTS: treatment_weights,
            INTERVAL_SIZE: self._interval_size,
            CORRECTION_METHOD: self._correction_method,
            IS_BINARY: self._is_binary,
            FINAL_EXPECTED_SAMPLE_SIZE: final_expected_sample_size_column,
        }
        return arg_dict, group_columns, sample_size_df

    def compute_optimal_weights_and_sample_size(
        self, sample_size_df: DataFrame, number_of_groups: int
    ) -> Tuple[Iterable, int]:
        sample_size_df = (
            sample_size_df.reset_index(drop=True)
            .assign(**{OPTIMAL_KAPPA: lambda df: df.apply(_optimal_kappa, is_binary_column=self._is_binary, axis=1)})
            .assign(
                **{
                    OPTIMAL_WEIGHTS: lambda df: df.apply(
                        lambda row: _optimal_weights(row[OPTIMAL_KAPPA], number_of_groups), axis=1
                    )
                }
            )
        )

        group_columns = [column for column in sample_size_df.index.names if column is not None] + [self._method_column]
        arg_dict = {
            METHOD: self._method_column,
            IS_BINARY: self._is_binary,
        }
        return _find_optimal_group_weights_across_rows(sample_size_df, number_of_groups, group_columns, arg_dict)

    def _add_adjusted_power(self, df: DataFrame) -> DataFrame:
        if self._correction_method in CORRECTION_METHODS_THAT_REQUIRE_METRIC_INFO:
            if self._metric_column is None:
                return df.assign(**{ADJUSTED_POWER: None})
            else:
                number_total_metrics = (
                    1 if self._single_metric else df.groupby(self._metric_column, sort=False).ngroups
                )
                if self._single_metric:
                    if df[df[NIM].isnull()].shape[0] > 0:
                        number_success_metrics = 1
                    else:
                        number_success_metrics = 0
                else:
                    number_success_metrics = df[df[NIM].isnull()].groupby(self._metric_column, sort=False).ngroups

                number_guardrail_metrics = number_total_metrics - number_success_metrics
                power_correction = (
                    number_guardrail_metrics if number_success_metrics == 0 else number_guardrail_metrics + 1
                )
                return df.assign(**{ADJUSTED_POWER: 1 - (1 - df[POWER]) / power_correction})
        else:
            return df.assign(**{ADJUSTED_POWER: df[POWER]})

    def achieved_power(self, level_1, level_2, mde, alpha, groupby):
        groupby = listify(groupby)
        level_columns = get_remaning_groups(self._all_group_columns, groupby)
        arg_dict = {NUMERATOR: self._numerator, DENOMINATOR: self._denominator}
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
            .assign(
                achieved_power=lambda df: df.apply(
                    lambda row: confidence_computers[row[self._method_column]].achieved_power(
                        row, mde=mde, alpha=alpha, arg_dict=arg_dict
                    ),
                    axis=1,
                )
            )
        )[["level_1", "level_2", "achieved_power"]]

    def _get_num_comparisons(
        self, df: DataFrame, correction_method: str, number_of_level_comparisons: int, groupby: Iterable
    ) -> int:
        if correction_method == BONFERRONI:
            return max(
                1,
                number_of_level_comparisons * df.assign(_dummy_=1).groupby(groupby + ["_dummy_"], sort=False).ngroups,
            )
        elif correction_method == BONFERRONI_ONLY_COUNT_TWOSIDED:
            return max(
                number_of_level_comparisons
                * df.query(f'{PREFERENCE_TEST} == "{TWO_SIDED}"')
                .assign(_dummy_=1)
                .groupby(groupby + ["_dummy_"], sort=False)
                .ngroups,
                1,
            )
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
                return max(
                    1,
                    number_of_level_comparisons
                    * df[df[NIM].isnull()].assign(_dummy_=1).groupby(groupby + ["_dummy_"], sort=False).ngroups,
                )
            else:
                if self._single_metric:
                    if df[df[NIM].isnull()].shape[0] > 0:
                        number_success_metrics = 1
                    else:
                        number_success_metrics = 0
                else:
                    number_success_metrics = df[df[NIM].isnull()].groupby(self._metric_column, sort=False).ngroups

                number_segments = (
                    1
                    if len(self._segments) == 0 or not all(item in df.index.names for item in self._segments)
                    else df.groupby(self._segments, sort=False).ngroups
                )

                return max(1, number_of_level_comparisons * max(1, number_success_metrics) * number_segments)
        else:
            raise ValueError(f"Unsupported correction method: {correction_method}.")


def add_nim_input_columns_from_tuple_or_dict(df, nims: NIM_TYPE, mde_column: str) -> DataFrame:
    if type(nims) is tuple:
        return df.assign(**{NIM_COLUMN_DEFAULT: nims[0]}).assign(**{PREFERRED_DIRECTION_COLUMN_DEFAULT: nims[1]})
    elif type(nims) is dict:
        nim_values = {key: value[0] for key, value in nims.items()}
        nim_preferences = {key: value[1] for key, value in nims.items()}
        return df.assign(**{NIM_COLUMN_DEFAULT: lambda df: df.index.to_series().map(nim_values)}).assign(
            **{PREFERRED_DIRECTION_COLUMN_DEFAULT: lambda df: df.index.to_series().map(nim_preferences)}
        )
    elif nims is None:
        return df.assign(**{NIM_COLUMN_DEFAULT: None}).assign(
            **{
                PREFERRED_DIRECTION_COLUMN_DEFAULT: None
                if PREFERRED_DIRECTION_COLUMN_DEFAULT not in df or mde_column is None
                else df[PREFERRED_DIRECTION_COLUMN_DEFAULT]
            }
        )
    else:
        return df


def add_nims_and_mdes(df: DataFrame, mde_column: str, nim_column: str, preferred_direction_column: str) -> DataFrame:
    def _set_nims_and_mdes(grp: DataFrame) -> DataFrame:
        nim = grp[nim_column].astype(float)
        input_preference = grp[preferred_direction_column].values[0]
        mde = None if mde_column is None else grp[mde_column]

        nim_is_na = nim.isna().all()
        mde_is_na = True if mde is None else mde.isna().all()
        if input_preference is None or (type(input_preference) is float and isnan(input_preference)):
            signed_nim = 0.0 if nim_is_na else nim * grp[POINT_ESTIMATE]
            preference = TWO_SIDED
            signed_mde = None if mde_is_na else mde * grp[POINT_ESTIMATE]
        elif input_preference.lower() == INCREASE_PREFFERED:
            signed_nim = 0.0 if nim_is_na else -nim * grp[POINT_ESTIMATE]
            preference = "larger"
            signed_mde = None if mde_is_na else mde * grp[POINT_ESTIMATE]
        elif input_preference.lower() == DECREASE_PREFFERED:
            signed_nim = 0.0 if nim_is_na else nim * grp[POINT_ESTIMATE]
            preference = "smaller"
            signed_mde = None if mde_is_na else -mde * grp[POINT_ESTIMATE]
        else:
            raise ValueError(f"{input_preference.lower()} not in " f"{[INCREASE_PREFFERED, DECREASE_PREFFERED]}")

        return (
            grp.assign(**{NIM: nim})
            .assign(**{PREFERENCE: preference})
            .assign(**{NULL_HYPOTHESIS: signed_nim})
            .assign(**{ALTERNATIVE_HYPOTHESIS: signed_mde if nim_is_na else 0.0})
        )

    index_names = [name for name in df.index.names if name is not None]
    return (
        df.groupby(
            [nim_column, preferred_direction_column] + listify(mde_column), dropna=False, as_index=False, sort=False
        )
        .apply(_set_nims_and_mdes)
        .pipe(lambda df: df.reset_index(index_names))
        .reset_index(drop=True)
        .pipe(lambda df: df if index_names == [] else df.set_index(index_names))
    )


def _compute_comparisons(df: DataFrame, arg_dict: Dict) -> DataFrame:
    return (
        df.assign(**{DIFFERENCE: lambda df: df[POINT_ESTIMATE + SFX2] - df[POINT_ESTIMATE + SFX1]})
        .assign(**{STD_ERR: confidence_computers[df[arg_dict[METHOD]].values[0]].std_err(df, arg_dict)})
        .pipe(_add_p_value_and_ci, arg_dict=arg_dict)
        .pipe(_powered_effect_and_required_sample_size_from_difference_df, arg_dict=arg_dict)
        .pipe(_adjust_if_absolute, absolute=arg_dict[ABSOLUTE])
        .assign(**{PREFERENCE: lambda df: df[PREFERENCE].map(PREFERENCE_DICT)})
        .pipe(_add_variance_reduction_rate, arg_dict=arg_dict)
    )


def _add_variance_reduction_rate(df: DataFrame, arg_dict: Dict) -> DataFrame:
    denominator = arg_dict[DENOMINATOR]
    method_column = arg_dict[METHOD]
    if (df[method_column] == ZTESTLINREG).any():
        variance_no_reduction = (
            df[ORIGINAL_VARIANCE + SFX1] / df[denominator + SFX1]
            + df[ORIGINAL_VARIANCE + SFX2] / df[denominator + SFX2]
        )
        variance_w_reduction = (
            df[VARIANCE + SFX1] / df[denominator + SFX1] + df[VARIANCE + SFX2] / df[denominator + SFX2]
        )
        df = df.assign(**{VARIANCE_REDUCTION: 1 - np.divide(variance_w_reduction, variance_no_reduction)})
    return df


def _add_p_value_and_ci(df: DataFrame, arg_dict: Dict) -> DataFrame:
    def _add_adjusted_p_and_is_significant(df: DataFrame, arg_dict: Dict) -> DataFrame:
        n_comparisons = arg_dict[NUMBER_OF_COMPARISONS]
        if arg_dict[FINAL_EXPECTED_SAMPLE_SIZE] is not None:
            if arg_dict[CORRECTION_METHOD] not in [
                BONFERRONI,
                BONFERRONI_ONLY_COUNT_TWOSIDED,
                BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
                SPOT_1,
            ]:
                raise ValueError(
                    f"{arg_dict[CORRECTION_METHOD]} not supported for sequential tests. Use one of"
                    f"{BONFERRONI}, {BONFERRONI_ONLY_COUNT_TWOSIDED}, "
                    f"{BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY}, {SPOT_1}"
                )
            adjusted_alpha = _compute_sequential_adjusted_alpha(df, arg_dict[METHOD], arg_dict)
            df = df.merge(adjusted_alpha, left_index=True, right_index=True)
            df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
            df[P_VALUE] = None
            df[ADJUSTED_P] = None
        elif arg_dict[CORRECTION_METHOD] in [
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
            if arg_dict[CORRECTION_METHOD].startswith("spot-"):
                correction_method = arg_dict[CORRECTION_METHOD][7:]
            else:
                correction_method = arg_dict[CORRECTION_METHOD]
            df[ADJUSTED_ALPHA] = df[ALPHA] / n_comparisons
            is_significant, adjusted_p, _, _ = multipletests(
                pvals=df[P_VALUE], alpha=1 - arg_dict[INTERVAL_SIZE], method=correction_method
            )
            df[ADJUSTED_P] = adjusted_p
            df[IS_SIGNIFICANT] = is_significant
        elif arg_dict[CORRECTION_METHOD] in [
            BONFERRONI,
            BONFERRONI_ONLY_COUNT_TWOSIDED,
            BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
            SPOT_1,
        ]:
            df[ADJUSTED_ALPHA] = df[ALPHA] / n_comparisons
            df[ADJUSTED_P] = df[P_VALUE].map(lambda p: min(p * n_comparisons, 1))
            df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
        else:
            raise ValueError("Can't figure out which correction method to use :(")

        return df

    def _compute_sequential_adjusted_alpha(df: DataFrame, method_column: str, arg_dict: Dict) -> Series:
        if all(df[method_column] == "z-test"):
            return confidence_computers["z-test"].compute_sequential_adjusted_alpha(df, arg_dict)
        else:
            raise NotImplementedError("Sequential testing is only supported for z-tests")

    def _add_ci(df: DataFrame, arg_dict: Dict) -> DataFrame:
        lower, upper = confidence_computers[df[arg_dict[METHOD]].values[0]].ci(df, ALPHA, arg_dict)

        if (
            arg_dict[CORRECTION_METHOD]
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
            if all(df[arg_dict[METHOD]] == "z-test"):
                adjusted_lower, adjusted_upper = confidence_computers["z-test"].ci_for_multiple_comparison_methods(
                    df, arg_dict[CORRECTION_METHOD], alpha=1 - arg_dict[INTERVAL_SIZE]
                )
            else:
                raise NotImplementedError(f"{arg_dict[CORRECTION_METHOD]} is only supported for ZTests")
        elif arg_dict[CORRECTION_METHOD] in [
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
            adjusted_lower, adjusted_upper = confidence_computers[df[arg_dict[METHOD]].values[0]].ci(
                df, ADJUSTED_ALPHA, arg_dict
            )
        else:
            warn(f"Confidence intervals not supported for {arg_dict[CORRECTION_METHOD]}")
            adjusted_lower = None
            adjusted_upper = None

        return (
            df.assign(**{CI_LOWER: lower})
            .assign(**{CI_UPPER: upper})
            .assign(**{ADJUSTED_LOWER: adjusted_lower})
            .assign(**{ADJUSTED_UPPER: adjusted_upper})
        )

    return (
        df.pipe(set_alpha_and_adjust_preference, arg_dict=arg_dict)
        .assign(**{P_VALUE: lambda df: df.pipe(_p_value, arg_dict=arg_dict)})
        .pipe(_add_adjusted_p_and_is_significant, arg_dict=arg_dict)
        .pipe(_add_ci, arg_dict=arg_dict)
    )


def set_alpha_and_adjust_preference(df: DataFrame, arg_dict: Dict) -> DataFrame:
    alpha_0 = 1 - arg_dict[INTERVAL_SIZE]
    return df.assign(
        **{
            ALPHA: df.apply(
                lambda row: 2 * alpha_0
                if arg_dict[CORRECTION_METHOD] == SPOT_1 and row[PREFERENCE] != TWO_SIDED
                else alpha_0,
                axis=1,
            )
        }
    ).assign(**{ADJUSTED_ALPHA_POWER_SAMPLE_SIZE: lambda df: df[ALPHA] / arg_dict[NUMBER_OF_COMPARISONS]})


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


def _p_value(df: DataFrame, arg_dict: Dict) -> float:
    if df[arg_dict[METHOD]].values[0] == CHI2 and (df[NIM].notna()).any():
        raise ValueError("Non-inferiority margins not supported in ChiSquared. Use StudentsTTest or ZTest instead.")
    return confidence_computers[df[arg_dict[METHOD]].values[0]].p_value(df, arg_dict)


def _powered_effect_and_required_sample_size_from_difference_df(df: DataFrame, arg_dict: Dict) -> DataFrame:
    if df[arg_dict[METHOD]].values[0] not in [ZTEST, ZTESTLINREG] and arg_dict[MDE] in df:
        raise ValueError("Minimum detectable effects only supported for ZTest.")
    elif df[arg_dict[METHOD]].values[0] not in [ZTEST, ZTESTLINREG] or (df[ADJUSTED_POWER].isna()).any():
        df[POWERED_EFFECT] = None
        df[REQUIRED_SAMPLE_SIZE] = None
        df[REQUIRED_SAMPLE_SIZE_METRIC] = None
        return df
    else:
        n1, n2 = df[arg_dict[DENOMINATOR] + SFX1], df[arg_dict[DENOMINATOR] + SFX2]
        kappa = n1 / n2
        binary = (df[arg_dict[NUMERATOR_SUM_OF_SQUARES] + SFX1] == df[arg_dict[NUMERATOR] + SFX1]).all()
        proportion_of_total = (n1 + n2) / df[f"current_total_{arg_dict[DENOMINATOR]}"]

        z_alpha = st.norm.ppf(
            1
            - df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE].values[0] / (2 if df[PREFERENCE_TEST].values[0] == TWO_SIDED else 1)
        )
        z_power = st.norm.ppf(df[ADJUSTED_POWER].values[0])

        nim = df[NIM].values[0]
        if isinstance(nim, float):
            non_inferiority = not isnan(nim)
        elif nim is None:
            non_inferiority = nim is not None

        df[POWERED_EFFECT] = confidence_computers[df[arg_dict[METHOD]].values[0]].powered_effect(
            df=df.assign(kappa=kappa)
            .assign(current_number_of_units=df[f"current_total_{arg_dict[DENOMINATOR]}"])
            .assign(proportion_of_total=proportion_of_total),
            z_alpha=z_alpha,
            z_power=z_power,
            binary=binary,
            non_inferiority=non_inferiority,
            avg_column=POINT_ESTIMATE + SFX1,
            var_column=VARIANCE + SFX1,
        )

        if ALTERNATIVE_HYPOTHESIS in df and NULL_HYPOTHESIS in df and (df[ALTERNATIVE_HYPOTHESIS].notna()).all():
            df[REQUIRED_SAMPLE_SIZE] = confidence_computers[df[arg_dict[METHOD]].values[0]].required_sample_size(
                proportion_of_total=1,
                z_alpha=z_alpha,
                z_power=z_power,
                binary=binary,
                non_inferiority=non_inferiority,
                hypothetical_effect=df[ALTERNATIVE_HYPOTHESIS] - df[NULL_HYPOTHESIS],
                control_avg=df[POINT_ESTIMATE + SFX1],
                control_var=df[VARIANCE + SFX1],
                kappa=kappa,
            )
            df[REQUIRED_SAMPLE_SIZE_METRIC] = confidence_computers[
                df[arg_dict[METHOD]].values[0]
            ].required_sample_size(
                proportion_of_total=proportion_of_total,
                z_alpha=z_alpha,
                z_power=z_power,
                binary=binary,
                non_inferiority=non_inferiority,
                hypothetical_effect=df[ALTERNATIVE_HYPOTHESIS] - df[NULL_HYPOTHESIS],
                control_avg=df[POINT_ESTIMATE + SFX1],
                control_var=df[VARIANCE + SFX1],
                kappa=kappa,
            )
        else:
            df[REQUIRED_SAMPLE_SIZE] = None
            df[REQUIRED_SAMPLE_SIZE_METRIC] = None

        return df


def _compute_sample_sizes_and_ci_widths(df: DataFrame, arg_dict: Dict) -> DataFrame:
    return df.pipe(_sample_size_from_summary_df, arg_dict=arg_dict).pipe(_ci_width, arg_dict=arg_dict)


def _sample_size_from_summary_df(df: DataFrame, arg_dict: Dict) -> DataFrame:
    if df[arg_dict[METHOD]].values[0] != ZTEST in df:
        raise ValueError("Sample size calculation only supported for ZTest.")
    elif df[arg_dict[METHOD]].values[0] != ZTEST or (df[ADJUSTED_POWER].isna()).any():
        df[REQUIRED_SAMPLE_SIZE_METRIC] = None
    else:
        all_weights = arg_dict[TREATMENT_WEIGHTS]
        control_weight, treatment_weights = all_weights[0], all_weights[1:]

        binary = df[arg_dict[IS_BINARY]].values[0]
        z_alpha = st.norm.ppf(
            1
            - df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE].values[0] / (2 if df[PREFERENCE_TEST].values[0] == TWO_SIDED else 1)
        )
        z_power = st.norm.ppf(df[ADJUSTED_POWER].values[0])
        non_inferiority = is_non_inferiority(df[NIM].values[0])

        max_sample_size = 0
        for treatment_weight in treatment_weights:
            kappa = control_weight / treatment_weight
            proportion_of_total = (control_weight + treatment_weight) / sum(all_weights)

            if ALTERNATIVE_HYPOTHESIS in df and NULL_HYPOTHESIS in df and (df[ALTERNATIVE_HYPOTHESIS].notna()).all():
                this_sample_size = confidence_computers[df[arg_dict[METHOD]].values[0]].required_sample_size(
                    proportion_of_total=proportion_of_total,
                    z_alpha=z_alpha,
                    z_power=z_power,
                    binary=binary,
                    non_inferiority=non_inferiority,
                    hypothetical_effect=df[ALTERNATIVE_HYPOTHESIS] - df[NULL_HYPOTHESIS],
                    control_avg=df[POINT_ESTIMATE],
                    control_var=df[VARIANCE],
                    kappa=kappa,
                )
                max_sample_size = max(this_sample_size.max(), max_sample_size)

        df[REQUIRED_SAMPLE_SIZE_METRIC] = None if max_sample_size == 0 else max_sample_size

    return df


def _compute_powered_effects(df: DataFrame, arg_dict: Dict) -> DataFrame:
    return df.pipe(_powered_effect_from_summary_df, arg_dict=arg_dict)


def _powered_effect_from_summary_df(df: DataFrame, arg_dict: Dict) -> DataFrame:
    if df[arg_dict[METHOD]].values[0] != ZTEST in df:
        raise ValueError("Powered effect calculation only supported for ZTest.")
    elif df[arg_dict[METHOD]].values[0] != ZTEST or (df[ADJUSTED_POWER].isna()).any():
        df[REQUIRED_SAMPLE_SIZE_METRIC] = None
    else:
        all_weights = arg_dict[TREATMENT_WEIGHTS]
        control_weight, treatment_weights = all_weights[0], all_weights[1:]

        current_number_of_units = arg_dict[FINAL_EXPECTED_SAMPLE_SIZE]

        binary = df[arg_dict[IS_BINARY]].values[0]
        z_alpha = st.norm.ppf(
            1
            - df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE].values[0] / (2 if df[PREFERENCE_TEST].values[0] == TWO_SIDED else 1)
        )
        z_power = st.norm.ppf(df[ADJUSTED_POWER].values[0])
        non_inferiority = is_non_inferiority(df[NIM].values[0])

        max_powered_effect = 0
        for treatment_weight in treatment_weights:
            kappa = control_weight / treatment_weight
            proportion_of_total = (control_weight + treatment_weight) / sum(all_weights)

            this_powered_effect = df[POWERED_EFFECT] = confidence_computers[
                df[arg_dict[METHOD]].values[0]
            ].powered_effect(
                df=df.assign(kappa=kappa)
                .assign(current_number_of_units=current_number_of_units)
                .assign(proportion_of_total=proportion_of_total),
                z_alpha=z_alpha,
                z_power=z_power,
                binary=binary,
                non_inferiority=non_inferiority,
                avg_column=POINT_ESTIMATE,
                var_column=VARIANCE,
            )

            max_powered_effect = max(this_powered_effect.max(), max_powered_effect)

        df[POWERED_EFFECT] = None if max_powered_effect == 0 else max_powered_effect

    return df


def _ci_width(df: DataFrame, arg_dict: Dict) -> DataFrame:
    expected_sample_size = (
        None if arg_dict[FINAL_EXPECTED_SAMPLE_SIZE] is None else df[arg_dict[FINAL_EXPECTED_SAMPLE_SIZE]].values[0]
    )
    if expected_sample_size is None or np.isnan(expected_sample_size):
        return df.assign(**{CI_WIDTH: None})

    all_weights = arg_dict[TREATMENT_WEIGHTS]
    control_weight, treatment_weights = all_weights[0], all_weights[1:]
    sum_of_weights = sum(all_weights)

    control_count = int((control_weight / sum_of_weights) * expected_sample_size)
    if control_count == 0:
        return df.assign(**{CI_WIDTH: float("inf")})

    else:
        binary = df[arg_dict[IS_BINARY]].values[0]
        z_alpha = st.norm.ppf(
            1
            - df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE].values[0] / (2 if df[PREFERENCE_TEST].values[0] == TWO_SIDED else 1)
        )

        non_inferiority = is_non_inferiority(df[NIM].values[0])
        max_ci_width = 0
        for treatment_weight in treatment_weights:
            treatment_count = int((treatment_weight / sum_of_weights) * expected_sample_size)
            if treatment_count == 0:
                return df.assign(**{CI_WIDTH: float("inf")})
            else:
                comparison_ci_width = confidence_computers[df[arg_dict[METHOD]].values[0]].ci_width(
                    z_alpha=z_alpha,
                    binary=binary,
                    non_inferiority=non_inferiority,
                    hypothetical_effect=df[ALTERNATIVE_HYPOTHESIS] - df[NULL_HYPOTHESIS],
                    control_avg=df[POINT_ESTIMATE],
                    control_var=df[VARIANCE],
                    control_count=control_count,
                    treatment_count=treatment_count,
                )

            max_ci_width = max(comparison_ci_width.max(), max_ci_width)

        df[CI_WIDTH] = None if max_ci_width == 0 else max_ci_width

    return df


def _optimal_kappa(row: Series, is_binary_column) -> float:
    def _binary_variance(p: float) -> float:
        return p * (1 - p)

    if row[is_binary_column]:
        if is_non_inferiority(row[NIM]):
            return 1.0
        else:
            if row[POINT_ESTIMATE] == 0.0:
                # variance will be 0 as well in this case. This if-branch is important to avoid divide by zero problems
                return 1.0
            else:
                hypothetical_effect = row[ALTERNATIVE_HYPOTHESIS] - row[NULL_HYPOTHESIS]
                return np.sqrt(
                    _binary_variance(row[POINT_ESTIMATE]) / _binary_variance(row[POINT_ESTIMATE] + hypothetical_effect)
                )
    else:
        return 1.0


def _optimal_weights(kappa: float, number_of_groups) -> Iterable:
    treatment_weight = 1 / (kappa + number_of_groups - 1)
    control_weight = kappa * treatment_weight
    return [control_weight] + [treatment_weight for _ in range(number_of_groups - 1)]


def _find_optimal_group_weights_across_rows(
    df: DataFrame, group_count: int, group_columns: Iterable, arg_dict: Dict
) -> (List[float], int):
    min_kappa = min(df[OPTIMAL_KAPPA])
    max_kappa = max(df[OPTIMAL_KAPPA])

    if min_kappa == max_kappa:
        optimal_weights = df[OPTIMAL_WEIGHTS][0]
        optimal_sample_size = _calculate_optimal_sample_size_given_weights(
            df, optimal_weights, group_columns, arg_dict
        )
        return optimal_weights, optimal_sample_size

    in_between_kappas = np.linspace(min_kappa, max_kappa, 100)
    min_optimal_sample_size = float("inf")
    optimal_weights = []
    for kappa in in_between_kappas:
        weights = _optimal_weights(kappa, group_count)
        optimal_sample_size = _calculate_optimal_sample_size_given_weights(df, weights, group_columns, arg_dict)
        if optimal_sample_size is not None and optimal_sample_size < min_optimal_sample_size:
            min_optimal_sample_size = optimal_sample_size
            optimal_weights = weights
    min_optimal_sample_size = np.nan if min_optimal_sample_size == 0 else min_optimal_sample_size
    return optimal_weights, min_optimal_sample_size


def _calculate_optimal_sample_size_given_weights(
    df: DataFrame, optimal_weights: List[float], group_columns: Iterable, arg_dict: Dict
) -> int:
    arg_dict[TREATMENT_WEIGHTS] = optimal_weights
    sample_size_df = groupbyApplyParallel(
        df.groupby(group_columns, as_index=False, sort=False),
        lambda df: _sample_size_from_summary_df(df, arg_dict=arg_dict),
    )

    if sample_size_df[REQUIRED_SAMPLE_SIZE_METRIC].isna().all():
        return None
    optimal_sample_size = sample_size_df[REQUIRED_SAMPLE_SIZE_METRIC].max()

    return np.ceil(optimal_sample_size) if np.isfinite(optimal_sample_size) else optimal_sample_size
