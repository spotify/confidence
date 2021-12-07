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

from pandas import DataFrame, Series
from statsmodels.stats.multitest import multipletests
from numpy import isnan

import spotify_confidence.analysis.frequentist.confidence_computers.bootstrap_computer as bootstrap_computer
import spotify_confidence.analysis.frequentist.confidence_computers.chi_squared_computer as chi_squared_computer
import spotify_confidence.analysis.frequentist.confidence_computers.t_test_computer as t_test_computer
import spotify_confidence.analysis.frequentist.confidence_computers.z_test_computer as z_test_computers
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
    NIM_INPUT_COLUMN_NAME,
    PREFERRED_DIRECTION_INPUT_NAME,
    INCREASE_PREFFERED,
    DECREASE_PREFFERED,
)

confidence_computers = {
    CHI2: chi_squared_computer,
    TTEST: t_test_computer,
    ZTEST: z_test_computers,
    BOOTSTRAP: bootstrap_computer,
}


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
            }
            groupby = [col for col in [self._method_column, self._metric_column] if col is not None]
            self._sufficient = (
                self._df.groupby(groupby)
                .apply(
                    lambda df: df.assign(
                        **{
                            POINT_ESTIMATE: lambda df: confidence_computers[
                                df[self._method_column].values[0]
                            ].point_estimate(df, arg_dict)
                        }
                    )
                    .assign(
                        **{
                            VARIANCE: lambda df: confidence_computers[df[self._method_column].values[0]].variance(
                                df, arg_dict
                            )
                        }
                    )
                    .pipe(
                        lambda df: confidence_computers[df[self._method_column].values[0]].add_point_estimate_ci(
                            df, arg_dict
                        )
                    )
                )
                .reset_index(drop=True)
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
            .pipe(add_nims_and_mdes, mde_column=mde_column)
            .pipe(join)
            .query(
                f"level_1 in {[l1 for l1, l2 in groups_to_compare]} and "
                + f"level_2 in {[l2 for l1, l2 in groups_to_compare]}"
                + "and level_1 != level_2"
            )
            .pipe(
                validate_and_rename_columns,
                [NIM, mde_column, PREFERENCE, final_expected_sample_size_column, self._method_column],
            )
            .pipe(
                drop_and_rename_columns,
                [NULL_HYPOTHESIS, ALTERNATIVE_HYPOTHESIS, f"current_total_{self._denominator}"],
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
            comparison_df, self._correction_method, ["level_1", "level_2"] + groups_except_ordinal
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
        }
        comparison_df = groupbyApplyParallel(
            comparison_df.assign(n_comparisons=n_comparisons).groupby(
                groups_except_ordinal + [self._method_column], as_index=False
            ),
            lambda df: _compute_comparisons(df, arg_dict=arg_dict),
        )
        return comparison_df

    def _add_adjusted_power(self, df: DataFrame) -> DataFrame:
        if self._correction_method in CORRECTION_METHODS_THAT_REQUIRE_METRIC_INFO:
            if self._metric_column is None or self._treatment_column is None:
                return df.assign(**{ADJUSTED_POWER: None})
            else:
                number_total_metrics = 1 if self._single_metric else df.groupby(self._metric_column).ngroups
                if self._single_metric:
                    if df[df[NIM].isnull()].shape[0] > 0:
                        number_success_metrics = 1
                    else:
                        number_success_metrics = 0
                else:
                    number_success_metrics = df[df[NIM].isnull()].groupby(self._metric_column).ngroups

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


def add_nim_input_columns_from_tuple_or_dict(df, nims: NIM_TYPE, mde_column: str) -> DataFrame:
    if type(nims) is tuple:
        return df.assign(**{NIM_INPUT_COLUMN_NAME: nims[0]}).assign(**{PREFERRED_DIRECTION_INPUT_NAME: nims[1]})
    elif type(nims) is dict:
        nim_values = {key: value[0] for key, value in nims.items()}
        nim_preferences = {key: value[1] for key, value in nims.items()}
        return df.assign(**{NIM_INPUT_COLUMN_NAME: lambda df: df.index.to_series().map(nim_values)}).assign(
            **{PREFERRED_DIRECTION_INPUT_NAME: lambda df: df.index.to_series().map(nim_preferences)}
        )
    elif nims is None:
        return df.assign(**{NIM_INPUT_COLUMN_NAME: None}).assign(
            **{
                PREFERRED_DIRECTION_INPUT_NAME: None
                if PREFERRED_DIRECTION_INPUT_NAME not in df or mde_column is None
                else df[PREFERRED_DIRECTION_INPUT_NAME]
            }
        )
    else:
        return df


def add_nims_and_mdes(df: DataFrame, mde_column: str) -> DataFrame:
    def _set_nims_and_mdes(grp: DataFrame) -> DataFrame:
        nim = grp[NIM_INPUT_COLUMN_NAME].astype(float)
        input_preference = grp[PREFERRED_DIRECTION_INPUT_NAME].values[0]
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
            [NIM_INPUT_COLUMN_NAME, PREFERRED_DIRECTION_INPUT_NAME] + listify(mde_column), dropna=False, as_index=False
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
        .pipe(_powered_effect_and_required_sample_size, arg_dict=arg_dict)
        .pipe(_adjust_if_absolute, absolute=arg_dict[ABSOLUTE])
        .assign(**{PREFERENCE: lambda df: df[PREFERENCE].map(PREFERENCE_DICT)})
    )


def _add_p_value_and_ci(df: DataFrame, arg_dict: Dict) -> DataFrame:
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
        )

    def _add_adjusted_p_and_is_significant(df: DataFrame, arg_dict: Dict) -> DataFrame:
        n_comparisons = df["n_comparisons"].values[0]
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
            df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE] = df[ALPHA] / n_comparisons
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
            df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE] = df[ADJUSTED_ALPHA]
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
            df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE] = df[ADJUSTED_ALPHA]
            df[ADJUSTED_P] = df[P_VALUE].map(lambda p: min(p * n_comparisons, 1))
            df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
        else:
            raise ValueError("Can't figure out which correction method to use :(")

        return df

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


def _powered_effect_and_required_sample_size(df: DataFrame, arg_dict: Dict) -> DataFrame:
    if df[arg_dict[METHOD]].unique() != ZTEST and arg_dict[MDE] in df:
        raise ValueError("Minimum detectable effects only supported for ZTest.")
    elif df[arg_dict[METHOD]].unique() != ZTEST or (df[ADJUSTED_POWER].isna()).any():
        df[POWERED_EFFECT] = None
        df[REQUIRED_SAMPLE_SIZE] = None
        return df
    else:
        return confidence_computers[df[arg_dict[METHOD]].values[0]].powered_effect_and_required_sample_size(
            df, arg_dict
        )


def _compute_sequential_adjusted_alpha(df: DataFrame, method_column: str, arg_dict: Dict) -> Series:
    if all(df[method_column] == "z-test"):
        return confidence_computers["z-test"].compute_sequential_adjusted_alpha(df, arg_dict)
    else:
        raise NotImplementedError("Sequential testing is only supported for z-tests")
