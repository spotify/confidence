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

import numpy as np
from pandas import DataFrame, Series
from scipy import stats as st

from spotify_confidence.analysis.confidence_utils import (
    get_all_categorical_group_columns,
    get_all_group_columns,
    validate_data,
    remove_group_columns,
    groupbyApplyParallel,
    is_non_inferiority,
    reset_named_indices,
    de_list_if_length_one,
)
from spotify_confidence.analysis.constants import (
    INTERVAL_SIZE,
    POINT_ESTIMATE,
    FINAL_EXPECTED_SAMPLE_SIZE,
    MDE,
    CORRECTION_METHOD,
    VARIANCE,
    NUMBER_OF_COMPARISONS,
    TREATMENT_WEIGHTS,
    IS_BINARY,
    CI_LOWER,
    CI_UPPER,
    DIFFERENCE,
    SFX1,
    ADJUSTED_ALPHA_POWER_SAMPLE_SIZE,
    POWER,
    POWERED_EFFECT,
    ADJUSTED_POWER,
    ADJUSTED_LOWER,
    ADJUSTED_UPPER,
    REQUIRED_SAMPLE_SIZE_METRIC,
    OPTIMAL_KAPPA,
    OPTIMAL_WEIGHTS,
    CI_WIDTH,
    NULL_HYPOTHESIS,
    ALTERNATIVE_HYPOTHESIS,
    NIM,
    PREFERENCE_TEST,
    TWO_SIDED,
    CORRECTION_METHODS,
    ZTEST,
    ORIGINAL_POINT_ESTIMATE,
    ORIGINAL_VARIANCE,
)
from spotify_confidence.analysis.frequentist.confidence_computers import confidence_computers
from spotify_confidence.analysis.frequentist.multiple_comparison import (
    get_num_comparisons,
    set_alpha_and_adjust_preference,
    get_preference,
    add_adjusted_power,
)
from spotify_confidence.analysis.frequentist.nims_and_mdes import (
    add_nims_and_mdes,
)


class SampleSizeComputer:
    def __init__(
        self,
        data_frame: DataFrame,
        categorical_group_columns: Union[str, Iterable],
        interval_size: float,
        correction_method: str,
        metric_column: str,
        power: float,
        point_estimate_column: str,
        var_column: str,
        is_binary_column: str,
    ):
        self._df = data_frame.reset_index(drop=True)
        self._point_estimate_column = point_estimate_column
        self._var_column = var_column
        self._is_binary = is_binary_column
        if self._point_estimate_column is not None and self._var_column is not None and self._is_binary is not None:
            mean = self._df.query(f"{self._is_binary} == True")[self._point_estimate_column]
            var = self._df.query(f"{self._is_binary} == True")[self._var_column]
            if not np.allclose(var, (mean * (1 - mean)), equal_nan=True):
                raise ValueError(
                    f"{var_column} doesn't equal {point_estimate_column}*(1-{point_estimate_column}) "
                    f"for all binary rows. Please check your data."
                )

        self._categorical_group_columns = get_all_categorical_group_columns(
            categorical_group_columns, metric_column, None
        )
        self._segments = remove_group_columns(self._categorical_group_columns, metric_column)
        self._segments = remove_group_columns(self._segments, None)
        self._metric_column = metric_column
        self._interval_size = interval_size
        self._power = power

        if correction_method.lower() not in CORRECTION_METHODS:
            raise ValueError(f"Use one of the correction methods in {CORRECTION_METHODS}")
        self._correction_method = correction_method

        self._single_metric = False
        if self._metric_column is not None and data_frame.groupby(self._metric_column, sort=False).ngroups == 1:
            self._single_metric = True

        self._all_group_columns = get_all_group_columns(self._categorical_group_columns, None)

        columns_that_must_exist = [self._point_estimate_column, self._var_column]
        validate_data(self._df, columns_that_must_exist, self._all_group_columns, None)

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
            groupby = [self._metric_column]
            self._sufficient = (
                self._df.groupby(groupby, sort=False, group_keys=True)
                .apply(
                    lambda df: df.assign(**{POINT_ESTIMATE: lambda df: df[self._point_estimate_column]})
                    .assign(**{ORIGINAL_POINT_ESTIMATE: lambda df: df[self._point_estimate_column]})
                    .assign(**{VARIANCE: lambda df: df[self._var_column]})
                    .assign(**{ORIGINAL_VARIANCE: lambda df: df[self._var_column]})
                )
                .pipe(reset_named_indices)
            )
        return self._sufficient

    def compute_sample_size(
        self,
        treatment_weights: Iterable,
        mde_column: str,
        nim_column: str,
        preferred_direction_column: str,
        final_expected_sample_size_column: str,
    ) -> DataFrame:
        kwargs, group_columns, sample_size_df = self._initialise_sample_size_and_power_computation(
            final_expected_sample_size_column, mde_column, nim_column, preferred_direction_column, treatment_weights
        )
        sample_size_df = groupbyApplyParallel(
            sample_size_df.pipe(set_alpha_and_adjust_preference, **kwargs).groupby(
                de_list_if_length_one(group_columns),
                as_index=False,
                sort=False,
            ),
            lambda df: _compute_sample_sizes_and_ci_widths(df, **kwargs),
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
        kwargs, group_columns, powered_effect_df = self._initialise_sample_size_and_power_computation(
            sample_size, mde_column, nim_column, preferred_direction_column, treatment_weights
        )
        powered_effect_df = groupbyApplyParallel(
            powered_effect_df.pipe(set_alpha_and_adjust_preference, **kwargs).groupby(
                de_list_if_length_one(group_columns),
                as_index=False,
                sort=False,
            ),
            lambda df: _compute_powered_effects(df, **kwargs),
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
            .assign(**{PREFERENCE_TEST: lambda df: get_preference(df, self._correction_method)})
            .assign(**{POWER: self._power})
            .pipe(
                add_adjusted_power,
                correction_method=self._correction_method,
                metric_column=self._metric_column,
                single_metric=self._single_metric,
            )
        )
        group_columns = [column for column in sample_size_df.index.names if column is not None]
        n_comparisons = get_num_comparisons(
            sample_size_df,
            self._correction_method,
            number_of_level_comparisons=len(treatment_weights) - 1,
            groupby=group_columns,
            metric_column=self._metric_column,
            treatment_column=None,
            single_metric=self._single_metric,
            segments=self._segments,
        )
        kwargs = {
            MDE: mde_column,
            NUMBER_OF_COMPARISONS: n_comparisons,
            TREATMENT_WEIGHTS: treatment_weights,
            INTERVAL_SIZE: self._interval_size,
            CORRECTION_METHOD: self._correction_method,
            IS_BINARY: self._is_binary,
            FINAL_EXPECTED_SAMPLE_SIZE: final_expected_sample_size_column,
        }
        return kwargs, group_columns, sample_size_df

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

        group_columns = [column for column in sample_size_df.index.names if column is not None] + [self._metric_column]
        kwargs = {
            IS_BINARY: self._is_binary,
        }
        return _find_optimal_group_weights_across_rows(sample_size_df, number_of_groups, group_columns, **kwargs)


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


def _compute_sample_sizes_and_ci_widths(df: DataFrame, **kwargs: Dict) -> DataFrame:
    return df.pipe(_sample_size_from_summary_df, **kwargs).pipe(_ci_width, **kwargs)


def _sample_size_from_summary_df(df: DataFrame, **kwargs: Dict) -> DataFrame:
    if (df[ADJUSTED_POWER].isna()).any():
        df[REQUIRED_SAMPLE_SIZE_METRIC] = None
    else:
        all_weights = kwargs[TREATMENT_WEIGHTS]
        control_weight, treatment_weights = all_weights[0], all_weights[1:]

        binary = df[kwargs[IS_BINARY]].values[0]
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
                this_sample_size = confidence_computers[ZTEST].required_sample_size(
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


def _compute_powered_effects(df: DataFrame, **kwargs: Dict) -> DataFrame:
    return df.pipe(_powered_effect_from_summary_df, **kwargs)


def _powered_effect_from_summary_df(df: DataFrame, **kwargs: Dict) -> DataFrame:
    if (df[ADJUSTED_POWER].isna()).any():
        df[REQUIRED_SAMPLE_SIZE_METRIC] = None
    else:
        all_weights = kwargs[TREATMENT_WEIGHTS]
        control_weight, treatment_weights = all_weights[0], all_weights[1:]

        current_number_of_units = kwargs[FINAL_EXPECTED_SAMPLE_SIZE]

        binary = df[kwargs[IS_BINARY]].values[0]
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

            this_powered_effect = df[POWERED_EFFECT] = confidence_computers[ZTEST].powered_effect(
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


def _ci_width(df: DataFrame, **kwargs: Dict) -> DataFrame:
    expected_sample_size = (
        None if kwargs[FINAL_EXPECTED_SAMPLE_SIZE] is None else df[kwargs[FINAL_EXPECTED_SAMPLE_SIZE]].values[0]
    )
    if expected_sample_size is None or np.isnan(expected_sample_size):
        return df.assign(**{CI_WIDTH: None})

    all_weights = kwargs[TREATMENT_WEIGHTS]
    control_weight, treatment_weights = all_weights[0], all_weights[1:]
    sum_of_weights = sum(all_weights)

    control_count = int((control_weight / sum_of_weights) * expected_sample_size)
    if control_count == 0:
        return df.assign(**{CI_WIDTH: float("inf")})

    else:
        binary = df[kwargs[IS_BINARY]].values[0]
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
                comparison_ci_width = confidence_computers[ZTEST].ci_width(
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
    df: DataFrame, group_count: int, group_columns: Iterable, **kwargs: Dict
) -> (List[float], int):
    min_kappa = min(df[OPTIMAL_KAPPA])
    max_kappa = max(df[OPTIMAL_KAPPA])

    if min_kappa == max_kappa:
        optimal_weights = df[OPTIMAL_WEIGHTS][0]
        optimal_sample_size = _calculate_optimal_sample_size_given_weights(
            df, optimal_weights, group_columns, **kwargs
        )
        return optimal_weights, optimal_sample_size

    in_between_kappas = np.linspace(min_kappa, max_kappa, 100)
    min_optimal_sample_size = float("inf")
    optimal_weights = []
    for kappa in in_between_kappas:
        weights = _optimal_weights(kappa, group_count)
        optimal_sample_size = _calculate_optimal_sample_size_given_weights(df, weights, group_columns, **kwargs)
        if optimal_sample_size is not None and optimal_sample_size < min_optimal_sample_size:
            min_optimal_sample_size = optimal_sample_size
            optimal_weights = weights
    min_optimal_sample_size = np.nan if min_optimal_sample_size == 0 else min_optimal_sample_size
    return optimal_weights, min_optimal_sample_size


def _calculate_optimal_sample_size_given_weights(
    df: DataFrame, optimal_weights: List[float], group_columns: Iterable, **kwargs: Dict
) -> int:
    kwargs[TREATMENT_WEIGHTS] = optimal_weights
    sample_size_df = groupbyApplyParallel(
        df.groupby(de_list_if_length_one(group_columns), as_index=False, sort=False),
        lambda df: _sample_size_from_summary_df(df, **kwargs),
    )

    if sample_size_df[REQUIRED_SAMPLE_SIZE_METRIC].isna().all():
        return None
    optimal_sample_size = sample_size_df[REQUIRED_SAMPLE_SIZE_METRIC].max()

    return np.ceil(optimal_sample_size) if np.isfinite(optimal_sample_size) else optimal_sample_size
