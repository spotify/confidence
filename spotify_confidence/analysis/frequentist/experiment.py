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

from typing import Union, Iterable, Tuple, Dict, List

from pandas import DataFrame

from spotify_confidence.analysis.frequentist.confidence_computers.generic_computer import GenericComputer
from .chartify_grapher import ChartifyGrapher
from ..abstract_base_classes.confidence_abc import ConfidenceABC
from ..abstract_base_classes.confidence_computer_abc import ConfidenceComputerABC
from ..abstract_base_classes.confidence_grapher_abc import ConfidenceGrapherABC
from ..confidence_utils import (
    validate_categorical_columns,
    listify,
    get_all_categorical_group_columns,
    get_all_group_columns,
)
from ..constants import BONFERRONI, NIM_TYPE, METHODS
from ..frequentist.sample_ratio_test import sample_ratio_test
from ...chartgrid import ChartGrid


class Experiment(ConfidenceABC):
    def __init__(
        self,
        data_frame: DataFrame,
        numerator_column: str,
        numerator_sum_squares_column: Union[str, None],
        denominator_column: str,
        categorical_group_columns: Union[str, Iterable],
        ordinal_group_column: Union[str, None] = None,
        interval_size: float = 0.95,
        correction_method: str = BONFERRONI,
        confidence_computer: ConfidenceComputerABC = None,
        confidence_grapher: ConfidenceGrapherABC = None,
        method_column: str = None,
        bootstrap_samples_column: str = None,
        metric_column=None,
        treatment_column=None,
        power: float = 0.8,
        feature_column: str = None,
        feature_sum_squares_column: str = None,
        feature_cross_sum_column: str = None,
    ):

        validate_categorical_columns(categorical_group_columns)
        self._df = data_frame
        self._numerator = numerator_column
        self._numerator_sumsq = numerator_sum_squares_column
        self._denominator = denominator_column
        self._categorical_group_columns = get_all_categorical_group_columns(
            categorical_group_columns, metric_column, treatment_column
        )
        self._ordinal_group_column = ordinal_group_column
        self._metric_column = metric_column
        self._treatment_column = treatment_column
        self._all_group_columns = get_all_group_columns(self._categorical_group_columns, self._ordinal_group_column)
        if method_column is None:
            raise ValueError("method column cannot be None")
        if not all(self._df[method_column].map(lambda m: m in METHODS)):
            raise ValueError(f"Values of method column must be in {METHODS}")

        if confidence_computer is not None:
            self._confidence_computer = confidence_computer
        else:
            self._confidence_computer = GenericComputer(
                data_frame=data_frame,
                numerator_column=numerator_column,
                numerator_sum_squares_column=numerator_sum_squares_column,
                denominator_column=denominator_column,
                categorical_group_columns=listify(categorical_group_columns),
                ordinal_group_column=ordinal_group_column,
                interval_size=interval_size,
                correction_method=correction_method.lower(),
                method_column=method_column,
                bootstrap_samples_column=bootstrap_samples_column,
                metric_column=metric_column,
                treatment_column=treatment_column,
                power=power,
                point_estimate_column=None,
                var_column=None,
                is_binary_column=None,
                feature_column=feature_column,
                feature_sum_squares_column=feature_sum_squares_column,
                feature_cross_sum_column=feature_cross_sum_column,
            )

        self._confidence_grapher = (
            confidence_grapher
            if confidence_grapher is not None
            else ChartifyGrapher(
                data_frame=self._df,
                numerator_column=self._numerator,
                denominator_column=self._denominator,
                categorical_group_columns=self._categorical_group_columns,
                ordinal_group_column=self._ordinal_group_column,
            )
        )

    def summary(self, verbose: bool = False) -> DataFrame:
        return self._confidence_computer.compute_summary(verbose)

    def difference(
        self,
        level_1: Union[str, Tuple],
        level_2: Union[str, Tuple],
        absolute: bool = True,
        groupby: Union[str, Iterable] = None,
        non_inferiority_margins: NIM_TYPE = None,
        final_expected_sample_size_column: str = None,
        verbose: bool = False,
        minimum_detectable_effects_column: str = None,
    ) -> DataFrame:
        self._validate_sequential(final_expected_sample_size_column, groupby)

        return self._confidence_computer.compute_difference(
            level_1,
            level_2,
            absolute,
            groupby,
            non_inferiority_margins,
            final_expected_sample_size_column,
            verbose,
            minimum_detectable_effects_column,
        )

    def differences(
        self,
        levels: Union[Tuple, List[Tuple]],
        absolute: bool = True,
        groupby: Union[str, Iterable] = None,
        non_inferiority_margins: NIM_TYPE = None,
        final_expected_sample_size_column: str = None,
        verbose: bool = False,
        minimum_detectable_effects_column: str = None,
    ) -> DataFrame:
        self._validate_sequential(final_expected_sample_size_column, groupby)
        return self._confidence_computer.compute_differences(
            levels,
            absolute,
            groupby,
            non_inferiority_margins,
            final_expected_sample_size_column,
            verbose,
            minimum_detectable_effects_column,
        )

    def multiple_difference(
        self,
        level: Union[str, Tuple],
        absolute: bool = True,
        groupby: Union[str, Iterable] = None,
        level_as_reference: bool = None,
        non_inferiority_margins: NIM_TYPE = None,
        final_expected_sample_size_column: str = None,
        verbose: bool = False,
        minimum_detectable_effects_column: str = None,
    ) -> DataFrame:
        self._validate_sequential(final_expected_sample_size_column, groupby)

        return self._confidence_computer.compute_multiple_difference(
            level,
            absolute,
            groupby,
            level_as_reference,
            non_inferiority_margins,
            final_expected_sample_size_column,
            verbose,
            minimum_detectable_effects_column,
        )

    def summary_plot(self, groupby: Union[str, Iterable] = None) -> ChartGrid:
        summary_df = self.summary()
        graph = self._confidence_grapher.plot_summary(summary_df, groupby)
        return graph

    def difference_plot(
        self,
        level_1: Union[str, Tuple],
        level_2: Union[str, Tuple],
        absolute: bool = True,
        groupby: Union[str, Iterable] = None,
        non_inferiority_margins: NIM_TYPE = None,
        use_adjusted_intervals: bool = False,
        final_expected_sample_size_column: str = None,
    ) -> ChartGrid:
        difference_df = self.difference(
            level_1=level_1,
            level_2=level_2,
            absolute=absolute,
            groupby=groupby,
            non_inferiority_margins=non_inferiority_margins,
            final_expected_sample_size_column=final_expected_sample_size_column,
        )
        chartgrid = self._confidence_grapher.plot_difference(
            difference_df, absolute, groupby, non_inferiority_margins, use_adjusted_intervals
        )
        return chartgrid

    def differences_plot(
        self,
        levels: List[Tuple],
        absolute: bool = True,
        groupby: Union[str, Iterable] = None,
        non_inferiority_margins: NIM_TYPE = None,
        use_adjusted_intervals: bool = False,
        final_expected_sample_size_column: str = None,
    ) -> ChartGrid:
        difference_df = self.differences(
            levels, absolute, groupby, non_inferiority_margins, final_expected_sample_size_column
        )
        chartgrid = self._confidence_grapher.plot_differences(
            difference_df, absolute, groupby, non_inferiority_margins, use_adjusted_intervals
        )
        return chartgrid

    def multiple_difference_plot(
        self,
        level: Union[str, Tuple],
        absolute: bool = True,
        groupby: Union[str, Iterable] = None,
        level_as_reference: bool = False,
        non_inferiority_margins: NIM_TYPE = None,
        use_adjusted_intervals: bool = False,
        final_expected_sample_size_column: str = None,
    ) -> ChartGrid:
        difference_df = self.multiple_difference(
            level,
            absolute,
            groupby,
            level_as_reference,
            non_inferiority_margins,
            None,
            final_expected_sample_size_column,
        )
        chartgrid = self._confidence_grapher.plot_multiple_difference(
            difference_df, absolute, groupby, level_as_reference, non_inferiority_margins, use_adjusted_intervals
        )
        return chartgrid

    def sample_ratio_test(self, expected_proportions: Dict) -> Tuple[float, DataFrame]:
        return sample_ratio_test(
            self._df,
            all_group_columns=self._all_group_columns,
            denominator=self._denominator,
            expected_proportions=expected_proportions,
        )

    def achieved_power(self, level_1, level_2, mde, alpha, groupby=None) -> DataFrame:
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
        return self._confidence_computer.achieved_power(level_1, level_2, mde, alpha, groupby)

    def _validate_sequential(self, final_expected_sample_size: float, groupby: Union[str, Iterable]):
        if final_expected_sample_size is not None:
            if self._ordinal_group_column not in listify(groupby):
                raise ValueError(
                    f"{self._ordinal_group_column} must be in groupby argument to use "
                    f"sequential testing with final_expected_sample_size"
                )
