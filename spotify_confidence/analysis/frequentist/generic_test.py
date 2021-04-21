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

from typing import (Union, Iterable, Tuple, Dict)

from pandas import DataFrame

from .chartify_grapher import ChartifyGrapher
from ..abstract_base_classes.confidence_abc import ConfidenceABC
from ..abstract_base_classes.confidence_computer_abc import \
    ConfidenceComputerABC
from ..abstract_base_classes.confidence_grapher_abc import ConfidenceGrapherABC
from ..confidence_utils import (validate_categorical_columns, listify,
                                get_all_group_columns, validate_data)
from ..constants import BONFERRONI, NIM_TYPE
from ...chartgrid import ChartGrid
from ..frequentist.sample_ratio_test import sample_ratio_test


class GenericTest(ConfidenceABC):

    def __init__(self,
                 data_frame: DataFrame,
                 numerator_column: str,
                 numerator_sum_squares_column: Union[str, None],
                 denominator_column: str,
                 categorical_group_columns: Union[str, Iterable],
                 ordinal_group_column: Union[str, None] = None,
                 interval_size: float = 0.95,
                 correction_method: str = BONFERRONI,
                 confidence_computer: ConfidenceComputerABC = None,
                 confidence_grapher: ConfidenceGrapherABC = None):

        validate_categorical_columns(categorical_group_columns)
        self._df = data_frame
        self._numerator = numerator_column
        self._numerator_sumsq = numerator_sum_squares_column
        self._denominator = denominator_column
        self._categorical_group_columns = listify(categorical_group_columns)
        self._ordinal_group_column = ordinal_group_column

        self._all_group_columns = get_all_group_columns(
            self._categorical_group_columns,
            self._ordinal_group_column)
        validate_data(self._df,
                      self._numerator,
                      self._numerator_sumsq,
                      self._denominator,
                      self._all_group_columns,
                      self._ordinal_group_column)

        if confidence_computer is not None:
            self._confidence_computer = confidence_computer
        else:
            raise ValueError("Pass in a ConfidenceComputer or use one of the "
                             "subclasses e.g. ZTest.")

        self._confidence_grapher = confidence_grapher if confidence_grapher \
            is not None \
            else ChartifyGrapher(
                data_frame=self._df,
                numerator_column=self._numerator,
                denominator_column=self._denominator,
                categorical_group_columns=self._categorical_group_columns,
                ordinal_group_column=self._ordinal_group_column)

    def summary(self) -> DataFrame:
        return self._confidence_computer.compute_summary()

    def difference(self,
                   level_1: Union[str, Tuple],
                   level_2: Union[str, Tuple],
                   absolute: bool = True,
                   groupby: Union[str, Iterable] = None,
                   non_inferiority_margins: NIM_TYPE = None,
                   final_expected_sample_size: float = None
                   ) -> DataFrame:
        self._validate_sequential(final_expected_sample_size, groupby)
        return self._confidence_computer.compute_difference(
            level_1,
            level_2,
            absolute,
            groupby,
            non_inferiority_margins,
            final_expected_sample_size)

    def multiple_difference(self, level: Union[str, Tuple],
                            absolute: bool = True,
                            groupby: Union[str, Iterable] = None,
                            level_as_reference: bool = None,
                            non_inferiority_margins: NIM_TYPE = None,
                            final_expected_sample_size: float = None
                            ) -> DataFrame:
        self._validate_sequential(final_expected_sample_size, groupby)
        return self._confidence_computer.compute_multiple_difference(
            level,
            absolute,
            groupby,
            level_as_reference,
            non_inferiority_margins,
            final_expected_sample_size)

    def summary_plot(self,
                     groupby: Union[str, Iterable] = None) -> ChartGrid:
        summary_df = self.summary()
        graph = self._confidence_grapher.plot_summary(summary_df, groupby)
        return graph

    def difference_plot(self,
                        level_1: Union[str, Tuple],
                        level_2: Union[str, Tuple],
                        absolute: bool = True,
                        groupby: Union[str, Iterable] = None,
                        non_inferiority_margins: NIM_TYPE = None,
                        use_adjusted_intervals: bool = False,
                        final_expected_sample_size: float = None
                        ) -> ChartGrid:
        difference_df = self.difference(level_1,
                                        level_2,
                                        absolute,
                                        groupby,
                                        non_inferiority_margins,
                                        final_expected_sample_size)
        chartgrid = self._confidence_grapher.plot_difference(
            difference_df,
            absolute,
            groupby,
            non_inferiority_margins,
            use_adjusted_intervals)
        return chartgrid

    def multiple_difference_plot(self,
                                 level: Union[str, Tuple],
                                 absolute: bool = True,
                                 groupby: Union[str, Iterable] = None,
                                 level_as_reference: bool = False,
                                 non_inferiority_margins: NIM_TYPE = None,
                                 use_adjusted_intervals: bool = False,
                                 final_expected_sample_size: float = None
                                 ) -> ChartGrid:
        difference_df = self.multiple_difference(level,
                                                 absolute,
                                                 groupby,
                                                 level_as_reference,
                                                 non_inferiority_margins,
                                                 final_expected_sample_size)
        chartgrid = self._confidence_grapher.plot_multiple_difference(
            difference_df,
            absolute,
            groupby,
            level_as_reference,
            non_inferiority_margins,
            use_adjusted_intervals)
        return chartgrid

    def sample_ratio_test(self, expected_proportions: Dict
                          ) -> Tuple[float, DataFrame]:
        return sample_ratio_test(self._df,
                                 all_group_columns=self._all_group_columns,
                                 denominator=self._denominator,
                                 expected_proportions=expected_proportions)

    def achieved_power(self, level_1, level_2,
                       mde, alpha, groupby=None) -> DataFrame:
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
        return self._confidence_computer.achieved_power(level_1, level_2,
                                                        mde, alpha, groupby)

    def _validate_sequential(self,
                             final_expected_sample_size: float,
                             groupby: Union[str, Iterable]):
        if final_expected_sample_size is not None:
            if self._ordinal_group_column not in listify(groupby):
                raise ValueError(f"{self._ordinal_group_column} must be in groupby argument to use "
                                 f"sequential testing with final_expected_sample_size")
