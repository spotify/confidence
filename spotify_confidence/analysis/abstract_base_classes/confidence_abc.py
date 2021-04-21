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

from abc import ABC, abstractmethod
from typing import (Union, Iterable, Tuple)

from pandas import DataFrame

from spotify_confidence.chartgrid import ChartGrid
from .confidence_computer_abc import ConfidenceComputerABC
from .confidence_grapher_abc import ConfidenceGrapherABC
from ..constants import NIM_TYPE


class ConfidenceABC(ABC):

    @property
    def _confidence_computer(self) -> ConfidenceComputerABC:
        return self._computer

    @_confidence_computer.setter
    def _confidence_computer(self, computer: ConfidenceComputerABC):
        self._computer = computer

    @property
    def _confidence_grapher(self) -> ConfidenceGrapherABC:
        return self._grapher

    @_confidence_grapher.setter
    def _confidence_grapher(self, grapher: ConfidenceGrapherABC):
        self._grapher = grapher

    @abstractmethod
    def __init__(self,
                 data_frame: DataFrame,
                 numerator_column: str,
                 numerator_sum_squares_column: Union[str, None],
                 denominator_column: str,
                 categorical_group_columns: Union[str, Iterable],
                 ordinal_group_column: Union[str, None],
                 interval_size: float,
                 correction_method: str):
        pass

    @abstractmethod
    def summary(self) -> DataFrame:
        """
        Returns:
            Dataframe containing summary statistics
        """
        pass

    @abstractmethod
    def difference(self,
                   level_1: Union[str, Tuple],
                   level_2: Union[str, Tuple],
                   absolute: bool,
                   groupby: Union[str, Iterable],
                   non_inferiority_margins: NIM_TYPE,
                   final_expected_sample_size: float
                   ) -> DataFrame:
        """Args:
            groupby (str): Name of column.
                If specified, will plot a separate chart for each level of the
                grouping.
            non_inferiority_margins (Union[Tuple[float, str],
                    Dict[str, Tuple[float, str]]]):
                Pass tuple(nim, preferred direction) to use the same NIM for all
                comparisons, e.g. (0.01, 'increase'), which means that we want
                level_2 to be grater than the average of level_1 times (1-0.01),
                or (0.05, 'decrease') which means that we want
                level_2 to be smaller than the average
                of level_1 times (1+0.01).
                Pass dictionary {{group:tuple(nim, preferred direction}} to use
                different non-inferiority margins for different values of
                groupby column.
                To performe a one-sided test without nim, use
                (None, preffered direction).
            final_expected_sample_size (float): Expected number of observations
                    at end of experiment.
                Use in combination with ordinal groupby to perform a
                sequential test. See https://cran.r-project.org/web/packages/ldbounds/index.html for details.

        Returns:
            Dataframe containing the difference in means between
            group 1 and 2, p-values and confidence intervals for each value
            in the groupby column
        """
        pass

    @abstractmethod
    def multiple_difference(self,
                            level: Union[str, Tuple],
                            absolute: bool,
                            groupby: Union[str, Iterable],
                            level_as_reference: bool,
                            non_inferiority_margins: NIM_TYPE,
                            final_expected_sample_size: float
                            ) -> DataFrame:
        """Args:
            groupby (str): Name of column.
                If specified, will plot a separate chart for each level of the
                grouping.
            level_as_reference (bool):
                If false (default), compare level to all other
                groups. If true, compare all other groups to level.
            non_inferiority_margins (Union[Tuple[float, str],
                    Dict[str, Tuple[float, str]]]):
                Pass tuple(nim, preferred direction) to use the same NIM for all
                comparisons, e.g. (0.01, 'increase'), which means that we want
                level_2 to be grater than the average of level_1 times (1-0.01),
                or (0.05, 'decrease') which means that we want
                level_2 to be smaller than the average
                of level_1 times (1+0.01).
                Pass dictionary {{group:tuple(nim, preferred direction}} to use
                different non-inferiority margins for different values of
                groupby column.
                To performe a one-sided test without nim, use
                (None, preffered direction).
            final_expected_sample_size (float): Expected number of observations
                    at end of experiment.
                Use in combination with ordinal groupby to perform a
                sequential test. See https://cran.r-project.org/web/packages/ldbounds/index.html for details.

        Returns:
            Dataframe containing the difference in means between
            group 1 and 2, p-values and confidence intervals for each value
            in the groupby column
        """
        pass

    @abstractmethod
    def summary_plot(self,
                     groupby: Union[str, Iterable]) -> ChartGrid:
        """Plot for each group in the data_frame:

        if ordinal level exists:
            line graph with area to represent confidence interval
        if categorical levels:
            Interval plots of confidence intervals by group

        Args:
            groupby (str): Name of column.
                If specified, will plot a separate chart for each level of the
                grouping.

        Returns:
            ChartGrid object and a DataFrame with numerical results.
        """
        pass

    @abstractmethod
    def difference_plot(self,
                        level_1: Union[str, Tuple],
                        level_2: Union[str, Tuple],
                        absolute: bool,
                        groupby: Union[str, Iterable],
                        non_inferiority_margins: NIM_TYPE,
                        use_adjusted_intervals: bool,
                        final_expected_sample_size: float
                        ) -> ChartGrid:
        """Plot representing the difference between group 1 and 2.
        - Difference in means or proportions, depending
            on the response variable type.

        - Plot interval plot with confidence interval of the
            difference between groups

        Args:
            level_1 (str, tuple of str): Name of first level.
            level_2 (str, tuple of str): Name of second level.
            absolute (bool): If True then return the absolute
                difference (level2 - level1)
                otherwise return the relative difference (level2 / level1 - 1)
            groupby (str): Name of column, or list of columns.
                If specified, will return an interval for each level
                of the grouped dimension, or a confidence band if the
                grouped dimension is ordinal
            non_inferiority_margins (Union[Tuple[float, str],
                    Dict[str, Tuple[float, str]]]):
                Pass tuple(nim, preferred direction) to use the same NIM for all
                comparisons, e.g. (0.01, 'increase'), which means that we want
                level_2 to be grater than the average of level_1 times (1-0.01),
                or (0.05, 'decrease') which means that we want
                level_2 to be smaller than the average
                of level_1 times (1+0.01).
                Pass dictionary {{group:tuple(nim, preferred direction}} to use
                different non-inferiority margins for different values of
                groupby column.
                To performe a one-sided test without nim, use
                (None, preffered direction).
            use_adjusted_intervals (bool):
                If true, use e.g. bon-ferroni corrected
                (or other method provided) confidence intervals
            final_expected_sample_size (float): Expected number of observations
                    at end of experiment.
                Use in combination with ordinal groupby to perform a
                sequential test. See https://cran.r-project.org/web/packages/ldbounds/index.html for details.

        Returns:
            Chartify Chart object and a DataFrame with numerical results.
        """

    @abstractmethod
    def multiple_difference_plot(self,
                                 level: Union[str, Tuple],
                                 absolute: bool,
                                 groupby: Union[str, Iterable],
                                 level_as_reference: bool,
                                 non_inferiority_margins: NIM_TYPE,
                                 use_adjusted_intervals: bool,
                                 final_expected_sample_size: float
                                 ) -> ChartGrid:
        """Compare level to all other groups or, if level_as_reference = True,
        all other groups to level.

        Args:
            level (str, tuple of str): Name of level.
            absolute (bool): If True then return the absolute
                difference (level2 - level1)
                otherwise return the relative difference (level2 / level1 - 1)
            groupby (str): Name of column, or list of columns.
                If specified, will return an interval for each level
                of the grouped dimension, or a confidence band if the
                grouped dimension is ordinal
            level_as_reference: If false (default), compare level to all other
             groups. If true, compare all other groups to level.
            non_inferiority_margins (Union[Tuple[float, str],
                    Dict[str, Tuple[float, str]]]):
                Pass tuple(nim, preferred direction) to use the same NIM for all
                comparisons, e.g. (0.01, 'increase'), which means that we want
                level_2 to be grater than the average of level_1 times (1-0.01),
                or (0.05, 'decrease') which means that we want
                level_2 to be smaller than the average
                of level_1 times (1+0.01).
                Pass dictionary {{group:tuple(nim, preferred direction}} to use
                different non-inferiority margins for different values of
                groupby column.
                To performe a one-sided test without nim, use
                (None, preffered direction).
            use_adjusted_intervals (bool):
                If true, use e.g. bon-ferroni corrected
                (or other method provided) confidence intervals
            final_expected_sample_size (float): Expected number of observations
                    at end of experiment.
                Use in combination with ordinal groupby to perform a
                sequential test. See https://cran.r-project.org/web/packages/ldbounds/index.html for details.

        Returns:
            ChartGrid object and a DataFrame with numerical results.
        """
