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
from typing import Union, Iterable

from pandas import DataFrame

from spotify_confidence.chartgrid import ChartGrid
from ..constants import NIM_TYPE


class ConfidenceGrapherABC(ABC):
    @abstractmethod
    def __init__(
        self,
        data_frame: DataFrame,
        numerator_column: str,
        denominator_column: str,
        categorical_group_columns: str,
        ordinal_group_column: str,
    ):
        pass

    @abstractmethod
    def plot_summary(self, summary_df: DataFrame, groupby: Union[str, Iterable]) -> ChartGrid:
        """Plot for each group in the data_frame:

        if ordinal level exists:
            line graph with area to represent confidence interval
        if categorical levels:
            Interval plots of confidence intervals by group

        Args:
            summary_df (DataFrame): A data frame produced by a
            ConfidenceComputer's summary method

        Returns:
            ChartGrid object.
        """
        pass

    @abstractmethod
    def plot_difference(
        self,
        difference_df: DataFrame,
        absolute: bool,
        groupby: Union[str, Iterable],
        nims: NIM_TYPE,
        use_adjusted_intervals: bool,
    ) -> ChartGrid:
        """Plot representing the difference between group 1 and 2 with
        confidence intervals.

        Args:
            difference_df (DataFrame): A dataframe produced by a
            ConfidenceComputer's difference method

        Returns:
            Chartify Chart object.
            :param groupby:
        """

    @abstractmethod
    def plot_differences(
        self,
        difference_df: DataFrame,
        absolute: bool,
        groupby: Union[str, Iterable],
        nims: NIM_TYPE,
        use_adjusted_intervals: bool,
    ) -> ChartGrid:
        """Plot representing the difference between group 1 and 2 with
        confidence intervals.

        Args:
            difference_df (DataFrame): A dataframe produced by a
            ConfidenceComputer's difference method

        Returns:
            Chartify Chart object.
            :param groupby:
        """

    @abstractmethod
    def plot_multiple_difference(
        self,
        difference_df: DataFrame,
        absolute: bool,
        groupby: Union[str, Iterable],
        level_as_reference: bool,
        nims: NIM_TYPE,
        use_adjusted_intervals: bool,
    ) -> ChartGrid:
        """Compare level to all other groups or, if level_as_reference = True,
        all other groups to level.

        Args:
            difference_df (DataFrame): A dataframe produced by a
            ConfidenceComputer's multiple_difference method

        Returns:
            ChartGrid object.
        """
