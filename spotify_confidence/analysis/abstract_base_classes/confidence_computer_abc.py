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
from typing import Union, Iterable, List, Tuple

from pandas import DataFrame

from ..constants import NIM_TYPE


class ConfidenceComputerABC(ABC):
    @abstractmethod
    def compute_summary(self, verbose: bool) -> DataFrame:
        """Return Pandas DataFrame with summary statistics."""
        pass

    @abstractmethod
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
        """Return dataframe containing the difference in means between
        group 1 and 2, p-value and confidence interval
        """
        pass

    @abstractmethod
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
        """Return dataframe containing the difference in means between
        level and all other groups, with p-value and confidence interval
        """
        pass

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
        """Return dataframe containing the difference in means between
        level and all other groups, with p-value and confidence interval
        """
        pass

    def achieved_power(
        self,
        level_1: Union[str, Iterable],
        level_2: Union[str, Iterable],
        mde: float,
        alpha: float,
        groupby: Union[str, Iterable],
    ) -> DataFrame:
        pass
