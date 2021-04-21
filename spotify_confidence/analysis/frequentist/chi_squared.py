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

from typing import (Union, Iterable, Tuple)

from pandas import DataFrame

from .statsmodels_computer import ChiSquaredComputer
from ..abstract_base_classes.confidence_computer_abc import \
    ConfidenceComputerABC
from ..abstract_base_classes.confidence_grapher_abc import ConfidenceGrapherABC
from .generic_test import GenericTest
from ..confidence_utils import listify
from ..constants import BONFERRONI, NIM_TYPE


class ChiSquared(GenericTest):

    def __init__(self,
                 data_frame: DataFrame,
                 numerator_column: str,
                 denominator_column: str,
                 categorical_group_columns: Union[str, Iterable],
                 ordinal_group_column: Union[str, None] = None,
                 interval_size: float = 0.95,
                 correction_method: str = BONFERRONI,
                 confidence_computer: ConfidenceComputerABC = None,
                 confidence_grapher: ConfidenceGrapherABC = None):

        computer = ChiSquaredComputer(
            data_frame=data_frame,
            numerator_column=numerator_column,
            numerator_sum_squares_column=numerator_column,
            denominator_column=denominator_column,
            categorical_group_columns=listify(categorical_group_columns),
            ordinal_group_column=ordinal_group_column,
            interval_size=interval_size,
            correction_method=correction_method.lower())

        super(ChiSquared, self).__init__(
            data_frame,
            numerator_column,
            numerator_column,
            denominator_column,
            categorical_group_columns,
            ordinal_group_column,
            interval_size,
            correction_method,
            computer,
            confidence_grapher)

    def difference(self,
                   level_1: Union[str, Tuple],
                   level_2: Union[str, Tuple],
                   absolute: bool = True,
                   groupby: Union[str, Iterable] = None,
                   non_inferiority_margins: NIM_TYPE = None,
                   final_expected_sample_size: float = None
                   ) -> DataFrame:
        if non_inferiority_margins is not None:
            raise ValueError('Non-inferiority margins not supported in '
                             'ChiSquared. Use StudentsTTest or ZTest instead.')
        return super(ChiSquared, self).difference(
            level_1,
            level_2,
            absolute,
            groupby,
            None,
            final_expected_sample_size)

    def multiple_difference(self, level: Union[str, Tuple],
                            absolute: bool = True,
                            groupby: Union[str, Iterable] = None,
                            level_as_reference: bool = None,
                            non_inferiority_margins: NIM_TYPE = None,
                            final_expected_sample_size: float = None
                            ) -> DataFrame:
        if non_inferiority_margins is not None:
            raise ValueError('Non-inferiority margins not supported in '
                             'ChiSquared. Use StudentsTTest or ZTest instead.')
        return super(ChiSquared, self).multiple_difference(
            level,
            absolute,
            groupby,
            level_as_reference,
            None,
            final_expected_sample_size)
