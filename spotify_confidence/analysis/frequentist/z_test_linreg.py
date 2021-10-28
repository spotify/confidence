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

from typing import (Union, Iterable)

from pandas import DataFrame


from .confidence_computers.generic_computer import ZTestLinregComputer
from ..abstract_base_classes.confidence_computer_abc import \
    ConfidenceComputerABC
from ..abstract_base_classes.confidence_grapher_abc import ConfidenceGrapherABC
from .generic_test import GenericTest
from ..confidence_utils import listify
from spotify_confidence.analysis.constants import BONFERRONI, METHOD_COLUMN_NAME



class ZTestLinreg(GenericTest):

    def __init__(self,
                 data_frame: DataFrame,
                 numerator_column: str,
                 numerator_sum_squares_column: Union[str, None],
                 denominator_column: str,
                 feature_column: str,
                 feature_sum_squares_column: str,
                 feature_cross_sum_column: str,
                 categorical_group_columns: Union[str, Iterable],
                 ordinal_group_column: Union[str, None] = None,
                 interval_size: float = 0.95,
                 correction_method: str = BONFERRONI,
                 confidence_computer: ConfidenceComputerABC = None,
                 confidence_grapher: ConfidenceGrapherABC = None):

        computer = ZTestLinregComputer(
            numerator=numerator_column,
            numerator_sumsq=numerator_sum_squares_column,
            denominator=denominator_column,
            feature_column=feature_column,
            feature_sum_squares_column=feature_sum_squares_column,
            feature_cross_sum_column=feature_cross_sum_column,
            ordinal_group_column=ordinal_group_column,
            interval_size=interval_size,
            method_column=METHOD_COLUMN_NAME
        )

        super().__init__(
            data_frame.assign(**{METHOD_COLUMN_NAME: "z-test-linreg"}),
            numerator_column,
            numerator_sum_squares_column,
            denominator_column,
            categorical_group_columns,
            ordinal_group_column,
            interval_size,
            correction_method,
            computer,
            confidence_grapher,
            METHOD_COLUMN_NAME
        )

