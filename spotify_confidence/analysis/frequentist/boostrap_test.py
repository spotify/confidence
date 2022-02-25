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

from typing import Union, Iterable

from pandas import DataFrame

from spotify_confidence.analysis.frequentist.confidence_computers.generic_computer import GenericComputer
from spotify_confidence.analysis.abstract_base_classes.confidence_computer_abc import ConfidenceComputerABC
from spotify_confidence.analysis.abstract_base_classes.confidence_grapher_abc import ConfidenceGrapherABC
from spotify_confidence.analysis.frequentist.generic_test import GenericTest
from spotify_confidence.analysis.confidence_utils import listify
from spotify_confidence.analysis.constants import BONFERRONI, METHOD_COLUMN_NAME


class BootstrapTest(GenericTest):
    def __init__(
        self,
        data_frame: DataFrame,
        bootstrap_samples_column: [],
        categorical_group_columns: Union[str, Iterable],
        ordinal_group_column: Union[str, None] = None,
        interval_size: float = 0.95,
        correction_method: str = BONFERRONI,
        confidence_computer: ConfidenceComputerABC = None,
        confidence_grapher: ConfidenceGrapherABC = None,
    ):

        if confidence_computer is None:
            confidence_computer = GenericComputer(
                data_frame=data_frame.assign(**{METHOD_COLUMN_NAME: "bootstrap"}),
                numerator_column=None,
                numerator_sum_squares_column=None,
                denominator_column=None,
                categorical_group_columns=listify(categorical_group_columns),
                ordinal_group_column=ordinal_group_column,
                interval_size=interval_size,
                correction_method=correction_method.lower(),
                method_column=METHOD_COLUMN_NAME,
                bootstrap_samples_column=bootstrap_samples_column,
            )

        super(BootstrapTest, self).__init__(
            data_frame.assign(**{METHOD_COLUMN_NAME: "bootstrap"}),
            None,
            None,
            None,
            categorical_group_columns,
            ordinal_group_column,
            interval_size,
            correction_method,
            confidence_computer,
            confidence_grapher,
            METHOD_COLUMN_NAME,
            bootstrap_samples_column,
        )
