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

from typing import (Tuple, Dict, Union)
POINT_ESTIMATE = "point_estimate"
VARIANCE = 'variance'
CI_LOWER, CI_UPPER = "ci_lower", "ci_upper"
ADJUSTED_LOWER, ADJUSTED_UPPER = "adjusted ci_lower", "adjusted ci_upper"
DIFFERENCE = "difference"
P_VALUE = "p-value"
ADJUSTED_P = "adjusted p-value"
SFX1, SFX2 = '_1', '_2'
STD_ERR = 'std_err'
Z_CRIT = "z_crit"
ALPHA = 'alpha'
ADJUSTED_ALPHA = 'adjusted_alpha'

BONFERRONI = "bonferroni"
HOLM = "holm"
HOMMEL = "hommel"
SIMES_HOCHBERG = "simes-hochberg"
BONFERRONI_ONLY_COUNT_TWOSIDED = "bonferroni-only-count-twosided"

NULL_HYPOTHESIS = "null_hypothesis"
NIM = "non-inferiority margin"
INCREASE_PREFFERED = "increase"
DECREASE_PREFFERED = "decrease"
TWO_SIDED = "two-sided"
PREFERENCE = "preference"
PREFERENCE_DICT = {'smaller': DECREASE_PREFFERED,
                   'larger': INCREASE_PREFFERED,
                   TWO_SIDED: TWO_SIDED}
NIM_TYPE = Union[Tuple[float, str], Dict[str, Tuple[float, str]]]
