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

from typing import Tuple, Dict, Union

POINT_ESTIMATE = "point_estimate"
VARIANCE = "variance"
CI_LOWER, CI_UPPER = "ci_lower", "ci_upper"
ADJUSTED_LOWER, ADJUSTED_UPPER = "adjusted ci_lower", "adjusted ci_upper"
DIFFERENCE = "difference"
P_VALUE = "p-value"
ADJUSTED_P = "adjusted p-value"
SFX1, SFX2 = "_1", "_2"
STD_ERR = "std_err"
Z_CRIT = "z_crit"
ALPHA = "alpha"
ADJUSTED_ALPHA = "adjusted_alpha"
ADJUSTED_ALPHA_POWER_SAMPLE_SIZE = "adjusted_alpha_power_sample_size"
POWER = "power"
POWERED_EFFECT = "powered_effect"
POWERED_EFFECT_METRIC = "powered_effect_for_metric"
ADJUSTED_POWER = "adjusted_power"
IS_SIGNIFICANT = "is_significant"
REQUIRED_SAMPLE_SIZE = "required_sample_size"
REQUIRED_SAMPLE_SIZE_METRIC = "required_sample_size_for_metric"

BONFERRONI = "bonferroni"
HOLM = "holm"
HOMMEL = "hommel"
SIMES_HOCHBERG = "simes-hochberg"
SIDAK = "sidak"
HOLM_SIDAK = "holm-sidak"
FDR_BH = "fdr_bh"
FDR_BY = "fdr_by"
FDR_TSBH = "fdr_tsbh"
FDR_TSBKY = "fdr_tsbky"

BONFERRONI_ONLY_COUNT_TWOSIDED = "bonferroni-only-count-twosided"
BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY = "bonferroni-do-not-count-non-inferiority"
SPOT_1 = "spot-1-bonferroni"

SPOT_1_HOLM = "spot-1-holm"
SPOT_1_HOMMEL = "spot-1-hommel"
SPOT_1_SIMES_HOCHBERG = "spot-1-simes-hochberg"
SPOT_1_SIDAK = "spot-1-sidak"
SPOT_1_HOLM_SIDAK = "spot-1-holm-sidak"
SPOT_1_FDR_BH = "spot-1-fdr_bh"
SPOT_1_FDR_BY = "spot-1-fdr_by"
SPOT_1_FDR_TSBH = "spot-1-fdr_tsbh"
SPOT_1_FDR_TSBKY = "spot-1-fdr_tsbky"

CORRECTION_METHODS = [
    BONFERRONI,
    HOLM,
    HOMMEL,
    SIMES_HOCHBERG,
    SIDAK,
    HOLM_SIDAK,
    FDR_BH,
    FDR_BY,
    FDR_TSBH,
    FDR_TSBKY,
    BONFERRONI_ONLY_COUNT_TWOSIDED,
    BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
    SPOT_1,
    SPOT_1_HOLM,
    SPOT_1_HOMMEL,
    SPOT_1_SIMES_HOCHBERG,
    SPOT_1_SIDAK,
    SPOT_1_HOLM_SIDAK,
    SPOT_1_FDR_BH,
    SPOT_1_FDR_BY,
    SPOT_1_FDR_TSBH,
    SPOT_1_FDR_TSBKY,
]

CORRECTION_METHODS_THAT_SUPPORT_CI = [
    BONFERRONI,
    HOLM,
    HOMMEL,
    SIMES_HOCHBERG,
    BONFERRONI_ONLY_COUNT_TWOSIDED,
    BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
    SPOT_1,
    SPOT_1_HOLM,
    SPOT_1_HOMMEL,
    SPOT_1_SIMES_HOCHBERG,
    SPOT_1_SIDAK,
    SPOT_1_HOLM_SIDAK,
    SPOT_1_FDR_BH,
    SPOT_1_FDR_BY,
    SPOT_1_FDR_TSBH,
    SPOT_1_FDR_TSBKY,
]

CORRECTION_METHODS_THAT_REQUIRE_METRIC_INFO = [
    BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
    SPOT_1,
    SPOT_1_HOLM,
    SPOT_1_HOMMEL,
    SPOT_1_SIMES_HOCHBERG,
    SPOT_1_SIDAK,
    SPOT_1_HOLM_SIDAK,
    SPOT_1_FDR_BH,
    SPOT_1_FDR_BY,
    SPOT_1_FDR_TSBH,
    SPOT_1_FDR_TSBKY,
]

CORRECTION_METHODS_THAT_DONT_REQUIRE_METRIC_INFO = list(
    set(CORRECTION_METHODS) - set(CORRECTION_METHODS_THAT_REQUIRE_METRIC_INFO)
)

NULL_HYPOTHESIS = "null_hypothesis"
ALTERNATIVE_HYPOTHESIS = "alternative_hypothesis"
NIM = "non-inferiority margin"
NIM_INPUT_COLUMN_NAME = "non_inferiority_margin"
PREFERRED_DIRECTION_INPUT_NAME = "preferred_direction"
INCREASE_PREFFERED = "increase"
DECREASE_PREFFERED = "decrease"
TWO_SIDED = "two-sided"
PREFERENCE = "preference"
PREFERENCE_TEST = "preference_used_in_test"
PREFERENCE_DICT = {"smaller": DECREASE_PREFFERED, "larger": INCREASE_PREFFERED, TWO_SIDED: TWO_SIDED}
NIM_TYPE = Union[Tuple[float, str], Dict[str, Tuple[float, str]], bool]
METHOD_COLUMN_NAME = "_method"
CHI2 = "chi-squared"
TTEST = "t-test"
ZTEST = "z-test"
BOOTSTRAP = "bootstrap"
METHODS = [CHI2, TTEST, ZTEST, BOOTSTRAP]
