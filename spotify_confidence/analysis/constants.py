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

NUMERATOR = "numerator"
NUMERATOR_SUM_OF_SQUARES = "numerator_sum_of_squares"
DENOMINATOR = "denominator"
BOOTSTRAPS = "bootstraps"
INTERVAL_SIZE = "interval_size"
ALPHA = "alpha"
FINAL_EXPECTED_SAMPLE_SIZE = "final_expected_sample_size"
ORDINAL_GROUP_COLUMN = "ordinal_group_column"
MDE = "mde"
METHOD = "method_column"
CORRECTION_METHOD = "correction_method"
ABSOLUTE = "absolute"
NUMBER_OF_COMPARISONS = "number_of_comparisons"
TREATMENT_WEIGHTS = "treatment_weights"
IS_BINARY = "is_binary"
FEATURE = "feature"
FEATURE_SUMSQ = "feature_sumsq"
FEATURE_CROSS = "feature_cross"
NUMBER_OF_COMPARISONS_VALIDATION = "number_of_comparisons_validation"

POINT_ESTIMATE = "point_estimate"
VARIANCE = "variance"
CI_LOWER, CI_UPPER = "ci_lower", "ci_upper"
ADJUSTED_LOWER, ADJUSTED_UPPER = "adjusted ci_lower", "adjusted ci_upper"
CI_WIDTH = "ci_width"
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
ADJUSTED_POWER = "adjusted_power"
IS_SIGNIFICANT = "is_significant"
REQUIRED_SAMPLE_SIZE = "required_sample_size"
REQUIRED_SAMPLE_SIZE_METRIC = "required_sample_size_for_metric"
OPTIMAL_KAPPA = "optimal_kappa"
OPTIMAL_WEIGHTS = "optimal_weigghts"
IS_FAILING = "is_failing_validation"
P_VALUE_VALIDATION = "p-value_validation"
ADJUSTED_P_VALIDATION = "adjusted p-value validation"
ADJUSTED_ALPHA_VALIDATION = "adjusted_alpha_validation"
VALIDATION_INTERVAL_SIZE = "validation_interval_size"
ALPHA_VALIDATION = "alpha_validation"
CI_LOWER_VALIDATION, CI_UPPER_VALIDATION = "ci_lower_validation", "ci_upper_validation"
ADJUSTED_LOWER_VALIDATION, ADJUSTED_UPPER_VALIDATION = "adjusted ci_lower_validation", "adjusted ci_upper_validation"

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
NIM_COLUMN_DEFAULT = "non_inferiority_margin"
PREFERRED_DIRECTION_COLUMN_DEFAULT = "preferred_direction"
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
ZTESTLINREG = "z-test-linreg"
BOOTSTRAP = "bootstrap"
METHODS = [CHI2, TTEST, ZTEST, BOOTSTRAP, ZTESTLINREG]
REGRESSION_PARAM = "regression_parameters"
VALIDATION = "validation"
SUCCESS = "success"
GUARDRAIL = "guardrail"
VALIDATIONS_ENABLED = "validations_enabled"

SEQUENTIAL_TEST = "sequential_test"
METRIC_CLASS = "metric_class"
SAMPLE_RATIO_MISMATCH = "sample ratio mismatch"
PRE_EXPOSURE_ACTIVITY = "pre-exposure activity"
TANKING = "tanking"
DECISION_DICT = {
    SAMPLE_RATIO_MISMATCH: VALIDATION,
    PRE_EXPOSURE_ACTIVITY: VALIDATION,
    TANKING: VALIDATION,
    SUCCESS: SUCCESS,
    GUARDRAIL: GUARDRAIL,
}
