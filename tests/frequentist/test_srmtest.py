import numpy as np
import pandas as pd
import pytest

import spotify_confidence
from spotify_confidence.analysis.constants import (
    INCREASE_PREFFERED,
    DECREASE_PREFFERED,
    POINT_ESTIMATE,
    CI_LOWER,
    CI_UPPER,
    P_VALUE,
    ADJUSTED_LOWER,
    ADJUSTED_UPPER,
    DIFFERENCE,
    BONFERRONI,
    BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
    CORRECTION_METHODS,
    SPOT_1,
    CORRECTION_METHODS_THAT_SUPPORT_CI,
    SRMTEST, IS_SIGNIFICANT_VALIDATION, SAMPLE_RATIO_MISMATCH,
)


class TestSequentialOrdinalPlusTwoCategorical(object):
    def setup(self):
        self.data = pd.DataFrame(
            {
                "variation_name": [
                    "test",
                    "control",
                    "test2",
                    "test",
                    "control",
                    "test2",
                    "test",
                    "control",
                    "test2",
                ],
                "sample_size": [100, 99, 102, 200, 196, 204, 300, 275, 306],
                "date": pd.to_datetime(
                    [
                        "2021-04-01",
                        "2021-04-01",
                        "2021-04-01",
                        "2021-04-02",
                        "2021-04-02",
                        "2021-04-02",
                        "2021-04-03",
                        "2021-04-03",
                        "2021-04-03",
                    ]
                ),
                "expected_proportions": [0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4],
                "metric": [
                    "m1",
                    "m1",
                    "m1",
                    "m1",
                    "m1",
                    "m1",
                    "m1",
                    "m1",
                    "m1",
                ],
                "method": [
                    SRMTEST,
                    SRMTEST,
                    SRMTEST,
                    SRMTEST,
                    SRMTEST,
                    SRMTEST,
                    SRMTEST,
                    SRMTEST,
                    SRMTEST,
                ],
                "decision_type": [
                    SAMPLE_RATIO_MISMATCH,
                    SAMPLE_RATIO_MISMATCH,
                    SAMPLE_RATIO_MISMATCH,
                    SAMPLE_RATIO_MISMATCH,
                    SAMPLE_RATIO_MISMATCH,
                    SAMPLE_RATIO_MISMATCH,
                    SAMPLE_RATIO_MISMATCH,
                    SAMPLE_RATIO_MISMATCH,
                    SAMPLE_RATIO_MISMATCH,
                ],
                "preferred_direction": [
                    None, None, None, None, None, None, None, None, None
                ]
            }
        ).assign(final_sample_size=900)

        self.test = spotify_confidence.Experiment(
            self.data,
            denominator_column="sample_size",
            numerator_column="expected_proportions",
            numerator_sum_squares_column=None,
            categorical_group_columns=["variation_name", "metric"],
            ordinal_group_column="date",
            metric_column="metric",
            method_column="method",
            treatment_column="variation_name",
            validations=True,
            correction_method=SPOT_1,
            decision_column="decision_type"
        )

    def test_srm(self):
        outcome = self.test.multiple_difference(
            level="control",
            groupby=["date", "metric"],
            level_as_reference=True,
            final_expected_sample_size_column="final_sample_size",
        )
        assert outcome[IS_SIGNIFICANT_VALIDATION][0] == 0
        assert outcome[IS_SIGNIFICANT_VALIDATION][5] == 1
