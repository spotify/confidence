from enum import Enum

import numpy as np
import pandas as pd
import pytest

import spotify_confidence
from spotify_confidence.analysis.constants import METHOD_COLUMN_NAME, ZTEST, P_VALUE_VALIDATION, \
    ADJUSTED_ALPHA_VALIDATION, CI_LOWER_VALIDATION, CI_UPPER_VALIDATION, ADJUSTED_LOWER_VALIDATION, \
    ADJUSTED_UPPER_VALIDATION, IS_SIGNIFICANT_VALIDATION, PREFERENCE, TANKING, GUARDRAIL, SPOT_1, TWO_SIDED, P_VALUE, \
    ADJUSTED_P, IS_SIGNIFICANT, CI_LOWER, CI_UPPER, ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_ALPHA_POWER_SAMPLE_SIZE


class TestBootstrap(object):
    def setup(self):
        np.random.seed(123)
        n_bootstraps = int(5e5)
        self.data = pd.DataFrame(
            {
                "variation_name": ["test", "control", "test2", "test3"],
                "bootstraps": [
                    np.random.randn(n_bootstraps) + 0.5,
                    np.random.randn(n_bootstraps) + 0.4,
                    np.random.randn(n_bootstraps) + 0.2,
                    np.random.randn(n_bootstraps) + 1 / 3,
                ],
                "method": ["bootstrap", "bootstrap", "bootstrap", "bootstrap"],
            }
        )

        self.test = spotify_confidence.Experiment(
            self.data,
            numerator_column=None,
            numerator_sum_squares_column=None,
            denominator_column=None,
            categorical_group_columns="variation_name",
            method_column="method",
            bootstrap_samples_column="bootstraps",
        )

    def test_summary(self):
        summary_df = self.test.summary()
        assert len(summary_df) == len(self.data)

    def test_difference(self):
        difference_df = self.test.difference(level_1="control", level_2="test", absolute=True)
        assert len(difference_df) == 1
        assert np.allclose(difference_df["difference"][0], 0.1, atol=1e-2)

    def test_difference_absolute_false(self):
        difference_df = self.test.difference(level_1="control", level_2="test", absolute=False)
        assert len(difference_df) == 1
        assert np.allclose(difference_df["difference"][0], 0.1 / 0.4, atol=1e-2)

    def test_multiple_difference(self):
        difference_df = self.test.multiple_difference(level="control", level_as_reference=True)
        assert len(difference_df) == self.data.variation_name.unique().size - 1
        assert np.allclose(difference_df["difference"][0], 0.1, atol=1e-2)

    def test_multiple_difference_level_as_reference_false(self):
        difference_df_true_true = self.test.multiple_difference(
            level="control", level_as_reference=True, absolute=True
        )

        difference_df = self.test.multiple_difference(level="control", level_as_reference=False, absolute=True)
        assert len(difference_df) == self.data.variation_name.unique().size - 1
        assert np.allclose(difference_df["difference"], -difference_df_true_true["difference"], atol=0)
        assert np.allclose(difference_df["ci_lower"], -difference_df_true_true["ci_upper"], atol=0)
        assert np.allclose(difference_df["ci_upper"], -difference_df_true_true["ci_lower"], atol=0)
        assert np.allclose(difference_df["p-value"], difference_df_true_true["p-value"], atol=0)

    def test_multiple_difference_absolute_false(self):
        control_mean = self.test.summary().query("variation_name == 'control'")["point_estimate"].values[0]
        difference_df_true_true = self.test.multiple_difference(
            level="control", level_as_reference=True, absolute=True
        )

        difference_df = self.test.multiple_difference(level="control", level_as_reference=True, absolute=False)
        assert len(difference_df) == self.data.variation_name.unique().size - 1
        assert np.allclose(difference_df["difference"], difference_df_true_true["difference"] / control_mean, atol=0)
        assert np.allclose(difference_df["ci_lower"], difference_df_true_true["ci_lower"] / control_mean, atol=0)
        assert np.allclose(difference_df["ci_upper"], difference_df_true_true["ci_upper"] / control_mean, atol=0)
        assert np.allclose(difference_df["p-value"], difference_df_true_true["p-value"], atol=0)

    def test_multiple_difference_level_as_reference_false_absolute_false(self):
        reference_mean = self.test.summary().query("variation_name != 'control'")["point_estimate"]
        difference_df_true_true = self.test.multiple_difference(
            level="control", level_as_reference=True, absolute=True
        )

        difference_df = self.test.multiple_difference(level="control", level_as_reference=False, absolute=False)
        assert len(difference_df) == self.data.variation_name.unique().size - 1
        assert np.allclose(
            difference_df["difference"], -difference_df_true_true["difference"] / reference_mean.values, atol=0
        )
        assert np.allclose(
            difference_df["ci_lower"], -difference_df_true_true["ci_upper"] / reference_mean.values, atol=0
        )
        assert np.allclose(
            difference_df["ci_upper"], -difference_df_true_true["ci_lower"] / reference_mean.values, atol=0
        )
        assert np.allclose(difference_df["p-value"], difference_df_true_true["p-value"], atol=0)

    def test_summary_plot(self):
        chart_grid = self.test.summary_plot()
        assert len(chart_grid.charts) == 1

    def test_difference_plot(self):
        chartgrid = self.test.difference_plot(level_1="control", level_2="test")
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot(self):
        chartgrid = self.test.multiple_difference_plot(level="control", level_as_reference=True)
        assert len(chartgrid.charts) == 1

    def test_differences_one_level_on_both_sides(self):
        df = pd.DataFrame(
            {
                "group_name": {0: "LT", 1: "HB", 2: "RNDM", 3: "ST"},
                "num_user": {0: 5832071, 1: 5830464, 2: 5775235, 3: 5829780},
                "sum": {
                    0: 3650267885.292686,
                    1: 3640976251.2644653,
                    2: 3543904424.4249864,
                    3: 3640408188.9692664,
                },
                "sum_squares": {
                    0: 11464986242442.066,
                    1: 11395508685623.664,
                    2: 10953117763878.217,
                    3: 11400833683366.701,
                },
            }
        )

        test = spotify_confidence.Experiment(
            data_frame=df.assign(**{METHOD_COLUMN_NAME: ZTEST}),
            numerator_column="sum",
            numerator_sum_squares_column="sum_squares",
            denominator_column="num_user",
            categorical_group_columns="group_name",
            method_column=METHOD_COLUMN_NAME,
        )

        diff = test.differences(
            levels=[
                ("RNDM", "HB"),
                ("ST", "HB"),
                ("LT", "HB"),
                ("ST", "LT"),
            ]
        )

        assert len(diff) == 4


class TestSequentialOrdinalPlusTwoCategorical2Tanking(object):
    def setup(self):
        DATE, GROUP, COUNT, SUM, SUM_OF_SQUARES = "date", "variation_name", "count", "sum", "sum_of_squares"
        self.data = pd.DataFrame(
            [
                {
                    "date": "2020-04-01",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 2016.416,
                    SUM_OF_SQUARES: 5082.122,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-01",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 2028.478,
                    SUM_OF_SQUARES: 5210.193,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-01",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1991.554,
                    SUM_OF_SQUARES: 4919.282,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-01",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1958.713,
                    SUM_OF_SQUARES: 4818.665,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-01",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 2030.252,
                    SUM_OF_SQUARES: 5129.574,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-01",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1966.138,
                    SUM_OF_SQUARES: 4848.321,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-01",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1995.389,
                    SUM_OF_SQUARES: 4992.710,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-01",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1952.098,
                    SUM_OF_SQUARES: 4798.772,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-02",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2986.667,
                    SUM_OF_SQUARES: 7427.582,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-02",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2989.488,
                    SUM_OF_SQUARES: 7421.710,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-02",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 3008.681,
                    SUM_OF_SQUARES: 7565.406,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-02",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2933.173,
                    SUM_OF_SQUARES: 7207.038,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-02",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2986.308,
                    SUM_OF_SQUARES: 7584.148,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-02",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2985.802,
                    SUM_OF_SQUARES: 7446.539,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-02",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 3008.190,
                    SUM_OF_SQUARES: 7532.521,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-02",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 3001.494,
                    SUM_OF_SQUARES: 7467.535,
                    "non_inferiority_margin": None,
                    "preferred_direction": None,
                },
                {
                    "date": "2020-04-01",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 2016.416,
                    SUM_OF_SQUARES: 5082.122,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-01",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 2028.478,
                    SUM_OF_SQUARES: 5210.193,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-01",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1991.554,
                    SUM_OF_SQUARES: 4919.282,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-01",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1958.713,
                    SUM_OF_SQUARES: 4818.665,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-01",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 2030.252,
                    SUM_OF_SQUARES: 5129.574,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-01",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1966.138,
                    SUM_OF_SQUARES: 4848.321,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-01",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1995.389,
                    SUM_OF_SQUARES: 4992.710,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-01",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1952.098,
                    SUM_OF_SQUARES: 4798.772,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-02",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2986.667,
                    SUM_OF_SQUARES: 7427.582,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-02",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2989.488,
                    SUM_OF_SQUARES: 7421.710,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-02",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 3008.681,
                    SUM_OF_SQUARES: 7565.406,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-02",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2933.173,
                    SUM_OF_SQUARES: 7207.038,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-02",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2986.308,
                    SUM_OF_SQUARES: 7584.148,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-02",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2985.802,
                    SUM_OF_SQUARES: 7446.539,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-02",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 3008.190,
                    SUM_OF_SQUARES: 7532.521,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
                {
                    "date": "2020-04-02",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 3001.494,
                    SUM_OF_SQUARES: 7467.535,
                    "non_inferiority_margin": 0.01,
                    "preferred_direction": "increase",
                },
            ]
        )
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data = (
            self.data.groupby(["date", GROUP, "country", "platform", "metric"])
            .sum()
            .groupby([GROUP, "country", "platform", "metric"])
            .cumsum()
            .reset_index()
            .assign(
                non_inferiority_margin=lambda df: df["metric"].map(
                    {"bananas_per_user_1d": None, "bananas_per_user_7d": 0.01}
                )
            )
            .assign(
                preferred_direction=lambda df: df["metric"].map(
                    {"bananas_per_user_1d": None, "bananas_per_user_7d": "increase"}
                )
            )
            .assign(final_expected_sample_size=5000)
            .assign(method="z-test")
        )
        self.data["decision_type"] = TANKING
        self.data.loc[self.data["metric"] == "bananas_per_user_7d", "decision_type"] = GUARDRAIL

        self.test3 = spotify_confidence.Experiment(
            self.data,
            numerator_column=SUM,
            numerator_sum_squares_column=SUM_OF_SQUARES,
            denominator_column=COUNT,
            categorical_group_columns=[GROUP, "country", "platform", "metric"],
            ordinal_group_column="date",
            interval_size=1 - 0.01,
            correction_method=SPOT_1,
            method_column="method",
            metric_column="metric",
            treatment_column=GROUP,
            validations=True,
            decision_column="decision_type",
            sequential_test=False,
        )

    def test_validation_one_guardrail_one_validation_metric(self):

        exp = spotify_confidence.Experiment(
            self.data,
            numerator_column="sum",
            numerator_sum_squares_column="sum_of_squares",
            denominator_column="count",
            categorical_group_columns=["variation_name", "country", "platform", "metric"],
            ordinal_group_column="date",
            interval_size=1 - 0.01,
            correction_method=SPOT_1,
            method_column="method",
            metric_column="metric",
            treatment_column="variation_name",
            validations=True,
            decision_column="decision_type",
        )
        difference_df = exp.multiple_difference(
            level="1",
            groupby=["date", "country", "platform", "metric"],
            level_as_reference=True,
            final_expected_sample_size_column="final_expected_sample_size",
            non_inferiority_margins=True,
            verbose=True,
        )
        assert IS_SIGNIFICANT_VALIDATION in difference_df.columns

    def test_validation_one_guardrail_one_success_metric_no_sequential(self):
        difference_df = self.test3.multiple_difference(
            level="1",
            groupby=["date", "country", "platform", "metric"],
            level_as_reference=True,
            final_expected_sample_size_column="final_expected_sample_size",
            non_inferiority_margins=True,
            verbose=True,
        )
        assert IS_SIGNIFICANT_VALIDATION in difference_df.columns
        for column_name in [
            P_VALUE_VALIDATION,
            ADJUSTED_ALPHA_VALIDATION,
            CI_LOWER_VALIDATION,
            CI_UPPER_VALIDATION,
            ADJUSTED_LOWER_VALIDATION,
            ADJUSTED_UPPER_VALIDATION,
            IS_SIGNIFICANT_VALIDATION,
        ]:
            assert difference_df.loc[difference_df[PREFERENCE] == TWO_SIDED, column_name].isnull().all()
        for column_name in [
            P_VALUE,
            ADJUSTED_P,
            IS_SIGNIFICANT,
            CI_LOWER,
            CI_UPPER,
            ADJUSTED_LOWER,
            ADJUSTED_UPPER,
            ADJUSTED_ALPHA_POWER_SAMPLE_SIZE,
        ]:
            assert difference_df.loc[difference_df["date"] == "2020-04-01", column_name].isnull().all()

    def test_validation_one_guardrail_one_success_metric_no_sequential_recommendation(self):
        difference_df = self.test3.multiple_difference(
            level="1",
            groupby=["date", "country", "platform", "metric"],
            level_as_reference=True,
            final_expected_sample_size_column="final_expected_sample_size",
            non_inferiority_margins=True,
            verbose=True,
        )
        assert isinstance(self.test3.get_recommendation(difference_df), Enum)

    def test_no_validation_one_guardrail_one_success_metric_no_sequential(self):
        self.data["decision_type"] = GUARDRAIL
        exp = spotify_confidence.Experiment(
            self.data,
            numerator_column="sum",
            numerator_sum_squares_column="sum_of_squares",
            denominator_column="count",
            categorical_group_columns=["variation_name", "country", "platform", "metric"],
            ordinal_group_column="date",
            interval_size=1 - 0.01,
            correction_method=SPOT_1,
            method_column="method",
            metric_column="metric",
            treatment_column="variation_name",
            validations=False,
            decision_column="decision_type",
            sequential_test=False,
        )
        difference_df = exp.multiple_difference(
            level="1",
            groupby=["date", "country", "platform", "metric"],
            level_as_reference=True,
            final_expected_sample_size_column="final_expected_sample_size",
            non_inferiority_margins=True,
            verbose=True,
        )
        for column_name in [
            P_VALUE_VALIDATION,
            ADJUSTED_ALPHA_VALIDATION,
            CI_LOWER_VALIDATION,
            CI_UPPER_VALIDATION,
            ADJUSTED_LOWER_VALIDATION,
            ADJUSTED_UPPER_VALIDATION,
            IS_SIGNIFICANT_VALIDATION,
        ]:
            assert column_name not in difference_df.columns

    def test_validation_one_guardrail_one_success_metric_wrong_decision_type(self):
        self.data["decision_type"] = "test"
        with pytest.raises(ValueError):
            spotify_confidence.Experiment(
                self.data,
                numerator_column="sum",
                numerator_sum_squares_column="sum_of_squares",
                denominator_column="count",
                categorical_group_columns=["variation_name", "country", "platform", "metric"],
                ordinal_group_column="date",
                interval_size=1 - 0.01,
                correction_method=SPOT_1,
                method_column="method",
                metric_column="metric",
                treatment_column="variation_name",
                validations=True,
                decision_column="decision_type",
                sequential_test=False,
            )