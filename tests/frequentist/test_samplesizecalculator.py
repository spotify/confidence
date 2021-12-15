import numpy as np
import pandas as pd

from spotify_confidence.analysis.constants import (
    SPOT_1,
    REQUIRED_SAMPLE_SIZE_METRIC,
    CI_WIDTH,
)
from spotify_confidence.analysis.frequentist.sample_size_calculator import SampleSizeCalculator


class TestSampleSizeCalculator(object):
    def test_sample_size_1(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["share_bananas_1d", True, 0.7, 0.21, 0.00617, None, "increase"],
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, "increase"],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.99,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 2000, 3000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["share_bananas_1d", REQUIRED_SAMPLE_SIZE_METRIC] / 1042868 < 1.001
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 95459 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 894863 < 1.001

    def test_sample_size_2(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["share_bananas_1d", True, 0.7, 0.21, 0.00617, None, None],
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, "increase"],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.99,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 2000, 3000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["share_bananas_1d", REQUIRED_SAMPLE_SIZE_METRIC] / 1170185 < 1.001
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 95459 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 1004113 < 1.001

    def test_sample_size_3(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["share_bananas_1d", True, 0.7, 0.21, 0.00617, None, None],
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, "increase"],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.99,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 2000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["share_bananas_1d", REQUIRED_SAMPLE_SIZE_METRIC] / 730009 < 1.001
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 58621 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 596991 < 1.001

    def test_sample_size_4(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["share_bananas_1d", True, 0.7, 0.21, None, 0.00617, "increase"],
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, "increase"],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.99,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 2000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["share_bananas_1d", REQUIRED_SAMPLE_SIZE_METRIC] / 718056 < 1.001
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 65337 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 586168 < 1.001

    def test_sample_size_5(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["share_bananas_1d", True, 0.7, 0.21, None, 0.00617, "increase"],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.99,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 2000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["share_bananas_1d", REQUIRED_SAMPLE_SIZE_METRIC] / 553620 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 451934 < 1.001

    def test_sample_size_6(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, None],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.99,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 2000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 58622 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 47854 < 1.001

    def test_sample_size_7(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["share_bananas_1d", True, 0.7, 0.21, 0.00617, None, None],
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, "increase"],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.95,
            power=0.9,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 2000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["share_bananas_1d", REQUIRED_SAMPLE_SIZE_METRIC] / 680575 < 1.001
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 52741 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 556565 < 1.001

    def test_sample_size_8(self):
        df = pd.DataFrame(
            columns=["metric_name", "country", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["share_bananas_1d", "denmark", True, 0.7, 0.21, 0.01, None, None],
                ["share_bananas_1d", "sweden", True, 0.4, 0.24, 0.01, None, None],
                ["bananas_per_user_7d", "denmark", False, 4.56, 2.13, 0.01, None, None],
                ["bananas_per_user_7d", "sweden", False, 3.81, 7.11, 0.01, None, None],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            categorical_group_columns="country",
            is_binary_column="binary",
            interval_size=0.99,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [1, 1, 1, 1]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc[("denmark", "share_bananas_1d"), REQUIRED_SAMPLE_SIZE_METRIC] / 595876 < 1.001
        assert 0.999 < ss.loc[("sweden", "share_bananas_1d"), REQUIRED_SAMPLE_SIZE_METRIC] / 2103232 < 1.001
        assert 0.999 < ss.loc[("denmark", "bananas_per_user_7d"), REQUIRED_SAMPLE_SIZE_METRIC] / 143396 < 1.001
        assert 0.999 < ss.loc[("sweden", "bananas_per_user_7d"), REQUIRED_SAMPLE_SIZE_METRIC] / 685656 < 1.001
        assert ss[CI_WIDTH].isna().all()

    def test_sample_size_calculation_ciwidth_nimless_with_expected_sample_size(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference", "expected_sample_size"],
            data=[
                ["share_bananas_1d", True, 0.7, 0.21, None, 0.0, "increase", 1e6],
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, "increase", int(1e6)],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.95,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 2000, 3000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            final_expected_sample_size_column="expected_sample_size",
        )

        assert len(ss) == len(df)
        assert ss.loc["share_bananas_1d", REQUIRED_SAMPLE_SIZE_METRIC] == np.float("inf")
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 75345 < 1.001

        np.testing.assert_almost_equal(ss.loc["share_bananas_1d", CI_WIDTH], 0.0047527)
        np.testing.assert_almost_equal(ss.loc["bananas_per_user_7d", CI_WIDTH], 0.0151362)

    def test_sample_size_calculation_ciwidth_matches_real_width_returned_by_onesided_test(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference", "expected_sample_size"],
            data=[
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, "increase", int(1e6)],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.95,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 5000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            final_expected_sample_size_column="expected_sample_size",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 25334 < 1.001
        np.testing.assert_almost_equal(ss.loc["bananas_per_user_7d", CI_WIDTH], 0.0096023)

    def test_sample_size_calculation_ciwidth_matches_real_width_returned_by_twosided_test_with_direction(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference", "expected_sample_size"],
            data=[
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, None, int(1e6)],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            avg_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.95,
            power=0.8,
            correction_method=SPOT_1,
        )
        treatment_weights = [5000, 5000]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            final_expected_sample_size_column="expected_sample_size",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss.loc["bananas_per_user_7d", REQUIRED_SAMPLE_SIZE_METRIC] / 32162 < 1.001
        np.testing.assert_almost_equal(ss.loc["bananas_per_user_7d", CI_WIDTH], 0.01144189)
