import numpy as np
import pandas as pd

from spotify_confidence.analysis.constants import (
    SPOT_1,
    REQUIRED_SAMPLE_SIZE_METRIC,
    CI_WIDTH,
    POWERED_EFFECT,
    POINT_ESTIMATE,
    ADJUSTED_LOWER,
)
from spotify_confidence.analysis.frequentist.sample_size_calculator import SampleSizeCalculator
from spotify_confidence.analysis.frequentist.z_test import ZTest

pd.options.display.max_columns = None
pd.options.display.width = 1000


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
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 1042868 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[1] / 95459 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 894863 < 1.001

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

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
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 1170185 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[1] / 95459 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 1004113 < 1.001

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

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
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 730009 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[1] / 58621 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 596991 < 1.001

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

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
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 718056 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[1] / 65337 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 586168 < 1.001

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["nim"].values[0], rtol=1e-3)

    def test_sample_size_5(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["share_bananas_1d", True, 0.7, 0.21, None, 0.00617, "increase"],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 553620 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 451934 < 1.001

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["nim"].values[0], rtol=1e-3)

    def test_sample_size_6(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, None],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 58622 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 47854 < 1.001

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

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
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 680575 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[1] / 52741 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 556565 < 1.001

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

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
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 595876 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[1] / 2103232 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[2] / 143396 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[3] / 685656 < 1.001
        assert ss[CI_WIDTH].isna().all()

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

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
            point_estimate_column="avg",
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
        assert ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] == np.float("inf")
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[1] / 75345 < 1.001

        np.testing.assert_almost_equal(ss[CI_WIDTH].values[0], 0.0047527)
        np.testing.assert_almost_equal(ss[CI_WIDTH].values[1], 0.0151362)

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isnan(relative_powered_effect.values[0])

    def test_sample_size_calculation_ciwidth_matches_real_width_returned_by_onesided_test(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference", "expected_sample_size"],
            data=[
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, "increase", int(1e6)],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 25334 < 1.001
        np.testing.assert_almost_equal(ss[CI_WIDTH].values[0], 0.0096023)

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

    def test_sample_size_calculation_ciwidth_matches_real_width_returned_by_twosided_test_with_direction(self):
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference", "expected_sample_size"],
            data=[
                ["bananas_per_user_7d", False, 4.56, 2.13, 0.01, None, None, int(1e6)],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            point_estimate_column="avg",
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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 32162 < 1.001
        np.testing.assert_almost_equal(ss[CI_WIDTH].values[0], 0.01144189)

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

    def test_ci_bound_equal_nim(self):
        mde_pp = 0.15
        baseline = 0.48
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                [
                    "share_users_with_bananas",
                    False,
                    baseline,
                    baseline * (1 - baseline),
                    None,
                    mde_pp / 100 / baseline,
                    "increase",
                ],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            point_estimate_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.95,
            power=0.5,
            correction_method="bonferroni",
        )
        treatment_weights = [0.5, 0.125, 0.125, 0.125, 0.125]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        assert len(ss) == len(df)
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0] / 5573168 < 1.001

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["nim"].values[0], rtol=1e-3)

        df_test = pd.DataFrame(
            data={
                "test_group": [
                    "control",
                    "same as control",
                    "MDE above control",
                    "2% of MDE above control",
                    "MDE below control",
                ],
                "num_users": [2786584, 696646, 696646, 696646, 696646],
                "num_users_with_bananas": [1337560.32, 334390.08, 342000, 341077.8816, 340000],
            }
        )

        test = ZTest(
            data_frame=df_test,
            numerator_column="num_users_with_bananas",
            numerator_sum_squares_column="num_users_with_bananas",
            denominator_column="num_users",
            categorical_group_columns="test_group",
            interval_size=0.95,
            power=0.5,
        )

        summary = test.summary()
        assert np.isclose(summary[POINT_ESTIMATE].values[0], 0.48)
        assert np.isclose(summary[POINT_ESTIMATE].values[1], 0.48)

        diff = test.multiple_difference(
            level="control", level_as_reference=True, non_inferiority_margins=(mde_pp / 100 / baseline, "increase")
        )

        assert np.isclose(diff.query("level_2 == 'same as control'")[ADJUSTED_LOWER], -mde_pp / 100)

    def test_ci_bound_equal_nim_2(self):
        mde_pp = 0.15
        baseline = 0.48
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                [
                    "share_users_with_bananas",
                    False,
                    baseline,
                    baseline * (1 - baseline),
                    None,
                    mde_pp / 100 / baseline,
                    "increase",
                ],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            point_estimate_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.95,
            power=0.5,
            correction_method="bonferroni",
        )
        treatment_weights = [0.75, 0.25]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        required_sample_size = ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0]

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["nim"].values[0], rtol=1e-3)

        df_test = pd.DataFrame(
            data={
                "test_group": [
                    "control",
                    "same as control",
                ],
                "num_users": [
                    required_sample_size * treatment_weights[0],
                    required_sample_size * treatment_weights[1],
                ],
                "num_users_with_bananas": [
                    required_sample_size * treatment_weights[0] * baseline,
                    required_sample_size * treatment_weights[1] * baseline,
                ],
            }
        )

        test = ZTest(
            data_frame=df_test,
            numerator_column="num_users_with_bananas",
            numerator_sum_squares_column="num_users_with_bananas",
            denominator_column="num_users",
            categorical_group_columns="test_group",
            interval_size=0.95,
            power=0.5,
        )

        summary = test.summary()
        assert np.isclose(summary[POINT_ESTIMATE].values[0], 0.48)
        assert np.isclose(summary[POINT_ESTIMATE].values[1], 0.48)

        diff = test.multiple_difference(
            level="control", level_as_reference=True, non_inferiority_margins=(mde_pp / 100 / baseline, "increase")
        )

        assert np.isclose(diff[ADJUSTED_LOWER], -mde_pp / 100)

    def test_ci_bound_equal_mde(self):
        mde_pp = 0.15
        baseline = 0.48
        df = pd.DataFrame(
            columns=["metric_name", "binary", "avg", "var", "mde", "nim", "preference"],
            data=[
                [
                    "share_users_with_bananas",
                    True,
                    baseline,
                    baseline * (1 - baseline),
                    mde_pp / 100 / baseline,
                    None,
                    None,
                ],
            ],
        )

        ssc = SampleSizeCalculator(
            data_frame=df,
            point_estimate_column="avg",
            var_column="var",
            metric_column="metric_name",
            is_binary_column="binary",
            interval_size=0.95,
            power=0.5,
            correction_method="bonferroni",
        )
        treatment_weights = [0.75, 0.25]
        ss = ssc.sample_size(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
        )

        required_sample_size = ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0]

        powered_effect_df = ssc.powered_effect(
            treatment_weights=treatment_weights,
            mde_column="mde",
            nim_column="nim",
            preferred_direction_column="preference",
            sample_size=ss[REQUIRED_SAMPLE_SIZE_METRIC].values[0],
        )

        relative_powered_effect = powered_effect_df[POWERED_EFFECT] / powered_effect_df["avg"]

        assert np.isclose(relative_powered_effect.values[0], df["mde"].values[0], rtol=1e-3)

        df_test = pd.DataFrame(
            data={
                "test_group": [
                    "control",
                    "same as control",
                ],
                "num_users": [
                    required_sample_size * treatment_weights[0],
                    required_sample_size * treatment_weights[1],
                ],
                "num_users_with_bananas": [
                    required_sample_size * treatment_weights[0] * baseline,
                    required_sample_size * treatment_weights[1] * baseline,
                ],
            }
        )

        test = ZTest(
            data_frame=df_test,
            numerator_column="num_users_with_bananas",
            numerator_sum_squares_column="num_users_with_bananas",
            denominator_column="num_users",
            categorical_group_columns="test_group",
            interval_size=0.95,
            power=0.5,
        )

        summary = test.summary()
        assert np.isclose(summary[POINT_ESTIMATE].values[0], 0.48)
        assert np.isclose(summary[POINT_ESTIMATE].values[1], 0.48)

        diff = test.multiple_difference(
            level="control",
            level_as_reference=True,  # non_inferiority_margins=(mde_pp / 100 / baseline, "increase")
        )

        assert np.isclose(diff[ADJUSTED_LOWER], -mde_pp / 100, rtol=0.001)
