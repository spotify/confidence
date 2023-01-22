import numpy as np
import pandas as pd
import pytest

import spotify_confidence
from spotify_confidence.analysis.constants import METHOD_COLUMN_NAME, ZTEST, ADJUSTED_LOWER, ADJUSTED_UPPER


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

    def test_multiple_differences_nims_false_none_true(self):
        pd.options.display.max_columns = None
        test = self.get_experiment_with_some_nims()

        diff_nim_false = test.multiple_difference(
            level="Control",
            level_as_reference=True,
            groupby="metric",
            non_inferiority_margins=False,
        )

        diff_nim_none = test.multiple_difference(
            level="Control",
            level_as_reference=True,
            groupby="metric",
            non_inferiority_margins=None,
        )

        assert np.allclose(diff_nim_false[ADJUSTED_LOWER], diff_nim_none[ADJUSTED_LOWER])
        assert np.allclose(diff_nim_false[ADJUSTED_UPPER], diff_nim_none[ADJUSTED_UPPER])

        diff_nim_true = test.multiple_difference(
            level="Control",
            level_as_reference=True,
            groupby="metric",
            non_inferiority_margins=True,
        )

        assert np.all(diff_nim_true[ADJUSTED_LOWER] > diff_nim_none[ADJUSTED_LOWER])
        assert np.all(diff_nim_true[ADJUSTED_UPPER] < diff_nim_none[ADJUSTED_UPPER])

    @pytest.mark.parametrize(
        "nims",
        [
            False,
            None,
            True,
        ],
        ids=lambda x: f"non_inferiority_margins: {x}",
    )
    def test_multiple_differences_plot_some_nims_doesnt_raise_exception(self, nims):
        pd.options.display.max_columns = None
        test = self.get_experiment_with_some_nims()
        try:
            ch = test.multiple_difference_plot(
                level="Control",
                level_as_reference=True,
                groupby="metric",
                absolute=True,
                use_adjusted_intervals=True,
                split_plot_by_groups=True,
                non_inferiority_margins=nims,
            )
            assert len(ch.charts) > 0
        except Exception as e:
            assert False, f"Using non_inferiority_margins={nims} raised an exception: {e}."

    def get_experiment_with_some_nims(self):
        columns = [
            "group_name",
            "num_user",
            "sum",
            "sum_squares",
            "method",
            "metric",
            "preferred_direction",
            "non_inferiority_margin",
        ]
        data = [
            ["Control", 6267728, 3240932, 3240932, "z-test", "m1", "increase", 0.15],
            ["Test", 6260737, 3239706, 3239706, "z-test", "m1", "increase", 0.15],
            ["Test", 6260737, 38600871, 12432573969, "z-test", "m2", "increase", None],
            ["Control", 6267728, 35963863, 18433512959, "z-test", "m2", "increase", None],
            ["Test", 6260737, 67382943, 8974188321, "z-test", "m3", "increase", None],
            ["Control", 6267728, 67111374, 8728934960, "z-test", "m3", "increase", None],
        ]
        df = pd.DataFrame(columns=columns, data=data)
        test = spotify_confidence.Experiment(
            data_frame=df.assign(**{METHOD_COLUMN_NAME: ZTEST}),
            numerator_column="sum",
            numerator_sum_squares_column="sum_squares",
            denominator_column="num_user",
            categorical_group_columns="metric",
            interval_size=0.99,
            correction_method="spot-1-bonferroni",
            metric_column="metric",
            treatment_column="group_name",
            method_column="method",
        )
        return test
