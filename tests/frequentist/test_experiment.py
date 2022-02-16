import numpy as np
import pandas as pd

import spotify_confidence
from spotify_confidence.analysis.constants import METHOD_COLUMN_NAME, ZTEST


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
