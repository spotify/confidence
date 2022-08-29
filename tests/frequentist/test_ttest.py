"""Tests for `confidence` categorical variables."""

import numpy as np
import pandas as pd
import pytest

import spotify_confidence
from spotify_confidence.analysis.constants import (
    POINT_ESTIMATE,
    VARIANCE,
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    DENOMINATOR,
    SFX1,
    SFX2,
    INCREASE_PREFFERED,
    DECREASE_PREFFERED,
    DIFFERENCE,
    P_VALUE,
    CI_UPPER,
    CI_LOWER,
)
from spotify_confidence.analysis.frequentist.confidence_computers import t_test_computer as computer


def chart_data(chart_object, series_name):
    """Retrieve data from chart object based on series name.

    Note: if there's only one series the name is None.

    Returns:
        Dictionary populated with data from the chart.
    """
    cannonical_series_name = chart_object.plot._cannonical_series_name(series_name)
    return chart_object.figure.select(cannonical_series_name)[0].data


class TestCategorical(object):
    def setup(self):
        np.random.seed(123)

        self.data = pd.DataFrame(
            {
                "variation_name": ["test", "control", "test2", "test", "control", "test2"],
                "nr_of_items": [1969, 312, 2955, 195, 24, 330],
                "nr_of_items_sumsq": [5767, 984, 8771, 553, 80, 1010],
                "users": [1009, 104, 1502, 100, 10, 150],
                "country": [
                    "us",
                    "us",
                    "us",
                    "gb",
                    "gb",
                    "gb",
                ],
            }
        )

        self.test = spotify_confidence.StudentsTTest(
            self.data,
            numerator_column="nr_of_items",
            numerator_sum_squares_column="nr_of_items_sumsq",
            denominator_column="users",
            categorical_group_columns=["country", "variation_name"],
            interval_size=0.95,
        )

    def test_sample_variance(self):
        n = 1000
        x = np.random.randn(n)
        var = x.var()

        sum = np.sum(x)
        sum_squared = np.sum(x * x)

        arg_dict = {NUMERATOR: "sum", NUMERATOR_SUM_OF_SQUARES: "sumsq", DENOMINATOR: "n"}
        var_to_verify = pd.DataFrame(
            {"sum": [sum], "sumsq": [sum_squared], POINT_ESTIMATE: [x.mean()], "n": [n]}
        ).apply(lambda row: computer.variance(row, arg_dict), axis=1)

        assert np.allclose(var_to_verify, var, rtol=2e-3)

    def test_standard_error_diff_mean_from_sums(self):
        n1 = int(1e6)
        mu1 = 1
        sigma1 = 1
        x1 = np.random.normal(mu1, sigma1, n1)

        n2 = int(1e6)
        mu2 = 2
        sigma2 = 2
        x2 = np.random.normal(mu2, sigma2, n2)

        std_diff = np.sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2)

        arg_dict = {DENOMINATOR: "n"}
        diff_se = computer.std_err(
            pd.DataFrame(
                {
                    VARIANCE + SFX1: [x1.var()],
                    VARIANCE + SFX2: [x2.var()],
                    "n" + SFX1: [n1],
                    "n" + SFX2: [n2],
                    "method": ["t-test"],
                }
            ),
            arg_dict,
        )

        assert np.allclose(std_diff, diff_se, atol=1e-5)

    def test_summary(self):
        summary = self.test.summary()

        assert np.array_equal(summary.country, np.array(["us", "us", "us", "gb", "gb", "gb"]))
        assert np.array_equal(summary.point_estimate, self.data.nr_of_items / self.data.users)

        assert np.allclose(
            summary["ci_lower"],
            np.array([1.86607468, 2.86724041, 1.89633269, 1.68789181, 1.27143773, 1.97725383]),
            rtol=1e-3,
        )

        assert np.allclose(
            summary["ci_upper"],
            np.array([2.03679946, 3.13275959, 2.03842097, 2.21210819, 3.52856227, 2.42274617]),
            rtol=2e-3,
        )

    def test_summary_plot(self):
        """Area plot tests"""
        ch = self.test.summary_plot()
        assert len(ch.charts) == 1
        ch = self.test.summary_plot(groupby="country")
        assert len(ch.charts) == 2

    def test_p_value_is_symmetric_and_correct(self):
        df = pd.DataFrame(
            {
                "group_name": ["Control", "Test"],
                "users": [48351, 50571],
                "nr_of_items": [1.438602e06, 1.521974e06],
                "nr_of_items_sumsq": [7.330581e07, 7.862121e07],
            }
        )

        ttest = spotify_confidence.StudentsTTest(
            data_frame=df,
            denominator_column="users",
            numerator_column="nr_of_items",
            numerator_sum_squares_column="nr_of_items_sumsq",
            categorical_group_columns="group_name",
        )

        diff_summary_1 = ttest.difference(level_1="Control", level_2="Test")
        diff_summary_2 = ttest.difference(level_1="Test", level_2="Control")

        assert diff_summary_1["p-value"].iloc[0] == diff_summary_2["p-value"].iloc[0]
        assert np.isclose(diff_summary_1["p-value"].iloc[0], 0.03334, 0.01)

    def test_difference(self):
        diff = self.test.difference(("us", "control"), ("us", "test"))
        assert np.allclose(diff["difference"].iloc[0], -1.0485629, rtol=1e-3)
        assert np.allclose(diff["ci_lower"].iloc[0], -1.205402, rtol=1e-3)
        assert np.allclose(diff["ci_upper"].iloc[0], -0.891723, rtol=1e-3)
        assert np.allclose(diff["p-value"].iloc[0], 0.00000, rtol=1e-3)

        diff = self.test.difference(("gb", "control"), ("gb", "test2"))
        assert np.allclose(diff["difference"].iloc[0], -0.2, rtol=1e-3)
        assert np.allclose(diff["ci_lower"].iloc[0], -1.340533, rtol=1e-3)
        assert np.allclose(diff["ci_upper"].iloc[0], 0.940533, rtol=1e-3)
        assert np.allclose(diff["p-value"].iloc[0], 0.704021, rtol=1e-3)

        diff = self.test.difference(("us", "control"), ("us", "test"), absolute=False)
        assert np.allclose(diff["difference"].iloc[0], -0.349520, rtol=1e-3)
        assert np.allclose(diff["ci_lower"].iloc[0], -0.4018008, rtol=1e-3)
        assert np.allclose(diff["ci_upper"].iloc[0], -0.297241, rtol=1e-3)
        assert np.allclose(diff["p-value"].iloc[0], 0.00000, rtol=1e-3)

        diff = self.test.difference("control", "test", groupby="country")
        assert np.allclose(diff["difference"], np.array([-0.450000, -1.048563]), rtol=1e-3)
        assert np.allclose(diff["ci_lower"], np.array([-1.595350, -1.205963]), rtol=1e-3)
        assert np.allclose(diff["ci_upper"], np.array([0.695350, -0.891163]), rtol=1e-3)
        assert np.allclose(diff["p-value"], np.array([0.4030961, 0.000000]), rtol=1e-3)

    def test_difference_with_interval_sizes(self):
        """
        https: // www.quantitativeskills.com / sisa / statistics / t - test.htm
        was used to validate results
        """
        self.test._confidence_computer._interval_size = 0.99

        diff = self.test.difference(("us", "control"), ("us", "test"))
        assert np.allclose(diff["difference"].iloc[0], -1.0485629, rtol=1e-3)
        assert np.allclose(diff["ci_lower"].iloc[0], -1.255391, rtol=1e-3)
        assert np.allclose(diff["ci_upper"].iloc[0], -0.841735, rtol=1e-3)
        assert np.allclose(diff["p-value"].iloc[0], 0.00000, rtol=1e-3)

        self.test._confidence_computer._interval_size = 0.999
        diff = self.test.difference(("us", "control"), ("us", "test"))
        assert np.allclose(diff["difference"].iloc[0], -1.0485629, rtol=1e-3)
        assert np.allclose(diff["ci_lower"].iloc[0], -1.314140, rtol=1e-3)
        assert np.allclose(diff["ci_upper"].iloc[0], -0.782024, rtol=1e-3)
        assert np.allclose(diff["p-value"].iloc[0], 0.00000, rtol=1e-3)

    def test_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.difference_plot(("bad_value", "bad_value"), ("bad_value", "bad_value"))

        ch = self.test.difference_plot(("us", "control"), ("us", "test"))
        assert len(ch.charts) == 1

        ch = self.test.difference_plot("control", "test", groupby="country")
        assert len(ch.charts) == 1

    def test_multiple_difference(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference(("bad_value", "bad_value"), level_as_reference=False)

        diff = self.test.multiple_difference(("us", "control"), level_as_reference=False)
        assert len(diff) > 0

        diff = self.test.multiple_difference("test", groupby="country", level_as_reference=False)
        assert len(diff) > 0

    def test_multiple_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_plot(
                ("bad_value", "bad_value"), ("bad_value", "bad_value"), level_as_reference=False
            )

        ch = self.test.multiple_difference_plot(("us", "control"), level_as_reference=False)
        assert len(ch.charts) == 1

        ch = self.test.multiple_difference_plot("test", groupby="country", level_as_reference=False)
        assert len(ch.charts) == 1

    def test_achieved_power(self):
        power_df = self.test.achieved_power(level_1=("us", "test2"), level_2=("us", "test"), mde=0.2, alpha=0.05)
        power = power_df.achieved_power
        expected_power = 0.9409124  # Based on G*Power calculation

        assert np.allclose(power, expected_power, 0.005)


class TestOrdinal(object):
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
                    "test",
                    "control",
                    "test2",
                    "test",
                    "control",
                    "test2",
                ],
                "nr_of_items": [500, 8, 100, 510, 8, 100, 520, 9, 104, 530, 7, 100, 530, 8, 103],
                "nr_of_items_sumsq": [2500, 12, 150, 2510, 13, 140, 2520, 14, 154, 2530, 15, 160, 2530, 16, 103],
                "users": [1010, 22, 150, 1000, 20, 153, 1030, 23, 154, 1000, 20, 150, 1040, 21, 155],
                "days_since_reg": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            }
        )

        self.test = spotify_confidence.StudentsTTest(
            self.data,
            numerator_column="nr_of_items",
            numerator_sum_squares_column="nr_of_items_sumsq",
            denominator_column="users",
            categorical_group_columns="variation_name",
            ordinal_group_column="days_since_reg",
            interval_size=0.95,
        )

    def test_summary(self):
        summary = self.test.summary()

        assert np.array_equal(summary.point_estimate, self.data.nr_of_items / self.data.users)

        assert np.allclose(
            summary["ci_lower"],
            np.array(
                [
                    0.40279383,
                    0.07191692,
                    0.54600746,
                    0.41687347,
                    0.06387915,
                    0.54166658,
                    0.41429269,
                    0.09286814,
                    0.5575301,
                    0.43689002,
                    -0.03036827,
                    0.53897297,
                    0.4198777,
                    0.01463557,
                    0.58935338,
                ]
            ),
            rtol=1e-3,
        )

        assert np.allclose(
            summary["ci_upper"],
            np.array(
                [
                    0.58730518,
                    0.6553558,
                    0.78732587,
                    0.60312653,
                    0.73612085,
                    0.76552296,
                    0.59541605,
                    0.68974056,
                    0.79311926,
                    0.62310998,
                    0.73036827,
                    0.79436036,
                    0.59935306,
                    0.7472692,
                    0.73967887,
                ]
            ),
            rtol=1e-3,
        )

    def test_summary_plot(self):
        """Area plot tests"""
        ch = self.test.summary_plot()
        assert len(ch.charts) == 1

        ch = self.test.summary_plot().charts[0]
        np.array_equal(chart_data(ch, "control")["days_since_reg"], np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1]))

    def test_difference(self):
        with pytest.raises(ValueError):
            self.test.difference(("control", "us"), ("test", "usf"))

        diff = self.test.difference(("control", 1), ("test", 1))
        assert np.allclose(diff["difference"].iloc[0], 0.131413, rtol=1e-3)
        assert np.allclose(diff["ci_lower"].iloc[0], -0.172704, rtol=1e-3)
        assert np.allclose(diff["ci_upper"].iloc[0], 0.435531, rtol=1e-3)
        assert np.allclose(diff["p-value"].iloc[0], 0.382552, rtol=1e-3)

        diff = self.test.difference(("control", 1), ("test", 1), absolute=False)
        assert np.allclose(diff["difference"].iloc[0], 0.361386, rtol=1e-3)
        assert np.allclose(diff["ci_lower"].iloc[0], -0.474938, rtol=1e-3)
        assert np.allclose(diff["ci_upper"].iloc[0], 1.19771, rtol=1e-3)
        assert np.allclose(diff["p-value"].iloc[0], 0.382552, rtol=1e-3)

        diff = self.test.difference("control", "test", groupby="days_since_reg")
        assert np.allclose(diff["difference"], np.array([0.13141314, 0.11, 0.11355002, 0.18, 0.128663]), rtol=1e-3)

    def test_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.difference_plot(("control", 10), ("test", 10))

        ch = self.test.difference_plot(("control", 1), ("test", 1))
        assert len(ch.charts) == 1

        ch = self.test.difference_plot("control", "test", groupby="days_since_reg")
        assert len(ch.charts) == 1

    def test_multiple_difference(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference(
                ("bad_value", "bad_value"), ("bad_value", "bad_value"), level_as_reference=False
            )

        diff = self.test.multiple_difference(("control", 1), level_as_reference=False)
        assert len(diff) > 0

        diff = self.test.multiple_difference("test", groupby="days_since_reg", level_as_reference=False)
        assert len(diff) > 0

    def test_multiple_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_plot(
                ("bad_value", "bad_value"), ("bad_value", "bad_value"), level_as_reference=False
            )

        ch = self.test.multiple_difference_plot(("control", 1), level_as_reference=False)
        assert len(ch.charts) == 1

        ch = self.test.multiple_difference_plot("test", groupby="days_since_reg", level_as_reference=False)
        assert len(ch.charts) == 1

    def test_achieved_power(self):
        power_df = self.test.achieved_power(level_1=("control", 1), level_2=("test", 1), mde=1, alpha=0.05)
        power = power_df.achieved_power
        expected_power = 0.8790801  # Based on G*Power calculation
        assert np.allclose(power, expected_power, 0.005)


class TestOrdinalPlusTwoCategorical(object):
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
                    "test",
                    "control",
                    "test2",
                    "test",
                    "control",
                    "test2",
                    "test",
                    "control",
                    "test2",
                    "test",
                    "control",
                    "test2",
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
                "nr_of_items": [
                    500,
                    8,
                    100,
                    510,
                    8,
                    100,
                    520,
                    9,
                    104,
                    530,
                    7,
                    100,
                    530,
                    8,
                    103,
                    500,
                    8,
                    100,
                    510,
                    8,
                    100,
                    520,
                    9,
                    104,
                    530,
                    7,
                    100,
                    530,
                    8,
                    103,
                ],
                "nr_of_items_sumsq": [
                    1010,
                    32,
                    250,
                    1000,
                    30,
                    253,
                    1030,
                    33,
                    254,
                    1000,
                    30,
                    250,
                    1040,
                    31,
                    255,
                    1010,
                    22,
                    150,
                    1000,
                    20,
                    153,
                    1030,
                    23,
                    154,
                    1000,
                    20,
                    150,
                    1040,
                    21,
                    155,
                ],
                "users": [
                    2010,
                    42,
                    250,
                    2000,
                    40,
                    253,
                    2030,
                    43,
                    254,
                    2000,
                    40,
                    250,
                    2040,
                    41,
                    255,
                    1010,
                    22,
                    150,
                    1000,
                    20,
                    153,
                    1030,
                    23,
                    154,
                    1000,
                    20,
                    150,
                    1040,
                    21,
                    155,
                ],
                "days_since_reg": [
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                ],
                "country": [
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "us",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                    "gb",
                ],
            }
        )

        self.test = spotify_confidence.StudentsTTest(
            self.data,
            numerator_column="nr_of_items",
            numerator_sum_squares_column="nr_of_items_sumsq",
            denominator_column="users",
            categorical_group_columns=["variation_name", "country"],
            ordinal_group_column="days_since_reg",
        )

    def test_summary_plot(self):
        ch = self.test.summary_plot()
        assert len(ch.charts) == 1

        ch = self.test.summary_plot(groupby=["country"])
        assert len(ch.charts) == 2

        ch = self.test.summary_plot(groupby=["days_since_reg"])
        assert len(ch.charts) == 5

        ch = self.test.summary_plot(groupby=["country", "days_since_reg"])
        assert len(ch.charts) == 10

    def test_difference(self):
        with pytest.raises(ValueError):
            self.test.difference(("control", "us"), ("test", "usf"), groupby="days_since_reg")

        df = self.test.difference(level_1=("test", "us"), level_2=("control", "us"), groupby="days_since_reg")
        assert len(df) == 5
        assert "days_since_reg" in df.columns

        df = self.test.difference(level_1=("test", "us"), level_2=("control", "us"), groupby=["days_since_reg"])
        assert len(df) == 5
        assert "days_since_reg" in df.columns

        df = self.test.difference(level_1=("test", 1), level_2=("control", 1), groupby=["country"])
        assert len(df) == 2
        assert "country" in df.columns

    def test_differece_with_nims(self):
        df = self.test.difference(
            level_1=("test", "us"),
            level_2=("control", "us"),
            groupby="days_since_reg",
            non_inferiority_margins=(0.01, INCREASE_PREFFERED),
        )
        assert len(df) == 5
        assert "days_since_reg" in df.columns

        df = self.test.difference(
            level_1=("test", "us"),
            level_2=("control", "us"),
            groupby=["days_since_reg"],
            non_inferiority_margins=(0.01, DECREASE_PREFFERED),
        )
        assert len(df) == 5
        assert "days_since_reg" in df.columns

        df = self.test.difference(
            level_1=("test", 1),
            level_2=("control", 1),
            groupby=["country"],
            non_inferiority_margins={"us": (0.01, INCREASE_PREFFERED), "gb": (0.05, INCREASE_PREFFERED)},
        )
        assert len(df) == 2
        assert "country" in df.columns

        df = self.test.difference(
            level_1="test",
            level_2="control",
            groupby=["country", "days_since_reg"],
            non_inferiority_margins=(0.01, DECREASE_PREFFERED),
        )
        assert len(df) == 10
        assert "country" in df.columns
        assert "days_since_reg" in df.columns

        nims = {
            ("us", 1): (0.01, DECREASE_PREFFERED),
            ("us", 2): (0.1, INCREASE_PREFFERED),
            ("us", 3): (0.2, DECREASE_PREFFERED),
            ("us", 4): (0.5, INCREASE_PREFFERED),
            ("us", 5): (0.99, DECREASE_PREFFERED),
            ("gb", 1): (1.01, INCREASE_PREFFERED),
            ("gb", 2): (2.01, DECREASE_PREFFERED),
            ("gb", 3): (3.01, INCREASE_PREFFERED),
            ("gb", 4): (4.01, DECREASE_PREFFERED),
            ("gb", 5): (5.01, INCREASE_PREFFERED),
        }
        df = self.test.difference(
            level_1="test", level_2="control", groupby=["country", "days_since_reg"], non_inferiority_margins=nims
        )
        assert len(df) == 10
        assert "country" in df.columns
        assert "days_since_reg" in df.columns

        df = self.test.multiple_difference(
            level="control",
            level_as_reference=True,
            groupby=["country", "days_since_reg"],
            non_inferiority_margins=nims,
        )
        assert len(df) == 20

    def test_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.difference_plot(("control", "us", 10), ("test", "us", 10))

        ch = self.test.difference_plot(level_1=("control", "us", 1), level_2=("test", "us", 1))
        assert len(ch.charts) == 1

        ch = self.test.difference_plot(level_1=("control", "us"), level_2=("test", "us"), groupby="days_since_reg")
        assert len(ch.charts) == 1

        ch = self.test.difference_plot(level_1="control", level_2="test", groupby=["country", "days_since_reg"])
        assert len(ch.charts) == 1

        ch = self.test.difference_plot(level_1=("control", 1), level_2=("test", 1), groupby="country")
        assert len(ch.charts) == 1

        ch = self.test.difference_plot(level_1=("control", "us"), level_2=("test", "us"), groupby="days_since_reg")
        assert len(ch.charts) == 1

        ch = self.test.difference_plot(level_1="control", level_2="test", groupby=["country", "days_since_reg"])
        assert len(ch.charts) == 1

    def test_multiple_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_plot(level="control")

        ch = self.test.multiple_difference_plot(
            level="control", groupby=["country", "days_since_reg"], level_as_reference=True
        )
        assert len(ch.charts) == 1

        ch = self.test.multiple_difference_plot(
            level="us", groupby=["variation_name", "days_since_reg"], level_as_reference=True
        )
        assert len(ch.charts) == 1

        ch = self.test.multiple_difference_plot(
            level=1, groupby=["country", "variation_name"], level_as_reference=True
        )
        assert len(ch.charts) == 1

    def test_differece_plot_with_nims(self):
        ch = self.test.difference_plot(
            level_1=("test", "us"),
            level_2=("control", "us"),
            groupby="days_since_reg",
            non_inferiority_margins=(0.01, INCREASE_PREFFERED),
        )
        assert len(ch.charts) == 1

        ch = self.test.difference_plot(
            level_1=("test", "us"),
            level_2=("control", "us"),
            groupby=["days_since_reg"],
            non_inferiority_margins=(0.01, DECREASE_PREFFERED),
        )
        assert len(ch.charts) == 1

        ch = self.test.difference_plot(
            level_1=("test", 1),
            level_2=("control", 1),
            groupby=["country"],
            non_inferiority_margins={"us": (0.01, INCREASE_PREFFERED), "gb": (0.05, INCREASE_PREFFERED)},
        )
        assert len(ch.charts) == 1

        ch = self.test.difference_plot(
            level_1="test",
            level_2="control",
            groupby=["country", "days_since_reg"],
            non_inferiority_margins=(0.01, DECREASE_PREFFERED),
        )
        assert len(ch.charts) == 1

        nims = {
            ("us", 1): (0.01, DECREASE_PREFFERED),
            ("us", 2): (0.1, INCREASE_PREFFERED),
            ("us", 3): (0.2, DECREASE_PREFFERED),
            ("us", 4): (0.5, INCREASE_PREFFERED),
            ("us", 5): (0.99, DECREASE_PREFFERED),
            ("gb", 1): (1.01, INCREASE_PREFFERED),
            ("gb", 2): (2.01, DECREASE_PREFFERED),
            ("gb", 3): (3.01, INCREASE_PREFFERED),
            ("gb", 4): (4.01, DECREASE_PREFFERED),
            ("gb", 5): (5.01, INCREASE_PREFFERED),
        }
        ch = self.test.difference_plot(
            level_1="test", level_2="control", groupby=["country", "days_since_reg"], non_inferiority_margins=nims
        )
        assert len(ch.charts) == 1

        ch = self.test.multiple_difference_plot(
            level="control",
            level_as_reference=True,
            groupby=["country", "days_since_reg"],
            non_inferiority_margins=nims,
        )
        assert len(ch.charts) == 1


class TestCategoricalBinomialData(object):
    def setup(self):
        np.random.seed(123)

        self.data = pd.DataFrame(
            {
                "variation_name": ["test", "control", "test2", "test", "control", "test2"],
                "success": [500, 42, 1005, 50, 4, 100],
                "total": [1009, 104, 1502, 100, 10, 150],
                "country": [
                    "us",
                    "us",
                    "us",
                    "gb",
                    "gb",
                    "gb",
                ],
            }
        )

        self.test = spotify_confidence.StudentsTTest(
            self.data,
            numerator_column="success",
            numerator_sum_squares_column="success",
            denominator_column="total",
            categorical_group_columns=["country", "variation_name"],
        )

    def test_summary(self):
        """Area plot tests"""

        summary = self.test.summary()
        assert np.array_equal(summary.country, np.array(["us", "us", "us", "gb", "gb", "gb"]))
        assert np.array_equal(summary.point_estimate, self.data.success / self.data.total)
        assert np.allclose(
            summary["ci_lower"], np.array([0.464653, 0.308424, 0.645293, 0.400789, 0.049548, 0.590610]), rtol=1e-3
        )
        assert np.allclose(
            summary["ci_upper"], np.array([0.526427, 0.499269, 0.692923, 0.599211, 0.750452, 0.742723]), rtol=1e-3
        )

    def test_multiple_difference(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference(("bad_value", "bad_value"), level_as_reference=False)

        diff = self.test.multiple_difference(("us", "control"), level_as_reference=False)
        assert np.allclose(
            diff["adjusted p-value"],
            np.array([1e00, 8.36850129e-01, 1.49172324e-04, 3.62369184e-01, 2.27154393e-06]),
            rtol=1e-3,
        )
        assert np.allclose(
            diff["p-value"],
            np.array([9.81516149e-01, 1.67370026e-01, 2.98344648e-05, 7.24738369e-02, 4.54308787e-07]),
            rtol=1e-3,
        )
        assert np.allclose(
            diff["adjusted ci_lower"], np.array([-0.501747, -0.276600, -0.422946, -0.224093, -0.395207]), rtol=1e-3
        )
        assert np.allclose(
            diff["adjusted ci_upper"], np.array([0.509440, 0.084292, -0.102695, 0.040705, -0.135317]), rtol=1e-3
        )

        diff = self.test.multiple_difference("test", groupby="country", level_as_reference=False)
        assert np.allclose(diff["adjusted p-value"], np.array([1, 0.035594, 0.289895, 0.0]), rtol=1e-3)
        assert np.allclose(diff["p-value"], np.array([0.55155672, 0.00889848, 0.07247384, 0.0]), rtol=1e-3)
        assert np.allclose(
            diff["adjusted ci_lower"], np.array([-0.385571, -0.325682, -0.036587, -0.223262]), rtol=1e-3
        )
        assert np.allclose(diff["adjusted ci_upper"], np.array([0.585571, -0.0076513, 0.219975, -0.123874]), rtol=1e-3)


class TestWithNims(object):
    def setup(self):
        self.data = pd.DataFrame(
            [
                {
                    "group": "1",
                    "count": 5000,
                    "sum": 10021.0,
                    "sum_of_squares": 25142.0,
                    "avg": 2.004210,
                    "var": 1.0116668,
                },
                {
                    "group": "2",
                    "count": 5000,
                    "sum": 9892.0,
                    "sum_of_squares": 24510.0,
                    "avg": 1.978424,
                    "var": 0.9881132,
                },
            ]
        )

        self.test = spotify_confidence.StudentsTTest(
            self.data,
            numerator_column="sum",
            numerator_sum_squares_column="sum_of_squares",
            denominator_column="count",
            categorical_group_columns="group",
            interval_size=0.99,
        )

    def test_compare_series_non_inferiority_improve_postitive(self):
        summary = self.test.summary()
        control_avg = self.data.query("group == '1'").avg.values[0]
        assert np.allclose(control_avg, summary.query("group == '1'")[POINT_ESTIMATE])

        diff = self.test.difference(level_1="1", level_2="2", non_inferiority_margins=(0.02, "increase"))

        np.testing.assert_almost_equal(diff[DIFFERENCE].values[0], -0.0258, 3)
        assert diff[CI_UPPER].values[0] == float("inf")
        np.testing.assert_almost_equal(diff[CI_LOWER].values[0], -0.0723, 3)
        assert diff[P_VALUE].values[0] > 0.01

    def test_compare_series_non_inferiority_improve_negative(self):
        summary = self.test.summary()
        control_avg = self.data.query("group == '1'").avg.values[0]
        assert np.allclose(control_avg, summary.query("group == '1'")[POINT_ESTIMATE])

        diff = self.test.difference(level_1="1", level_2="2", non_inferiority_margins=(0.02, "decrease"))

        np.testing.assert_almost_equal(diff[DIFFERENCE].values[0], -0.0258, 3)
        assert diff[CI_LOWER].values[0] == -float("inf")
        np.testing.assert_almost_equal(diff[CI_UPPER].values[0], 0.0207, 3)
        assert diff[P_VALUE].values[0] < 0.01

    def test_one_sided_ztest_positive(self):
        summary = self.test.summary()
        control_avg = self.data.query("group == '1'").avg.values[0]
        assert np.allclose(control_avg, summary.query("group == '1'")[POINT_ESTIMATE])

        diff = self.test.difference(level_1="1", level_2="2", non_inferiority_margins=(None, "increase"))

        np.testing.assert_almost_equal(diff[DIFFERENCE].values[0], -0.0258, 3)
        assert diff[CI_UPPER].values[0] == float("inf")
        np.testing.assert_almost_equal(diff[CI_LOWER].values[0], -0.0723, 3)
        assert diff[P_VALUE].values[0] > 0.01

    def test_one_sided_ztest_negative(self):
        summary = self.test.summary()
        control_avg = self.data.query("group == '1'").avg.values[0]
        assert np.allclose(control_avg, summary.query("group == '1'")[POINT_ESTIMATE])

        diff = self.test.difference(level_1="1", level_2="2", non_inferiority_margins=(None, "decrease"))

        np.testing.assert_almost_equal(diff[DIFFERENCE].values[0], -0.0258, 3)
        assert diff[CI_LOWER].values[0] == -float("inf")
        np.testing.assert_almost_equal(diff[CI_UPPER].values[0], 0.0207, 3)
        assert diff[P_VALUE].values[0] > 0.01
