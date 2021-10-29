import numpy as np
import pandas as pd

import spotify_confidence
from spotify_confidence.analysis.constants import REGRESSION_PARAM


class TestUnivariateSingleMetric(object):
    def setup(self):
        np.random.seed(123)
        n = 10000
        d = np.random.randint(2, size=n)
        x = np.random.standard_normal(size=n)
        y = 0.5 * d + 0.5 * x + np.random.standard_normal(size=n)
        data = pd.DataFrame({"variation_name": list(map(str, d)), "y": y, "x": x})
        data = (
            data.assign(xy=y * x)
            .assign(x2=x ** 2)
            .assign(y2=y ** 2)
            .groupby(["variation_name"])
            .agg({"y": ["sum", "count"], "y2": "sum", "x": "sum", "x2": "sum", "xy": "sum"})
            .reset_index()
        )

        data.columns = data.columns.map("_".join).str.strip("_")
        data = data.assign(**{"metric_name": "metricA"})
        self.n = n
        self.x = x
        self.y = y
        self.d = d
        self.data = data


        self.test = spotify_confidence.ZTestLinreg(
            data_frame=data,
            numerator_column="y_sum",
            numerator_sum_squares_column="y2_sum",
            denominator_column="y_count",
            categorical_group_columns=["variation_name"],
            feature_column="x_sum",
            feature_sum_squares_column="x2_sum",
            feature_cross_sum_column="xy_sum",
            interval_size=0.99,
            correction_method="bonferroni",
            metric_column='metric_name'
        )

    def test_summary(self):
        summary_df = self.test.summary(verbose=True)
        print(summary_df)
        assert len(summary_df) == len(self.data)

    def test_parameters_univariate(self):
        def linreg(X, y):
            return np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))
        summary_df = self.test.summary(verbose=True)
        X = np.ones((self.n, 2))
        X[:, 1] = self.x
        b0 = linreg(X, self.y)
        y_adj = self.y - np.matmul(X, b0)
        X[:, 1] = self.d
        b1 = linreg(X, y_adj)
        assert np.allclose(b0[1], summary_df[REGRESSION_PARAM][0], rtol=0.0001)

        diff = self.test.difference(level_1='0', level_2='1')
        assert np.allclose(b1[1], diff['difference'])

class TestUnivariateMultiMetric(object):
    def setup(self):
        np.random.seed(123)
        n = 20000
        d = np.random.randint(2, size=n)
        x = np.random.standard_normal(size=n)
        y = 0.5 * d + 0.5 * x + np.random.standard_normal(size=n)
        m = np.random.randint(2, size=n)
        data = pd.DataFrame({"variation_name": list(map(str, d)),
                             "metric_name": list(map(str, m)),
                             "y": y, "x": x})
        data = (
            data.assign(xy=y * x)
                .assign(x2=x ** 2)
                .assign(y2=y ** 2)
                .groupby(["variation_name", "metric_name"])
                .agg({"y": ["sum", "count"], "y2": "sum", "x": "sum", "x2": "sum", "xy": "sum"})
                .reset_index()
        )

        data.columns = data.columns.map("_".join).str.strip("_")

        self.n = n
        self.x = x
        self.y = y
        self.d = d
        self.m = m
        self.data = data

        self.test = spotify_confidence.ZTestLinreg(
            data_frame=data,
            numerator_column="y_sum",
            numerator_sum_squares_column="y2_sum",
            denominator_column="y_count",
            categorical_group_columns=["variation_name"],
            feature_column="x_sum",
            feature_sum_squares_column="x2_sum",
            feature_cross_sum_column="xy_sum",
            interval_size=0.99,
            correction_method="bonferroni",
            metric_column='metric_name'
        )

    def test_summary(self):
        summary_df = self.test.summary(verbose=True)
        print(summary_df)
        assert len(summary_df) == len(self.data)

    def test_parameters_univariate(self):
        def linreg(X, y):
            return np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))

        # TODO: Do this metric by metric
        summary_df = self.test.summary(verbose=True)
        n0 = self.m[self.m == 0].size
        n1 = self.n - n0
        X = np.ones((n0, 2))
        X[:, 1] = self.x[self.m == 0]
        b0 = linreg(X, self.y[self.m == 0])
        y_adj = self.y[self.m == 0] - np.matmul(X, b0)
        X[:, 1] = self.d[self.m == 0]
        b1 = linreg(X, y_adj)
        assert np.allclose(b0[1], summary_df[REGRESSION_PARAM][0], rtol=0.0001)

        diff = self.test.difference(level_1=('0', '0'), level_2=('1', '0'))
        assert np.allclose(b1[1], diff['difference'])
