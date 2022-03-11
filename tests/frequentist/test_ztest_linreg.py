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
            metric_column="metric_name",
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

        diff = self.test.difference(level_1="0", level_2="1", verbose=True, groupby="metric_name")
        assert np.allclose(b1[1], diff["difference"])

        v0 = np.var(y_adj[self.d == 0])
        v1 = np.var(y_adj[self.d == 1])
        n0 = y_adj[self.d == 0].size
        n1 = y_adj[self.d == 1].size
        assert np.allclose(diff["std_err"], np.sqrt(v0 / n0 + v1 / n1), rtol=1e-3)


class TestUnivariateMultiMetric(object):
    def setup(self):
        np.random.seed(123)
        n = 20000
        d = np.random.randint(2, size=n)
        x = np.random.standard_normal(size=n)
        y = 0.5 * d + 0.5 * x + np.random.standard_normal(size=n)
        m = np.repeat(["metricA", "metricB"], int(n / 2))
        data = pd.DataFrame({"variation_name": list(map(str, d)), "metric_name": m, "y": y, "x": x})
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
            metric_column="metric_name",
        )

    def test_summary(self):
        summary_df = self.test.summary(verbose=True)
        print(summary_df)
        assert len(summary_df) == len(self.data)

    def test_parameters_univariate(self):
        def linreg(X, y):
            return np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))

        summary_df = self.test.summary(verbose=True)
        N0 = self.m[self.m == "metricA"].size
        N1 = self.n - N0
        X = np.ones((N0, 2))
        X[:, 1] = self.x[self.m == "metricA"]
        b0 = linreg(X, self.y[self.m == "metricA"])
        y_adj = self.y[self.m == "metricA"] - np.matmul(X, b0)
        X[:, 1] = self.d[self.m == "metricA"]
        b1 = linreg(X, y_adj)
        assert np.allclose(b0[1], summary_df[REGRESSION_PARAM][0], rtol=0.0001)

        diff = self.test.difference(level_1=("0", "metricA"), level_2=("1", "metricA"), verbose=True)
        assert np.allclose(b1[1], diff["difference"])

        idx = self.d[self.m == "metricA"]
        v0 = np.var(y_adj[idx == 0])
        v1 = np.var(y_adj[idx == 1])
        n0 = y_adj[idx == 0].size
        n1 = y_adj[idx == 1].size
        assert np.allclose(diff["std_err"], np.sqrt(v0 / n0 + v1 / n1), rtol=1e-3)

        X = np.ones((N1, 2))
        X[:, 1] = self.x[self.m == "metricB"]
        b0 = linreg(X, self.y[self.m == "metricB"])
        y_adj = self.y[self.m == "metricB"] - np.matmul(X, b0)
        X[:, 1] = self.d[self.m == "metricB"]
        b1 = linreg(X, y_adj)
        assert np.allclose(b0[1], summary_df[REGRESSION_PARAM][1], rtol=0.0001)

        diff = self.test.difference(level_1=("0", "metricB"), level_2=("1", "metricB"), verbose=True)
        assert np.allclose(b1[1], diff["difference"])

        idx = self.d[self.m == "metricB"]
        v0 = np.var(y_adj[idx == 0])
        v1 = np.var(y_adj[idx == 1])
        n0 = y_adj[idx == 0].size
        n1 = y_adj[idx == 1].size
        assert np.allclose(diff["std_err"], np.sqrt(v0 / n0 + v1 / n1), rtol=1e-3)


class TestUnivariateNoFeatures(object):
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

        self.ztest = spotify_confidence.ZTest(
            self.data,
            numerator_column="sum",
            numerator_sum_squares_column="sum_of_squares",
            denominator_column="count",
            categorical_group_columns="group",
            interval_size=0.99,
        )

        self.ztestlinreg = spotify_confidence.ZTestLinreg(
            data_frame=self.data,
            numerator_column="sum",
            numerator_sum_squares_column="sum_of_squares",
            denominator_column="count",
            categorical_group_columns="group",
            feature_column=None,
            feature_sum_squares_column=None,
            feature_cross_sum_column=None,
            interval_size=0.99,
        )

    def test_summary(self):
        summary_ztest = self.ztest.summary(verbose=True).drop(["_method"], axis=1)
        summary_ztestlinreg = self.ztestlinreg.summary(verbose=True).drop(
            ["_method", "original_variance", "original_point_estimate"], axis=1
        )
        pd.testing.assert_frame_equal(summary_ztest, summary_ztestlinreg)


class TestMultivariateSingleMetric(object):
    def setup(self):
        np.random.seed(123)

        n = 10000
        d = np.random.randint(2, size=n)
        x1 = np.random.standard_normal(size=n)
        x2 = np.random.standard_normal(size=n)
        y = 0.5 * d + 0.5 * x1 + 0.5 * x2 + np.random.standard_normal(size=n)
        df = pd.DataFrame({"variation_name": list(map(str, d)), "y": y, "x1": x1, "x2": x2})

        data = df.assign(y2=y ** 2).groupby(["variation_name"]).agg({"y": ["sum", "count"], "y2": "sum"}).reset_index()

        data.columns = data.columns.map("_".join).str.strip("_")

        def _to_XX(data, features):
            X = data[features].to_numpy()
            XX = np.matmul(np.transpose(X), X)
            return XX

        def _to_Xy(data, features, metric):
            X = data[features].to_numpy()
            y = data[metric].to_numpy()
            Xy = np.matmul(np.transpose(X), y)
            return Xy

        def _to_Xs(data, features):
            X = data[features].to_numpy()
            Xs = np.sum(X, axis=0)
            return Xs

        XX = (
            df.groupby(["variation_name"])
            .apply(lambda x: pd.Series({"out": _to_XX(x, features=["x1", "x2"])}))
            .to_records()["out"]
        )
        Xy = (
            df.groupby(["variation_name"])
            .apply(lambda x: pd.Series({"out": _to_Xy(x, features=["x1", "x2"], metric="y")}))
            .to_records()["out"]
        )
        Xs = (
            df.groupby(["variation_name"])
            .apply(lambda x: pd.Series({"out": _to_Xs(x, features=["x1", "x2"])}))
            .to_records()["out"]
        )

        data["XX"] = XX
        data["Xy"] = Xy
        data["Xs"] = Xs
        data = data.assign(**{"metric_name": "metricA"})
        self.n = n
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.d = d
        self.data = data

        self.test = spotify_confidence.ZTestLinreg(
            data_frame=data,
            numerator_column="y_sum",
            numerator_sum_squares_column="y2_sum",
            denominator_column="y_count",
            categorical_group_columns=["variation_name"],
            feature_column="Xs",
            feature_sum_squares_column="XX",
            feature_cross_sum_column="Xy",
            interval_size=0.99,
            correction_method="bonferroni",
            metric_column="metric_name",
        )

    def test_summary(self):
        summary_df = self.test.summary(verbose=True)
        print(summary_df)
        assert len(summary_df) == len(self.data)

    def test_parameters_univariate(self):
        def linreg(X, y):
            return np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))

        summary_df = self.test.summary(verbose=True)
        X = np.ones((self.n, 3))
        X[:, 1] = self.x1
        X[:, 2] = self.x2
        b0 = linreg(X, self.y)
        y_adj = self.y - np.matmul(X, b0)
        X = np.ones((self.n, 2))
        X[:, 1] = self.d
        b1 = linreg(X, y_adj)
        assert np.allclose(b0[1:], summary_df[REGRESSION_PARAM][0].reshape(1, 2), rtol=0.0001)

        diff = self.test.difference(level_1="0", level_2="1", groupby="metric_name", verbose=True)
        assert np.allclose(b1[1], diff["difference"])

        v0 = np.var(y_adj[self.d == 0])
        v1 = np.var(y_adj[self.d == 1])
        n0 = y_adj[self.d == 0].size
        n1 = y_adj[self.d == 1].size
        assert np.allclose(diff["std_err"], np.sqrt(v0 / n0 + v1 / n1), rtol=1e-3)


class TestMultivariateMultipleMetrics(object):
    def setup(self):
        np.random.seed(123)

        n = 10000
        d = np.random.randint(2, size=n)
        x1 = np.random.standard_normal(size=n)
        x2 = np.random.standard_normal(size=n)
        y = 0.5 * d + 0.5 * x1 + 0.5 * x2 + np.random.standard_normal(size=n)
        metric_name = np.random.choice(["metricA", "metricB"], n)
        df = pd.DataFrame(
            {"variation_name": list(map(str, d)), "y": y, "x1": x1, "x2": x2, "metric_name": metric_name}
        )

        data = (
            df.assign(y2=y ** 2)
            .groupby(["variation_name", "metric_name"])
            .agg({"y": ["sum", "count"], "y2": "sum"})
            .reset_index()
        )

        data.columns = data.columns.map("_".join).str.strip("_")

        def _to_XX(data, features):
            X = data[features].to_numpy()
            XX = np.matmul(np.transpose(X), X)
            return XX

        def _to_Xy(data, features, metric):
            X = data[features].to_numpy()
            y = data[metric].to_numpy()
            Xy = np.matmul(np.transpose(X), y)
            return Xy

        def _to_Xs(data, features):
            X = data[features].to_numpy()
            Xs = np.sum(X, axis=0)
            return Xs

        XX = (
            df.groupby(["variation_name", "metric_name"])
            .apply(lambda x: pd.Series({"out": _to_XX(x, features=["x1", "x2"])}))
            .to_records()["out"]
        )
        Xy = (
            df.groupby(["variation_name", "metric_name"])
            .apply(lambda x: pd.Series({"out": _to_Xy(x, features=["x1", "x2"], metric="y")}))
            .to_records()["out"]
        )
        Xs = (
            df.groupby(["variation_name", "metric_name"])
            .apply(lambda x: pd.Series({"out": _to_Xs(x, features=["x1", "x2"])}))
            .to_records()["out"]
        )

        data["XX"] = XX
        data["Xy"] = Xy
        data["Xs"] = Xs

        self.n = n
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.d = d
        self.data = data
        self.m = metric_name

        self.test = spotify_confidence.ZTestLinreg(
            data_frame=data,
            numerator_column="y_sum",
            numerator_sum_squares_column="y2_sum",
            denominator_column="y_count",
            categorical_group_columns=["variation_name"],
            feature_column="Xs",
            feature_sum_squares_column="XX",
            feature_cross_sum_column="Xy",
            interval_size=0.99,
            correction_method="bonferroni",
            metric_column="metric_name",
        )

    def test_summary(self):
        summary_df = self.test.summary(verbose=True)
        print(summary_df)
        assert len(summary_df) == len(self.data)

    def test_parameters_multivariate(self):
        def linreg(X, y):
            return np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))

        summary_df = self.test.summary(verbose=True)
        N0 = self.m[self.m == "metricA"].size
        # N1 = self.n - N0
        X = np.ones((N0, 3))
        X[:, 1] = self.x1[self.m == "metricA"]
        X[:, 2] = self.x2[self.m == "metricA"]
        b0 = linreg(X, self.y[self.m == "metricA"])
        y_adj = self.y[self.m == "metricA"] - np.matmul(X, b0)
        X = np.ones((N0, 2))
        X[:, 1] = self.d[self.m == "metricA"]
        b1 = linreg(X, y_adj)
        assert np.allclose(b0[1], summary_df[REGRESSION_PARAM][0][0], rtol=0.0001)
        assert np.allclose(b0[2], summary_df[REGRESSION_PARAM][0][1], rtol=0.0001)

        diff = self.test.difference(level_1=("0", "metricA"), level_2=("1", "metricA"), verbose=True)
        assert np.allclose(b1[1], diff["difference"])

        summary_df = self.test.summary(verbose=True)
        N0 = self.m[self.m == "metricB"].size
        # N1 = self.n - N0
        X = np.ones((N0, 3))
        X[:, 1] = self.x1[self.m == "metricB"]
        X[:, 2] = self.x2[self.m == "metricB"]
        b0 = linreg(X, self.y[self.m == "metricB"])
        y_adj = self.y[self.m == "metricB"] - np.matmul(X, b0)
        X = np.ones((N0, 2))
        X[:, 1] = self.d[self.m == "metricB"]
        b1 = linreg(X, y_adj)
        assert np.allclose(b0[1], summary_df[REGRESSION_PARAM][1][0], rtol=0.0001)
        assert np.allclose(b0[2], summary_df[REGRESSION_PARAM][1][1], rtol=0.0001)

        diff = self.test.difference(level_1=("0", "metricB"), level_2=("1", "metricB"), verbose=True)
        assert np.allclose(b1[1], diff["difference"])


# TODO: Test for sequential data (w/ ordinal column)
# TODO: Test for segmentation
