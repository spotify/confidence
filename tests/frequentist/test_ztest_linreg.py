import numpy as np
import pandas as pd

import spotify_confidence


class Test(object):
    def setup(self):
        np.random.seed(123)
        n = 10000
        d = np.random.randint(2, size=n)
        x = np.random.standard_normal(size=n)
        y = 0.5 * d + 0.5 * x + np.random.standard_normal(size=n)
        country = np.random.choice(["us", "se"], n)
        data = pd.DataFrame({"variation_name": list(map(str, d)), "y": y, "x": x, "country": country})
        data = (
            data.assign(xy=y * x)
            .assign(x2=x ** 2)
            .assign(y2=y ** 2)
            .groupby(["variation_name", "country"])
            .agg({"y": ["sum", "count"], "y2": "sum", "x": "sum", "x2": "sum", "xy": "sum"})
            .reset_index()
        )

        data.columns = data.columns.map("_".join).str.strip("_")
        data.assign(**{"metric_name": "metricA"})
        self.data = data


        self.test = spotify_confidence.ZTestLinreg(
            data_frame=data,
            numerator_column="y_sum",
            numerator_sum_squares_column="y2_sum",
            denominator_column="y_count",
            categorical_group_columns=["variation_name", "country"],
            feature_column="x_sum",
            feature_sum_squares_column="x2_sum",
            feature_cross_sum_column="xy_sum",
            interval_size=0.99,
            correction_method="bonferroni",
        )

    def test_summary(self):
        summary_df = self.test.summary(verbose=True)
        print(summary_df)
        assert len(summary_df) == len(self.data)
