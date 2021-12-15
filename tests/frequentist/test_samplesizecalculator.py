import pandas as pd
import numpy as np
from spotify_confidence.analysis.frequentist.sample_size_calculator import SampleSizeCalculator
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
    POWERED_EFFECT,
    REQUIRED_SAMPLE_SIZE,
    REQUIRED_SAMPLE_SIZE_METRIC,
    CI_WIDTH,
)


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
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC][0] / 1042868 < 1.001
        assert 0.999 < ss[REQUIRED_SAMPLE_SIZE_METRIC][1] / 95459 < 1.001
        assert ss[CI_WIDTH].isna().all()

        optimal_weights, optimal_sample_size = ssc.optimal_weights_and_sample_size(ss, len(treatment_weights))
        assert len(optimal_weights) == len(treatment_weights)
        assert 0.999 < optimal_sample_size / 894863 < 1.001
