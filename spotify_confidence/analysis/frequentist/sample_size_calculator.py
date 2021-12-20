from typing import Union, Iterable, Tuple

from pandas import DataFrame

from spotify_confidence.analysis.frequentist.confidence_computers.generic_computer import GenericComputer
from ..abstract_base_classes.confidence_computer_abc import ConfidenceComputerABC
from ..confidence_utils import (
    listify,
)
from ..constants import BONFERRONI, ZTEST, METHOD_COLUMN_NAME


class SampleSizeCalculator:
    def __init__(
        self,
        data_frame: DataFrame,
        point_estimate_column: str,
        var_column: str,
        is_binary_column: str,
        categorical_group_columns: Union[None, str, Iterable] = None,
        interval_size: float = 0.95,
        correction_method: str = BONFERRONI,
        confidence_computer: ConfidenceComputerABC = None,
        metric_column=None,
        power: float = 0.8,
    ):
        if confidence_computer is not None:
            self._confidence_computer = confidence_computer
        else:
            self._confidence_computer = GenericComputer(
                data_frame=data_frame.assign(**{METHOD_COLUMN_NAME: ZTEST}),
                numerator_column=None,
                numerator_sum_squares_column=None,
                denominator_column=None,
                categorical_group_columns=listify(categorical_group_columns),
                ordinal_group_column=None,
                interval_size=interval_size,
                correction_method=correction_method.lower(),
                method_column=METHOD_COLUMN_NAME,
                bootstrap_samples_column=None,
                metric_column=metric_column,
                treatment_column=None,
                power=power,
                point_estimate_column=point_estimate_column,
                var_column=var_column,
                is_binary_column=is_binary_column,
                feature_column=None,
                feature_sum_squares_column=None,
                feature_cross_sum_column=None,
            )

    def sample_size(
        self,
        treatment_weights: Iterable,
        mde_column: str,
        nim_column: str,
        preferred_direction_column: str,
        final_expected_sample_size_column: str = None,
    ) -> DataFrame:
        """Args:
            treatment_weights (Iterable): The first weight is treated as control, the rest as treatment groups.
            mde_column (str): Name of column in source dataframe containing the minimum detectable effect sizes
            nim_column (str): Name of column in source dataframe containing the non-inferiority margins.
            preferred_direction_column (str): Name of column in source dataframe containing the preferred direction
            of the metric, which can be "increase", "decrease" or None, the latter meaning a two sided test will be
            performed.
            final_expected_sample_size_column (str): Name of column in source dataframe containing an expected
            sample size. If this is given, a confidence interval width around the avg will be returned in the
            "ci_width" column
        Returns:
            Dataframe containing a column "required_sample_size_metric" containing the sample size ot the control vs
            treatment comparison that requires the largest sample size.

        One of mde or nim has to be set for each row of the dataframe. If mde is set, the null hypothesis is no
            difference between control and treatment, and the sample size returned will be enough to detect a relative
            difference of mde or and absolute difference of mde*avg. If nim is set the null hypothesis is -nim*avg
            if preferred_direction is "increase" and +nim*avg if preferred direction is "decrease"
        """
        return self._confidence_computer.compute_sample_size(
            treatment_weights, mde_column, nim_column, preferred_direction_column, final_expected_sample_size_column
        )

    def optimal_weights_and_sample_size(
        self, sample_size_df: DataFrame, number_of_groups: int
    ) -> Tuple[Iterable, int]:
        """Args:
            sample_size_df (DataFrame): A data frame returned by the sample_size method of this class
            number_of_groups (int): Number of groups in the experiment, including control
        Returns:
            Tuple (list, int), where the list contains optimal weights and the int is the sample size that would be
            required if those weights were used
        """
        return self._confidence_computer.compute_optimal_weights_and_sample_size(sample_size_df, number_of_groups)

    def powered_effect(
        self,
        treatment_weights: Iterable,
        mde_column: str,
        nim_column: str,
        preferred_direction_column: str,
        sample_size: int,
    ) -> DataFrame:
        """Args:
            treatment_weights (Iterable): The first weight is treated as control, the rest as treatment groups.
            mde_column (str): Name of column in source dataframe containing the minimum detectable effect sizes
            nim_column (str): Name of column in source dataframe containing the non-inferiority margins.
            preferred_direction_column (str): Name of column in source dataframe containing the preferred direction
            of the metric, which can be "increase", "decrease" or None, the latter meaning a two sided test will be
            performed.
            sample_size (int): Total sample size across all groups to base the powered effect calculation on
        Returns:
            Dataframe containing a column "powered_effect" containing the powered effect of the control vs
            treatment comparison that requires the largest powered effect.

        Mde or nims are only needed for some multiple correction methods and are there to give results
        consistent with the sample size calculations, i.e. if you first calculate a sample size for a
        specific MDE and then calculate the powered effect for that sample size, the original MDE will be returned.
        """
        return self._confidence_computer.compute_powered_effect(
            treatment_weights, mde_column, nim_column, preferred_direction_column, sample_size
        )
