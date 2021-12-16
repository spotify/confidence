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
            )

    def sample_size(
        self,
        treatment_weights: Iterable,
        mde_column: str,
        nim_column: str,
        preferred_direction_column: str,
        final_expected_sample_size_column: str = None,
    ) -> DataFrame:
        return self._confidence_computer.compute_sample_size(
            treatment_weights, mde_column, nim_column, preferred_direction_column, final_expected_sample_size_column
        )

    def optimal_weights_and_sample_size(
        self, sample_size_df: DataFrame, number_of_groups: int
    ) -> Tuple[Iterable, int]:
        return self._confidence_computer.compute_optimal_weights_and_sample_size(sample_size_df, number_of_groups)
