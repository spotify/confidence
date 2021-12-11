from typing import Union, Iterable, Tuple, Dict, List
from pandas import DataFrame
from spotify_confidence.analysis.frequentist.confidence_computers.generic_computer import GenericComputer
from ..abstract_base_classes.confidence_computer_abc import ConfidenceComputerABC
from ..confidence_utils import (
    validate_categorical_columns,
    listify,
    get_all_categorical_group_columns,
    get_all_group_columns,
)
from ..constants import BONFERRONI, NIM_TYPE, METHODS


class SampleSizeCalculator:
    def __init__(
        self,
        data_frame: DataFrame,
        avg_column: str,
        var_column: str,
        categorical_group_columns: Union[str, Iterable],
        ordinal_group_column: Union[str, None] = None,
        interval_size: float = 0.95,
        correction_method: str = BONFERRONI,
        confidence_computer: ConfidenceComputerABC = None,
        method_column: str = None,
        metric_column=None,
        treatment_column=None,
        power: float = 0.8,
    ):
        validate_categorical_columns(categorical_group_columns)
        self._df = data_frame
        self._avg_column = avg_column
        self._var_column = var_column
        self._categorical_group_columns = get_all_categorical_group_columns(
            categorical_group_columns, metric_column, treatment_column
        )
        self._ordinal_group_column = ordinal_group_column
        self._metric_column = metric_column
        self._treatment_column = treatment_column
        self._all_group_columns = get_all_group_columns(self._categorical_group_columns, self._ordinal_group_column)
        if method_column is None:
            raise ValueError("method column cannot be None")
        if not all(self._df[method_column].map(lambda m: m in METHODS)):
            raise ValueError(f"Values of method column must be in {METHODS}")

        if confidence_computer is not None:
            self._confidence_computer = confidence_computer
        else:
            self._confidence_computer = GenericComputer(
                data_frame=data_frame,
                numerator_column=None,
                numerator_sum_squares_column=None,
                denominator_column=None,
                categorical_group_columns=listify(categorical_group_columns),
                ordinal_group_column=ordinal_group_column,
                interval_size=interval_size,
                correction_method=correction_method.lower(),
                method_column=method_column,
                bootstrap_samples_column=None,
                metric_column=metric_column,
                treatment_column=treatment_column,
                power=power,
            )

    def sample_size(
        self,
        treatment_weights: Iterable,
        mde_column: str,
        nim_column: str,
        preferred_direction_column: str,
        final_expected_sample_size_column: str,
    ) -> DataFrame:
        return self._confidence_computer.compute_sample_size(
            treatment_weights, mde_column, nim_column, preferred_direction_column, final_expected_sample_size_column
        )
