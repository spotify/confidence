from numpy import isnan
from pandas import DataFrame

from spotify_confidence.analysis.confidence_utils import listify
from spotify_confidence.analysis.constants import (
    NIM_TYPE,
    NIM_COLUMN_DEFAULT,
    PREFERRED_DIRECTION_COLUMN_DEFAULT,
    ORIGINAL_POINT_ESTIMATE,
    TWO_SIDED,
    INCREASE_PREFFERED,
    DECREASE_PREFFERED,
    NIM,
    PREFERENCE,
    NULL_HYPOTHESIS,
    ALTERNATIVE_HYPOTHESIS,
)


def add_nim_input_columns_from_tuple_or_dict(df, nims: NIM_TYPE, mde_column: str) -> DataFrame:
    if type(nims) is tuple:
        return df.assign(**{NIM_COLUMN_DEFAULT: nims[0]}).assign(**{PREFERRED_DIRECTION_COLUMN_DEFAULT: nims[1]})
    elif type(nims) is dict:
        nim_values = {key: value[0] for key, value in nims.items()}
        nim_preferences = {key: value[1] for key, value in nims.items()}
        return df.assign(**{NIM_COLUMN_DEFAULT: lambda df: df.index.to_series().map(nim_values)}).assign(
            **{PREFERRED_DIRECTION_COLUMN_DEFAULT: lambda df: df.index.to_series().map(nim_preferences)}
        )
    elif nims is None or not nims:
        return df.assign(**{NIM_COLUMN_DEFAULT: None}).assign(
            **{
                PREFERRED_DIRECTION_COLUMN_DEFAULT: None
                if PREFERRED_DIRECTION_COLUMN_DEFAULT not in df or mde_column is None
                else df[PREFERRED_DIRECTION_COLUMN_DEFAULT]
            }
        )
    else:
        return df


def add_nims_and_mdes(
    df: DataFrame,
    mde_column: str,
    nim_column: str,
    preferred_direction_column: str,
) -> DataFrame:
    def _set_nims_and_mdes(grp: DataFrame) -> DataFrame:
        nim = grp[nim_column].astype(float)
        input_preference = grp[preferred_direction_column].values[0]
        mde = None if mde_column is None else grp[mde_column]

        nim_is_na = nim.isna().all()
        mde_is_na = True if mde is None else mde.isna().all()
        if input_preference is None or (type(input_preference) is float and isnan(input_preference)):
            signed_nim = 0.0 if nim_is_na else nim * grp[ORIGINAL_POINT_ESTIMATE]
            preference = TWO_SIDED
            signed_mde = None if mde_is_na else mde * grp[ORIGINAL_POINT_ESTIMATE]
        elif input_preference.lower() == INCREASE_PREFFERED:
            signed_nim = 0.0 if nim_is_na else -nim * grp[ORIGINAL_POINT_ESTIMATE]
            preference = "larger"
            signed_mde = None if mde_is_na else mde * grp[ORIGINAL_POINT_ESTIMATE]
        elif input_preference.lower() == DECREASE_PREFFERED:
            signed_nim = 0.0 if nim_is_na else nim * grp[ORIGINAL_POINT_ESTIMATE]
            preference = "smaller"
            signed_mde = None if mde_is_na else -mde * grp[ORIGINAL_POINT_ESTIMATE]
        else:
            raise ValueError(f"{input_preference.lower()} not in " f"{[INCREASE_PREFFERED, DECREASE_PREFFERED]}")

        return (
            grp.assign(**{NIM: nim})
            .assign(**{PREFERENCE: preference})
            .assign(**{NULL_HYPOTHESIS: signed_nim})
            .assign(**{ALTERNATIVE_HYPOTHESIS: signed_mde if nim_is_na else 0.0})
        )

    index_names = [name for name in df.index.names if name is not None]
    return (
        df.groupby(
            [nim_column, preferred_direction_column] + listify(mde_column),
            dropna=False,
            as_index=False,
            sort=False,
            group_keys=True,
        )
        .apply(_set_nims_and_mdes)
        .pipe(lambda df: df.reset_index(index_names))
        .reset_index(drop=True)
        .pipe(lambda df: df if index_names == [] else df.set_index(index_names))
    )
