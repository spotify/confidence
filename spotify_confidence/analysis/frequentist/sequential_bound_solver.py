# Copyright 2017-2020 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas
from scipy.stats import norm


def _alphas(alpha: np.array, phi: float, t: np.array):
    """Alpha spending function."""
    pe = np.zeros(len(t))
    pd = np.zeros(len(t))
    for j, tt in enumerate(t):
        pe[j] = alpha * np.power(tt, phi)
        pd[j] = pe[j] if j == 0 else pe[j] - pe[j - 1]
    return pe, pd


def _qp(xq: float, last: float, nints: int, yam1: float, ybm1: float, stdv: float):
    hlast = (ybm1 - yam1) / nints
    grid = np.linspace(yam1, ybm1, nints + 1)
    fun = last * norm.cdf(grid, xq, stdv)
    qp = 0.5 * hlast * (2 * np.sum(fun) - fun[0] - fun[len(fun) - 1])
    return qp


def _bsearch(
    last: np.array,
    nints: int,
    pd: float,
    stdv: float,
    ya: float,
    yb: float,
) -> np.array:
    """
    Note: function signature slightly modified in comparison to R implementation (which takes complete nints
    array instead of scalar), but should be semantically equivalent
    """
    max_iter = 50
    tol = 1e-7
    de = 10
    uppr = yb
    q = _qp(uppr, last, nints, ya, yb, stdv)
    while abs(q - pd) > tol:
        de = de / 10
        temp = 1 if q > (pd + tol) else 0
        incr = 2 * temp - 1
        j = 1
        while j <= max_iter:
            uppr = uppr + incr * de
            q = _qp(uppr, last, nints, ya, yb, stdv)
            if abs(q - pd) > tol and j >= max_iter:
                break
            elif (incr == 1 and q <= (pd + tol)) or (incr == -1 and q >= (pd - tol)):
                j = max_iter

            j += 1
    ybval = uppr
    return ybval


_NORM_CONSTANT = 1 / np.sqrt(2 * np.pi)


def _fast_norm_pdf_prescaled(x: np.array, scale):
    norm_constant2 = _NORM_CONSTANT / scale
    pdf_val = norm_constant2 * np.exp(-0.5 * np.power(x, 2))
    return pdf_val


def _fcab(last: np.array, nints: int, yam1: float, h: float, x: np.array, stdv: float):
    X, Y = np.meshgrid(x / stdv, (h * np.linspace(0, nints, nints + 1) + yam1) / stdv)
    scaled_x = Y - X
    pdf_prescaled = _fast_norm_pdf_prescaled(scaled_x, stdv)
    last_transposed = np.transpose(np.tile(last, len(x)).reshape(len(x), nints + 1))

    f = last_transposed * pdf_prescaled
    area = 0.5 * h * (2 * f.sum(0) - np.transpose(f[0, :]) - np.transpose(f[nints, :]))
    return area


# TODO use dataclass as soon as stets was migrated to Python 3.7
class ComputationState:
    """
    Internal state that can be fed into bounds(...). Whenever the internal state changes, a new ComputationState object
    will be created.

    It is not intended that other packages code operates on the attributes of this class because the internal
    structure may be changed anytime.
    """

    def __init__(self, df: pandas.DataFrame, last_fcab: np.array):
        if df is None or any(df["zb"].isnull()) or len(df) > 0 and last_fcab is None:
            raise ValueError()

        self._df = df
        self._last_fcab = last_fcab

    @property
    def df(self):
        # copy to avoid side effects
        return self._df.copy()

    @property
    def last_fcab(self):
        """fcab calculation referring to the last row of df"""

        # copy to avoid side effects
        return None if self._last_fcab is None else np.copy(self._last_fcab)

    def __eq__(self, other):
        if isinstance(other, ComputationState):
            return self._df.equals(other._df) and np.array_equal(self._last_fcab, other._last_fcab)
        return False


def landem(
    t: np.array,
    alpha: float,
    phi: float,
    ztrun: float,
    state: ComputationState,
    max_nints: int = None,
):
    """
    This function is a Python implementation of landem.R of ldbounds package.
    https://cran.r-project.org/web/packages/ldbounds/index.html
    Source code of that landem.R: https://github.com/cran/ldbounds/blob/master/R/landem.R

    After making any changes, please run test_compare_with_ldbounds.py to gain confidence that functionality is
    not broken.

    :param t: Monotonically increasing information ratios
    :param alpha: corrected alpha (other than R implementation, we do not modify alpha based on number of sides)
    :param phi: exponent used by alpha-sepending function
    :param ztrun: max value for truncating bounds
    :param state: state to build the computation upon
    :param max_nints: max value that internal nints parameter can take. Limiting this value reduces accuracy of the
        calculation but can lead to crucial performance improvement
    :return: A dataframe where the "zb" column contains the bounds and the i-th row reflects the results
    for the i-th information ratio from t
    """

    df = state.df  # reading the property will copy the df to avoid side effects
    last_fcab = state.last_fcab

    if len(t) <= len(df):
        # Simply return complete state and the existing result
        return df.iloc[: len(t)], state
    elif len(t) > len(df):
        # We reindex because appending rows *individually* to a DataFrame is expensive
        df = df.reindex(range(len(t)))

    h = 0.05
    zninf = -ztrun
    tol = 1e-7

    # t2 = t  # ldbounds:::bounds() rescales t2=t/t.max() by default. We omit this because impact on bounds unclear

    if df.isnull().all().all():
        # start at index 0 if df was not yet initialized
        start = 0
    else:
        # start at the first index where "zb" column is null (or at the very end if all "zb" values are not null)
        zb_null_arr = np.where(df["zb"].isnull())
        start = zb_null_arr[0][0] - 1 if len(zb_null_arr[0]) > 0 else len(df) - 1

    rangestart = start + 1
    if start == 0:
        df.loc[0, "stdv"] = np.sqrt(t[0])

    df.loc[start + 1 : len(t), "stdv"] = np.sqrt(t[start + 1 : len(t)] - t[start : len(t) - 1])

    df["pe"], df["pd"] = _alphas(alpha, phi, t)
    df.loc[start:, "sdproc"] = np.sqrt(t[start:])
    df.loc[start:, "information_ratio"] = t[start:]

    if df.isnull().all(axis=0)[0]:
        # this needs to be done only to compute the very first row
        if df.at[start, "pd"] < 1:
            df.at[start, "zb"] = norm.ppf(1 - df.at[start, "pd"])
            if df.at[start, "zb"] > ztrun:
                df.at[start, "zb"] = ztrun
                df.at[start, "pd"] = 1 - norm.cdf(df.at[start, "zb"])
                df.at[start, "pe"] = df.at[start, "pd"]
                if len(t) > 1:
                    df.at[1, "pd"] = df.at[start + 1, "pe"] - df.at[start, "pe"]
            df.at[start, "yb"] = df.at[start, "zb"] * df.at[start, "stdv"]

        df.at[start, "za"] = zninf
        df.at[start, "ya"] = df.at[start, "za"] * df.at[start, "stdv"]
        df.at[start, "nints"] = np.ceil((df.at[start, "yb"] - df.at[start, "ya"]) / (h * df.at[start, "stdv"]))

        grid = np.linspace(
            df.at[start, "ya"],
            df.at[start, "yb"],
            int(df.at[start, "nints"] + 1),
        )
        scaled_x = grid / df.at[start, "stdv"]
        last_fcab = _fast_norm_pdf_prescaled(scaled_x, df.at[start, "stdv"])

    if len(t) >= 2:
        for i in range(rangestart, len(t)):
            if df["information_ratio"][i] - df["information_ratio"][i - 1] <= 1e-5:
                # If information ratio difference between time steps is 0, re-use result calculated for the previous
                # time step. Normally, it means that no data was added. We have to catch this case because nints
                # becomes float("inf") and makes the procedure crash. We check against 10e-6 instead of against 0
                # because an almost-zero information gain can cause pretty big numerical inaccuracy in practice.
                df.iloc[i] = df.iloc[i - 1]
                continue

            # Possible error in spending function.  May be due to truncation.
            if df.at[i, "pd"] != 1.0:
                df.at[i, "pd"] = df.at[i, "pe"] - df.at[i - 1, "pe"]
            df.at[i, "pd"] = df.at[i, "pd"].clip(0, 1)

            if df.at[i, "pd"] < tol:
                df.at[i, "zb"] = -zninf
                if df.at[i, "zb"] > ztrun:
                    df.at[i, "zb"] = ztrun
                    df.at[i, "pd"] = _qp(
                        df.at[i, "zb"] * df.at[i, "sdproc"],
                        last_fcab,
                        df.at[i - 1, "nints"],
                        df.at[i - 1, "ya"],
                        df.at[i - 1, "yb"],
                        df.at[i, "stdv"],
                    )
                    df.at[i, "pe"] = df.at[i, "pd"] + df.at[i - 1, "pe"]

                df.at[i, "yb"] = df.at[i, "zb"] * df.at[i, "sdproc"]
            elif df.at[i, "pd"] == 1.0:
                df.at[i, "zb"] = 0.0
                df.at[i, "yb"] = 0.0
            elif tol <= df.at[i, "pd"] < 1:

                df.at[i, "yb"] = _bsearch(
                    last_fcab,
                    int(df.loc[i - 1]["nints"]),  # differs from R because we modified signature of bsearch
                    df.at[i, "pd"],
                    df.at[i, "stdv"],
                    df.at[i - 1, "ya"],
                    df.at[i - 1, "yb"],
                )

                df.at[i, "zb"] = df.at[i, "yb"] / df.at[i, "sdproc"]

                if df.at[i, "zb"] > ztrun:
                    df.at[i, "zb"] = ztrun
                    df.at[i, "pd"] = _qp(
                        df.at[i, "zb"] * df.at[i, "sdproc"],
                        last_fcab,
                        int(df.at[i - 1, "nints"]),
                        df.at[i - 1, "ya"],
                        df.at[i - 1, "yb"],
                        df.at[i, "stdv"],
                    )
                    df.at[i, "pe"] = df.at[i, "pd"] + df.at[i - 1, "pe"]

                df.at[i, "yb"] = df.at[i, "zb"] * df.at[i, "sdproc"]

            # in landem.R, the following two statements are in side==1 if clause
            df.at[i, "ya"] = zninf * df.at[i, "sdproc"]
            df.at[i, "za"] = zninf

            nints_calc = np.ceil((df.at[i, "yb"] - df.at[i, "ya"]) / (h * df.at[i, "stdv"]))
            df.at[i, "nints"] = nints_calc if max_nints is None or nints_calc < max_nints else max_nints

            if i < len(t):
                # in R implementation, i < len(t)-1. However we run until len(t) because that calculation will be
                # required if landem() is called again with df used as a starting point
                hlast = (df.at[i - 1, "yb"] - df.at[i - 1, "ya"]) / df.at[i - 1, "nints"]
                x = np.linspace(
                    df.at[i, "ya"],
                    df.at[i, "yb"],
                    int(df.at[i, "nints"] + 1),
                )
                last_fcab = _fcab(
                    last_fcab, int(df.at[i - 1, "nints"]), df.at[i - 1, "ya"], hlast, x, df.at[i, "stdv"]
                )
    return df, ComputationState(df, last_fcab)


# Simple type to return results in a structured way
class CalculationResult:
    def __init__(self, df: pandas.DataFrame, state: ComputationState):
        self._df = df
        self._state = state

    @property
    def df(self):
        return self._df

    @property
    def bounds(self):
        return self._df["zb"].values

    @property
    def state(self):
        return self._state


columns = ["za", "zb", "ya", "yb", "pe", "pd", "stdv", "sdproc", "nints", "information_ratio"]

# Initial state to be fed into bounds() to calculate sequential bounds from scratch
EMPTY_STATE = ComputationState(df=pandas.DataFrame(index=None, columns=columns, dtype=float), last_fcab=None)


def bounds(
    t: np.array,
    alpha: float,
    rho: float,
    ztrun: float,
    sides: int,
    state: ComputationState = EMPTY_STATE,
    max_nints=None,
) -> CalculationResult:
    """
    See landem() for parameter explanation

    :return: If a state is provided, returns a tuple of result and state. Otherwise, return only boundary result.
    """

    def get_input_str():
        return (
            f"input params: t={t}, alpha={alpha}, sides={sides}, rho={rho}, ztrun={ztrun},"
            f"state_df={state.df.to_json()}, state_fcab={state.last_fcab}, max_nints={max_nints}"
        )

    if any(t == 0.0):
        raise ValueError(f"Information ratio must must not be zero, {get_input_str()}")
    if any(t[i] > t[i + 1] for i in range(len(t) - 1)):
        raise ValueError(f"Information ratio must be monotonically increasing, {get_input_str()}")
    if not (sides == 1 or sides == 2):
        raise ValueError(f"sides must either be one a zero, {get_input_str()}")

    if state is None:
        state = EMPTY_STATE

    alph = alpha / sides

    df_result, new_state = landem(t, alph, rho, ztrun, state, max_nints)

    # guardrail check
    fixed_horizon_bound = norm.ppf(1 - alph)
    last_sequential_bound = df_result["zb"].values[-1]
    if fixed_horizon_bound > last_sequential_bound:
        raise Exception(
            f"Last bound ({last_sequential_bound}) is less conservative than fixed horizon bound "
            f"({fixed_horizon_bound}), {get_input_str()} "
        )

    return CalculationResult(df_result, new_state)
