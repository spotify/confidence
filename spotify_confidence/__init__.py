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

from pkg_resources import require as _require
from .analysis.bayesian.bayesian_models import BetaBinomial
from spotify_confidence.analysis.frequentist.chi_squared import ChiSquared
from spotify_confidence.analysis.frequentist.t_test import StudentsTTest
from spotify_confidence.analysis.frequentist.z_test import ZTest
from spotify_confidence.analysis.frequentist.z_test_linreg import ZTestLinreg
from spotify_confidence.analysis.frequentist.experiment import Experiment
from spotify_confidence.analysis.frequentist.sample_size_calculator import SampleSizeCalculator
from .samplesize.sample_size_calculator import SampleSize

from . import examples
from .options import options

__version__ = _require("spotify_confidence")[0].version

__all__ = [
    "BetaBinomial",
    "ChiSquared",
    "StudentsTTest",
    "ZTest",
    "ZTestLinreg",
    "Experiment",
    "SampleSizeCalculator",
    "examples",
    "options",
    "SampleSize",
]
