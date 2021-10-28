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

from collections import OrderedDict


class ConfidenceOptions(object):
    def __init__(self):
        self._options = OrderedDict(
            {
                "randomization_seed": OptionValue(None),
            }
        )

    def get_option(self, option_name):
        """Return the value of the given option"""
        return self._options[option_name].value

    def set_option(self, option_name, option_value):
        """Set the default value of the specified option.

        Available options:
            'randomization_seed': (int)
                Seed to use for methods that involve monte carlo sampling.
        """
        self._options[option_name].value = option_value


class OptionValue(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "%s" % self.value


options = ConfidenceOptions()
