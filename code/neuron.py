# Copyright (C) 2019 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Neuron Module
# Model of an abstraction of a neuron.


class Neuron:

  def __init__(self):
    self._spike = 1
    self._no_spike = 0
    self._threshold_current = 10 ** -5

  def apply_current(self, current):
    return self._spike if current > self._threshold_current else self._no_spike
