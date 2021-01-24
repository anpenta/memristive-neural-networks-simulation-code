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
#
# Spiking Neuron Module
# Model of a spiking neuron.
# Voltage is measured in mV and time in ms.


class SpikingNeuron:

  def __init__(self):
    self._spike = 1
    self._no_spike = 0
    self._threshold_voltage = 30

    self._a = 0.02
    self._b = 0.2
    self._c = -65
    self._d = 8
    self._v = self._c
    self._u = self._b * self._v
    self._step = 0.1

  # apply_current: Simulates the application of the given current to
  # the neuron for the given period of time.
  def apply_current(self, current, time):
    timesteps = int(time / self._step)

    spikes = []
    voltage = []
    for _ in range(timesteps):
      self._v += self._step * (0.04 * self._v ** 2 + 5 * self._v + 140 - self._u + current)
      self._u += self._step * self._a * (self._b * self._v - self._u)
      voltage.append(self._v)

      if self._v >= self._threshold_voltage:
        spikes.append(self._spike)
        self._v = self._c
        self._u += self._d
      else:
        spikes.append(self._no_spike)

    return spikes, voltage

  # rest: Simulates a relatively long period of time without application
  # of current to the neuron by resetting v and u.
  def rest(self):
    self._v = self._c
    self._u = self._b * self._v
