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

# Memristor Module
# Model of a memristor.


class Memristor:

  def __init__(self):
    self._least_resistance = 100
    self._greatest_resistance = 2.5 * 10 ** 8
    self._resistance = 10 ** 7

  def read_resistance(self):
    # Ideally a read pulse should not change the resistance.
    return self._resistance

  # set_resistance: Sets the resistance by simulating the continuous
  # application of long sweeps of the given voltage to the memristor.
  def set_resistance(self, voltage):
    if voltage >= 0:
      self._resistance = self._least_resistance
    else:
      self._resistance = self._greatest_resistance

  # apply_voltage: Applies the given voltage pulse to the memristor
  # and changes its resistance.
  def apply_voltage(self, voltage):
    if voltage >= 0:
      R1 = self._greatest_resistance
      c = -0.128 - 0.522 * voltage
    else:
      # R1 is now a magic number that produces the expected range of values.
      # The power law exponent c is now a constant (we have data only for -4 V
      # negative bias pulses, so we do not fit a linear equation).
      R1 = 1.3 * 10 ** 8
      c = 0.25

    # Calculate the previous pulse number n.
    n = ((self._resistance - self._least_resistance) / R1) ** (1 / c)

    # Calculate the new resistance for n + 1.
    self._resistance = self._least_resistance + R1 * (n + 1) ** c
