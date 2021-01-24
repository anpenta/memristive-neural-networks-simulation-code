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

# Pulses Module
# Functions to simulate the application of voltage pulses to memristors.

import matplotlib.pyplot as plt
import pandas as pd

import memristor
import utility

plt.rcParams.update({"font.size": 12})


# simulate_pulsing_experiment: Simulates the application of 10 cycles
# of 20 voltage pulses on a memristor.
# Before each cycle the resistance is set to a low resistance state
# by 2 positive bias voltage sweeps of 1 V.
# The first 10 pulses in each cycle are negative bias voltage pulses of -4 V.
# The last 10 pulses in each cycle are positive bias voltage pulses that
# range from 1 V to 0.1 V, each time decreasing by 0.1 V.
# The plot shows the resistance we get with a -1 V read pulse after each of
# the 20 pulses is applied.
def simulate_pulsing_experiment(output_to_csv=False):
  half_cycle_pulses = 10
  full_cycle_pulses = [x for x in range(1, half_cycle_pulses * 2 + 1)]
  positive_bias = [x * 0.1 for x in range(10, 0, -1)]
  negative_bias = -4
  lrs_set_voltage = 1

  device = memristor.Memristor()

  resistance = []
  # Perform the experiment.
  for voltage in positive_bias:
    device.set_resistance(lrs_set_voltage)

    for _ in full_cycle_pulses[1: half_cycle_pulses + 1]:
      device.apply_voltage(negative_bias)
      resistance.append(device.read_resistance())

    for _ in full_cycle_pulses[half_cycle_pulses:]:
      device.apply_voltage(voltage)
      resistance.append(device.read_resistance())

  # Store the data into a data frame.
  data = pd.DataFrame()
  data["positive_bias"] = (positive_bias * len(full_cycle_pulses))
  data.sort_values(by="positive_bias", ascending=False, inplace=True)
  data.reset_index(inplace=True, drop=True)
  data["resistance"] = resistance
  data["pulse_number"] = full_cycle_pulses * len(positive_bias)

  if output_to_csv:
    utility.save_data(data, "./output", "simulate-pulsing-experiment")

  # Plot the results.
  ax = plt.subplots(figsize=(10, 7))[1]
  data.groupby(["positive_bias"]).plot(x="pulse_number", y="resistance", ax=ax)
  plt.legend(data["positive_bias"].unique()[::-1], title="Positive bias (V)")
  plt.xlabel("Pulse number")
  plt.ylabel("Resistance (Ω)")
  plt.show()


# simulate_pulses: Simulates the application of 1 cycle of the given number
# of voltage pulses on a memristor.
# All pulses have the given positive bias or negative bias voltage value.
# The plot shows the resistance we get with a -1 V read pulse after each of
# the pulses is applied.
def simulate_pulses(voltage, pulses, output_to_csv=False):
  full_cycle_pulses = [x for x in range(1, pulses + 1)]

  device = memristor.Memristor()

  resistance = []
  # Perform the experiment.
  for _ in full_cycle_pulses:
    device.apply_voltage(voltage)
    resistance.append(device.read_resistance())

  # Store the data into a data frame.
  data = pd.DataFrame()
  data["voltage"] = [voltage] * len(full_cycle_pulses)
  data["resistance"] = resistance
  data["pulse_number"] = full_cycle_pulses

  if output_to_csv:
    utility.save_data(data, "./output", "simulate-pulses")

  # Plot the results.
  ax = plt.subplots(figsize=(10, 7))[1]
  data.plot(x="pulse_number", y="resistance", ax=ax)
  plt.legend(data["voltage"].unique(), title="Voltage (V)")
  plt.xlabel("Pulse number")
  plt.ylabel("Resistance (Ω)")
  plt.show()
